from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from tqdm import tqdm

from neural_map_common import (
    build_chat_prompt,
    get_mlp_probe_target,
    get_transformer_layers,
    load_model_and_tokenizer,
    load_prompt_set,
    model_input_device,
)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_bounds(width: int, groups: int, group_id: int):
    bounds = np.linspace(0, width, groups + 1, dtype=int)
    return int(bounds[group_id]), int(bounds[group_id + 1])


def candidate_groups(profile, target_category: str, top_n: int):
    categories = profile["categories"]
    target = categories[target_category]["layers"]
    all_names = list(categories.keys())
    groups = int(profile["groups"])
    scored = []

    for layer_id, info in target.items():
        target_values = info["group_mean_abs"]
        for group_id in range(min(groups, len(target_values))):
            t = float(target_values[group_id])
            others = []
            for name in all_names:
                if name == target_category:
                    continue
                layer = categories[name]["layers"].get(layer_id)
                if layer and group_id < len(layer["group_mean_abs"]):
                    others.append(float(layer["group_mean_abs"][group_id]))
            baseline = mean(others) if others else 0.0
            # Specificity plus a small absolute-activation term so tiny noisy ratios do not dominate.
            specificity = (t + 1e-8) / (baseline + 1e-8)
            score = math.log1p(max(t, 0.0)) * specificity
            scored.append(
                {
                    "layer": int(layer_id),
                    "group": group_id,
                    "width": int(info["width"]),
                    "activation": t,
                    "other_activation": baseline,
                    "specificity_score": score,
                }
            )

    scored.sort(key=lambda x: x["specificity_score"], reverse=True)
    return scored[:top_n]


def prompt_nll(model, tokenizer, text: str, max_length: int):
    prompt = build_chat_prompt(tokenizer, text)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    device = model_input_device(model)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Self-NLL is used as a stable causal sensitivity metric. It does not claim task accuracy.
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        use_cache=False,
    )
    return float(out.loss.detach().float().item())


def install_group_mask(layer, start: int, end: int):
    target = get_mlp_probe_target(layer)
    if target is None:
        raise RuntimeError("Selected layer has no recognized MLP probe target")

    def pre_hook(module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        x = inputs[0]
        y = x.clone()
        y[..., start:end] = 0
        return (y,) + tuple(inputs[1:])

    return target.register_forward_pre_hook(pre_hook)


def evaluate_prompts(model, tokenizer, prompts, max_length):
    values = []
    with torch.inference_mode():
        for text in prompts:
            values.append(prompt_nll(model, tokenizer, text, max_length))
    return values


def main():
    parser = argparse.ArgumentParser(description="Causally ablate high-signal MLP groups.")
    parser.add_argument("--model", required=True, help="Local Hugging Face model DIRECTORY")
    parser.add_argument("--profile", default="artifacts/profile.json")
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--target", required=True, help="Category to map, e.g. coding")
    parser.add_argument("--output", default=None)
    parser.add_argument("--top-candidates", type=int, default=12)
    parser.add_argument("--limit-per-category", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    profile = load_json(args.profile)
    prompt_set = load_prompt_set(args.prompts)
    if args.target not in prompt_set:
        raise ValueError(f"Unknown target category {args.target!r}; available: {list(prompt_set)}")

    candidates = candidate_groups(profile, args.target, args.top_candidates)
    if not candidates:
        raise RuntimeError("No candidate groups were found in the profile")

    output = args.output or f"artifacts/ablation_{args.target}.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("[Ablation] Loading model...")
    tokenizer, model = load_model_and_tokenizer(args.model, use_4bit=not args.no_4bit)
    layers = get_transformer_layers(model)

    target_prompts = prompt_set[args.target][: args.limit_per_category]
    control_prompts = []
    for name, prompts in prompt_set.items():
        if name != args.target:
            control_prompts.extend(prompts[: max(1, args.limit_per_category // max(1, len(prompt_set) - 1))])

    print("[Ablation] Computing baseline losses...")
    target_baseline = evaluate_prompts(model, tokenizer, target_prompts, args.max_length)
    control_baseline = evaluate_prompts(model, tokenizer, control_prompts, args.max_length)
    baseline_target_mean = mean(target_baseline)
    baseline_control_mean = mean(control_baseline) if control_baseline else 0.0

    results = []
    for candidate in tqdm(candidates, desc="Ablating candidates"):
        layer_id = candidate["layer"]
        group_id = candidate["group"]
        width = candidate["width"]
        start, end = group_bounds(width, int(profile["groups"]), group_id)
        handle = install_group_mask(layers[layer_id], start, end)
        try:
            target_losses = evaluate_prompts(model, tokenizer, target_prompts, args.max_length)
            control_losses = evaluate_prompts(model, tokenizer, control_prompts, args.max_length)
        finally:
            handle.remove()

        target_mean = mean(target_losses)
        control_mean = mean(control_losses) if control_losses else 0.0
        target_delta = target_mean - baseline_target_mean
        control_delta = control_mean - baseline_control_mean

        # Positive means the group hurts target NLL more than controls when removed.
        causal_specificity = target_delta - control_delta
        results.append(
            {
                **candidate,
                "start": start,
                "end": end,
                "baseline_target_nll": baseline_target_mean,
                "ablated_target_nll": target_mean,
                "target_delta_nll": target_delta,
                "baseline_control_nll": baseline_control_mean,
                "ablated_control_nll": control_mean,
                "control_delta_nll": control_delta,
                "causal_specificity": causal_specificity,
            }
        )

    results.sort(key=lambda x: x["causal_specificity"], reverse=True)
    payload = {
        "schema_version": 1,
        "target": args.target,
        "groups": int(profile["groups"]),
        "candidate_count": len(results),
        "metric": "delta self-NLL under contiguous MLP-group ablation",
        "warning": "This measures causal sensitivity, not benchmark correctness. Validate useful circuits with task-specific accuracy later.",
        "results": results,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Ablation] Wrote {output}")
    print("Large positive causal_specificity is evidence that a group matters more for the target than controls.")


if __name__ == "__main__":
    main()
