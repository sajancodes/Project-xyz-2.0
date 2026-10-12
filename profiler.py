from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from neural_map_common import (
    build_chat_prompt,
    cuda_memory_gb,
    get_mlp_probe_target,
    get_transformer_layers,
    iter_prompts,
    load_model_and_tokenizer,
    load_prompt_set,
    model_input_device,
)


class ActivationProfiler:
    """Profiles MLP intermediate activations as contiguous groups, not scalar weights."""

    def __init__(self, model, groups: int = 16):
        self.model = model
        self.groups = groups
        self.layers = get_transformer_layers(model)
        self.handles = []
        self.current = {}
        self._install_hooks()

    @staticmethod
    def _group_mean_abs(x: torch.Tensor, groups: int):
        # x is usually [batch, sequence, intermediate_width].
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim >= 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim == 2:
            pass
        else:
            return []

        feature_mean = x.detach().float().abs().mean(dim=0)
        width = int(feature_mean.numel())
        groups = max(1, min(groups, width))

        boundaries = np.linspace(0, width, groups + 1, dtype=int)
        values = []
        for i in range(groups):
            start, end = int(boundaries[i]), int(boundaries[i + 1])
            if end <= start:
                values.append(0.0)
            else:
                values.append(float(feature_mean[start:end].mean().item()))
        return values

    def _make_hook(self, layer_id: int):
        def hook(module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            x = inputs[0]
            values = self._group_mean_abs(x, self.groups)
            if values:
                self.current[layer_id] = {
                    "width": int(x.shape[-1]),
                    "group_mean_abs": values,
                }
        return hook

    def _install_hooks(self):
        attached = 0
        for layer_id, layer in enumerate(self.layers):
            target = get_mlp_probe_target(layer)
            if target is None:
                continue
            self.handles.append(target.register_forward_pre_hook(self._make_hook(layer_id)))
            attached += 1
        if attached == 0:
            raise RuntimeError("No MLP probe targets were recognized in this model.")
        print(f"[Profiler] Attached hooks to {attached}/{len(self.layers)} transformer layers.")

    def reset(self):
        self.current = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main():
    parser = argparse.ArgumentParser(description="Profile task-specific MLP activation groups.")
    parser.add_argument("--model", required=True, help="Local Hugging Face model DIRECTORY")
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--output", default="artifacts/profile.json")
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prompt_set = load_prompt_set(args.prompts)

    print("[Profiler] Loading model...")
    tokenizer, model = load_model_and_tokenizer(args.model, use_4bit=not args.no_4bit)
    layers = get_transformer_layers(model)
    profiler = ActivationProfiler(model, groups=args.groups)
    device = model_input_device(model)

    sums = defaultdict(lambda: defaultdict(lambda: None))
    counts = defaultdict(lambda: defaultdict(int))
    widths = {}
    sample_records = []

    jobs = list(iter_prompts(prompt_set, args.limit_per_category))
    started = time.perf_counter()

    try:
        with torch.inference_mode():
            for category, prompt_index, user_text in tqdm(jobs, desc="Profiling"):
                profiler.reset()
                prompt = build_chat_prompt(tokenizer, user_text)
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(**encoded, use_cache=False)
                torch.cuda.synchronize()
                latency = time.perf_counter() - t0

                for layer_id, info in profiler.current.items():
                    arr = np.asarray(info["group_mean_abs"], dtype=np.float64)
                    if sums[category][layer_id] is None:
                        sums[category][layer_id] = np.zeros_like(arr)
                    sums[category][layer_id] += arr
                    counts[category][layer_id] += 1
                    widths[layer_id] = info["width"]

                alloc, peak = cuda_memory_gb()
                sample_records.append(
                    {
                        "category": category,
                        "prompt_index": prompt_index,
                        "tokens": int(encoded["input_ids"].shape[-1]),
                        "latency_s": latency,
                        "vram_gb": alloc,
                        "peak_vram_gb": peak,
                    }
                )
    finally:
        profiler.close()

    categories = {}
    for category in prompt_set:
        layer_data = {}
        for layer_id in range(len(layers)):
            n = counts[category].get(layer_id, 0)
            total = sums[category].get(layer_id)
            if n and total is not None:
                mean = (total / n).tolist()
                layer_data[str(layer_id)] = {
                    "width": int(widths[layer_id]),
                    "samples": int(n),
                    "group_mean_abs": [float(x) for x in mean],
                }
        categories[category] = {"layers": layer_data}

    result = {
        "schema_version": 1,
        "model": str(Path(args.model).expanduser()),
        "groups": args.groups,
        "layer_count": len(layers),
        "elapsed_s": time.perf_counter() - started,
        "categories": categories,
        "samples": sample_records,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[Profiler] Wrote {args.output}")
    print("Next: run ablation.py. Activation correlation alone is not a causal map.")


if __name__ == "__main__":
    main()
