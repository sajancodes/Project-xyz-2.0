from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def validate_model_dir(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if path.is_file():
        raise ValueError(
            "Pass the model DIRECTORY, not model.safetensors. "
            "Transformers also needs config/tokenizer files."
        )
    return path


def load_model_and_tokenizer(model_path: str, use_4bit: bool = True):
    model_dir = validate_model_dir(model_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. These experiments are intended for CUDA.")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }

    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def get_transformer_layers(model):
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
    ]
    for _, getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                return layers
        except Exception:
            pass
    raise RuntimeError(
        "Could not locate transformer layers. Run print(model) and extend get_transformer_layers()."
    )


def get_mlp_probe_target(layer):
    """Return a module whose INPUT is the MLP intermediate representation."""
    if not hasattr(layer, "mlp"):
        return None
    mlp = layer.mlp
    for name in ("down_proj", "c_proj", "dense_4h_to_h", "proj"):
        if hasattr(mlp, name):
            return getattr(mlp, name)
    return None


def build_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"USER: {user_text}\nASSISTANT:"


def load_prompt_set(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Prompt file must be a JSON object: {category: [prompts...]}")
    out: Dict[str, List[str]] = {}
    for category, prompts in data.items():
        if not isinstance(prompts, list):
            raise ValueError(f"Category {category!r} must contain a list of prompts")
        out[str(category)] = [str(x) for x in prompts]
    return out


def iter_prompts(prompt_set: Dict[str, List[str]], limit_per_category: int = 0):
    for category, prompts in prompt_set.items():
        selected = prompts if limit_per_category <= 0 else prompts[:limit_per_category]
        for idx, prompt in enumerate(selected):
            yield category, idx, prompt


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def cuda_memory_gb() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.max_memory_allocated() / 1024**3,
    )
