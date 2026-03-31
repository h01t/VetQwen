"""
Shared inference loading and generation helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from vetqwen_core.constants import SYSTEM_PROMPT

SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX = (3, 12)


def resolve_device(device: str) -> str:
  requested = device.casefold()
  if requested not in {"auto", "cpu", "cuda"}:
    raise ValueError(f"Unsupported device '{device}'. Use auto, cpu, or cuda.")

  if requested == "cpu":
    return "cpu"
  if requested == "cuda":
    return "cuda"

  try:
    import torch
  except ImportError:
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def load_inference_model(
  model_path: str,
  base_model: str | None = None,
  device: str = "cuda",
  enforce_supported_python: bool = False,
  remote_hint: str | None = None,
):
  resolved_device = resolve_device(device)

  if enforce_supported_python and not (
    SUPPORTED_PYTHON_MIN <= sys.version_info[:2] <= SUPPORTED_PYTHON_MAX
  ):
    message = (
      "Unsupported Python runtime for VetQwen inference.\n"
      f"Detected Python {sys.version_info.major}.{sys.version_info.minor}, "
      "but the pinned stack supports Python 3.10-3.12 and recommends 3.11."
    )
    if remote_hint:
      message += f"\n{remote_hint}"
    raise SystemExit(message)

  try:
    import torch
  except ImportError as exc:
    raise SystemExit(
      "Missing dependency: torch.\n"
      "Sync the project environment first, for example:\n"
      " uv sync --group demo --locked --python 3.11 "
    ) from exc

  try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # type: ignore
  except ImportError as exc:
    raise SystemExit(
      "Missing dependency: transformers.\n"
      "Sync the required dependencies with uv first."
    ) from exc

  if resolved_device == "cuda":
    if importlib.util.find_spec("accelerate") is None:
      message = (
        "Missing dependency: accelerate.\n"
        "Sync the remote research dependencies with:\n"
        " uv sync --group research --locked --python 3.11 "
      )
      if remote_hint:
        message += f"\n{remote_hint}"
      raise SystemExit(message)
    if not torch.cuda.is_available():
      message = "CUDA is not available in the current environment."
      if remote_hint:
        message += f"\n{remote_hint}"
      raise SystemExit(message)

  is_local_adapter = Path(model_path).exists() and Path(model_path).is_dir()
  tokenizer_source = base_model if is_local_adapter and base_model else model_path

  if resolved_device == "cuda":
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
    )
    model_kwargs = {
      "quantization_config": bnb_config,
      "device_map": {"": 0},
      "trust_remote_code": True,
    }
  else:
    model_kwargs = {
      "trust_remote_code": True,
      "torch_dtype": torch.float32,
    }

  if is_local_adapter:
    if not base_model:
      raise ValueError("--base-model must be specified when loading a local adapter")
    try:
      from peft import PeftModel # type: ignore
    except ImportError as exc:
      raise SystemExit(
        "Missing dependency: peft.\n"
        "Install the required dependencies for adapter inference."
      ) from exc

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(model, model_path)
  else:
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"

  model.eval()
  return model, tokenizer, resolved_device


def _model_device(model):
  device = getattr(model, "device", None)
  if device is not None:
    return device
  return next(model.parameters()).device


def generate_chat_response(
  model,
  tokenizer,
  user_content: str,
  do_sample: bool = False,
  temperature: float = 0.3,
  top_p: float = 0.9,
  max_new_tokens: int = 350,
  system_prompt: str = SYSTEM_PROMPT,
) -> str:
  import torch

  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_content},
  ]
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
  )
  inputs = tokenizer(text, return_tensors="pt").to(_model_device(model))

  with torch.no_grad():
    generation_kwargs = {
      **inputs,
      "max_new_tokens": max_new_tokens,
      "do_sample": do_sample,
      "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
      generation_kwargs["temperature"] = temperature
      generation_kwargs["top_p"] = top_p
    outputs = model.generate(**generation_kwargs)

  new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
  return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
