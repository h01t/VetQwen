"""
Shared Ollama LLM-as-judge helpers used by evaluation scripts.
"""

from __future__ import annotations

import json
import random
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = "qwen2.5:7b"

JUDGE_PROMPT_TEMPLATE = """You are an expert veterinary clinician reviewing an AI-generated diagnostic response.

Rate the following veterinary diagnostic response on a 1-5 scale for each criterion.

**Original case prompt:**
{prompt}

**AI response:**
{response}

**Rating criteria:**
- clinical_accuracy: Are the differential diagnoses plausible and appropriate for the species and symptoms?
- completeness: Does the response include reasoning, ranked differentials, triage, and next steps?
- tone: Is the language appropriate for a veterinary diagnostic assistant (clinical, professional)?
- hallucination: Does the response contain invented drug names, impossible symptoms, or fabricated facts? (1=no hallucination, 5=severe hallucination)

Return ONLY valid JSON with no additional text:
{{"clinical_accuracy": <1-5>, "completeness": <1-5>, "tone": <1-5>, "hallucination": <1-5>}}
"""


def _http_json(
    url: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 60
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_json_payload(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if "```" in candidate:
        for part in candidate.split("```"):
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                candidate = part
                break
    return json.loads(candidate)


def check_ollama(base_url: str, timeout: int = 5) -> tuple[bool, str]:
    try:
        response = _http_json(f"{base_url}/api/tags", timeout=timeout)
        models = [model.get("name", "") for model in response.get("models", [])]
        return True, ", ".join(models)
    except (URLError, HTTPError, json.JSONDecodeError, TimeoutError) as exc:
        return False, str(exc)


def select_sample_indices(total: int, sample_size: int | None = None, seed: int = 42) -> list[int]:
    indices = list(range(total))
    if sample_size is None or sample_size <= 0 or sample_size >= total:
        return indices
    rng = random.Random(seed)
    return sorted(rng.sample(indices, sample_size))


def judge_response(
    prompt: str,
    response: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: int = 60,
) -> dict[str, Any] | None:
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)
    payload = {
        "model": model,
        "prompt": judge_prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 100},
    }

    try:
        raw = _http_json(
            f"{base_url}/api/generate",
            method="POST",
            payload=payload,
            timeout=timeout,
        ).get("response", "")
        return extract_json_payload(raw)
    except (URLError, HTTPError, json.JSONDecodeError, TimeoutError):
        return None
