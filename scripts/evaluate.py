"""
evaluate.py — VetQwen evaluation script

Evaluates either the base model or the fine-tuned LoRA adapter on
the held-out test set using:
  - ROUGE-L (response similarity)
  - BERTScore F1 (semantic similarity)
  - Format compliance rate (structured template adherence)
  - Species-level breakdown
  - LLM-as-judge scores via Ollama (accuracy, completeness, triage quality, hallucination)

Results are saved to results/<run_name>.json.

Usage:
    # Baseline (base model)
    python scripts/evaluate.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --split test \
        --run-name baseline_3b --no-judge

    # Fine-tuned VetQwen
    python scripts/evaluate.py \
        --model ./adapter \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --split test \
        --run-name vetqwen_3b_r16 --no-judge
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are VetQwen, an expert veterinary diagnostic assistant. "
    "Given a patient signalment and clinical symptoms, provide a structured "
    "differential diagnosis with clinical reasoning, ranked differentials, and "
    "a triage recommendation. Always remind the user that your output is not a "
    "substitute for professional veterinary examination."
)

# Sections we expect to see in a well-formed response
EXPECTED_SECTIONS = [
    "**Species & Signalment:**",
    "**Presenting Symptoms:**",
    "**Assessment:**",
    "**Differential Diagnoses",
    "**Triage Recommendation:**",
    "**Suggested Next Steps:**",
]

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

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_path: str, base_model: str | None = None):
    """
    Load model for inference.
    - If model_path is a HuggingFace repo ID (no local adapter), load base model directly.
    - If model_path is a local path (LoRA adapter), load base model + adapter.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    is_local_adapter = Path(model_path).exists() and Path(model_path).is_dir()

    if is_local_adapter:
        # Load base model + LoRA adapter
        assert base_model, "--base-model must be specified when loading a local adapter"
        log.info(f"Loading base model: {base_model}")
        from peft import PeftModel  # type: ignore

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
        log.info(f"Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        # Load base model directly (for baseline evaluation)
        log.info(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_response(
    model,
    tokenizer,
    user_content: str,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_new_tokens: int = 350,
) -> str:
    """Generate a model response for a single user prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    from rouge_score import rouge_scorer  # type: ignore

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_bert_score(predictions: list[str], references: list[str]) -> float:
    from bert_score import score as bert_score_fn  # type: ignore

    log.info("Computing BERTScore (this may take a moment) ...")
    _, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return f1.mean().item()


def compute_format_compliance(predictions: list[str]) -> float:
    """Check what fraction of responses contain all expected section headers."""
    compliant = 0
    for pred in predictions:
        if all(section in pred for section in EXPECTED_SECTIONS):
            compliant += 1
    return compliant / len(predictions) if predictions else 0.0


def extract_species(user_content: str) -> str:
    text = user_content.lower()
    for s in ["dog", "cat", "cattle", "pig", "sheep"]:
        if s in text:
            return s
    return "unknown"


def compute_species_breakdown(
    predictions: list[str],
    references: list[str],
    user_contents: list[str],
) -> dict[str, dict]:
    """Per-species ROUGE-L scores."""
    from rouge_score import rouge_scorer  # type: ignore

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    buckets: dict[str, list[float]] = {}

    for pred, ref, user in zip(predictions, references, user_contents):
        species = extract_species(user)
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        buckets.setdefault(species, []).append(score)

    return {
        species: {"rouge_l": sum(scores) / len(scores), "n": len(scores)}
        for species, scores in buckets.items()
    }


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


def judge_response(prompt: str, response: str) -> dict[str, Any] | None:
    """Send a response to Ollama for LLM-as-judge scoring. Returns dict or None."""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

    payload = {
        "model": JUDGE_MODEL,
        "prompt": judge_prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 100},
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Extract JSON
        json_str = raw
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    json_str = part
                    break

        scores = json.loads(json_str)
        return scores
    except Exception as e:
        log.warning(f"Judge request failed: {e}")
        return None


def run_llm_judge(
    user_contents: list[str],
    predictions: list[str],
    sample_size: int = 100,
) -> dict[str, float]:
    """Run LLM-as-judge on up to sample_size predictions. Returns averaged scores."""
    import random

    indices = list(range(len(predictions)))
    if len(indices) > sample_size:
        indices = random.sample(indices, sample_size)

    results: list[dict] = []
    for i, idx in enumerate(indices):
        scores = judge_response(user_contents[idx], predictions[idx])
        if scores:
            results.append(scores)
        if (i + 1) % 10 == 0:
            log.info(f"  Judge progress: {i + 1}/{len(indices)}")
        time.sleep(0.1)

    if not results:
        return {}

    keys = ["clinical_accuracy", "completeness", "tone", "hallucination"]
    return {
        k: sum(r.get(k, 0) for r in results) / len(results)
        for k in keys
        if any(k in r for r in results)
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_test_split(path: str) -> tuple[list[str], list[str]]:
    """
    Load test JSONL. Returns (user_contents, assistant_references).
    """
    user_contents = []
    references = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record["messages"]
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            asst_msg = next(
                (m["content"] for m in messages if m["role"] == "assistant"), ""
            )
            user_contents.append(user_msg)
            references.append(asst_msg)
    return user_contents, references


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> None:
    # Determine test file path
    split_file = f"data/processed/{args.split}.jsonl"
    log.info(f"Loading test data from {split_file} ...")
    user_contents, references = load_test_split(split_file)

    # Optionally limit to N samples
    if args.limit:
        user_contents = user_contents[: args.limit]
        references = references[: args.limit]
    log.info(f"Evaluating on {len(user_contents)} samples")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model)

    # Generate predictions
    log.info("Generating responses ...")
    predictions = []
    for i, user_content in enumerate(user_contents):
        pred = generate_response(model, tokenizer, user_content)
        predictions.append(pred)
        if (i + 1) % 10 == 0:
            log.info(f"  {i + 1}/{len(user_contents)}")

    # Compute automatic metrics
    log.info("Computing ROUGE-L ...")
    rouge_l = compute_rouge_l(predictions, references)

    log.info("Computing BERTScore ...")
    bert_f1 = compute_bert_score(predictions, references)

    log.info("Computing format compliance ...")
    fmt_compliance = compute_format_compliance(predictions)

    log.info("Computing per-species breakdown ...")
    species_breakdown = compute_species_breakdown(
        predictions, references, user_contents
    )

    # LLM-as-judge
    judge_scores: dict = {}
    if not args.no_judge:
        log.info("Running LLM-as-judge evaluation (via Ollama) ...")
        judge_scores = run_llm_judge(
            user_contents, predictions, sample_size=args.judge_sample
        )

    # Print results
    log.info("\n" + "=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"ROUGE-L:           {rouge_l:.4f}")
    log.info(f"BERTScore F1:      {bert_f1:.4f}")
    log.info(f"Format compliance: {fmt_compliance:.2%}")
    for species, stats in species_breakdown.items():
        log.info(f"  [{species}] ROUGE-L: {stats['rouge_l']:.4f} (n={stats['n']})")
    if judge_scores:
        log.info(
            f"Judge — clinical accuracy: {judge_scores.get('clinical_accuracy', 0):.2f}/5"
        )
        log.info(
            f"Judge — completeness:      {judge_scores.get('completeness', 0):.2f}/5"
        )
        log.info(f"Judge — tone:              {judge_scores.get('tone', 0):.2f}/5")
        log.info(
            f"Judge — hallucination:     {judge_scores.get('hallucination', 0):.2f}/5"
        )
    log.info("=" * 60)

    # Save results to JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results = {
        "run_name": args.run_name,
        "model": args.model,
        "split": args.split,
        "n_samples": len(user_contents),
        "metrics": {
            "rouge_l": rouge_l,
            "bert_score_f1": bert_f1,
            "format_compliance": fmt_compliance,
        },
        "species_breakdown": species_breakdown,
    }
    if judge_scores:
        results["judge_scores"] = judge_scores

    results_path = results_dir / f"{args.run_name}.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Save predictions
    preds_path = results_dir / f"{args.run_name}_predictions.jsonl"
    with preds_path.open("w") as f:
        for user, ref, pred in zip(user_contents, references, predictions):
            f.write(
                json.dumps(
                    {
                        "user": user,
                        "reference": ref,
                        "prediction": pred,
                    }
                )
                + "\n"
            )
    log.info(f"Predictions saved to {preds_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VetQwen evaluation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model HuggingFace ID (required if --model is a local LoRA adapter)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="evaluation",
        help="Name for this evaluation run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to N samples (useful for quick checks)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (faster, no Ollama required)",
    )
    parser.add_argument(
        "--judge-sample",
        type=int,
        default=100,
        help="Number of samples to send to LLM judge (default: 100)",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
