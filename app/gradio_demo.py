"""
gradio_demo.py — VetQwen Gradio inference UI

Single-page demo app for the VetQwen veterinary diagnostic assistant.
Loads the fine-tuned LoRA adapter on top of Qwen2.5-3B-Instruct.

Usage:
    python app/gradio_demo.py --adapter ./adapter
    python app/gradio_demo.py --adapter ./adapter --base-model Qwen/Qwen2.5-3B-Instruct --share
    python app/gradio_demo.py --adapter ./adapter --cpu  # CPU inference (slow)
"""

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model settings
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = "./adapter"

SYSTEM_PROMPT = (
    "You are VetQwen, an expert veterinary diagnostic assistant. "
    "Given a patient signalment and clinical symptoms, provide a structured "
    "differential diagnosis with clinical reasoning, ranked differentials, and "
    "a triage recommendation. Always remind the user that your output is not a "
    "substitute for professional veterinary examination."
)

# Example cases for the UI
EXAMPLES = [
    [
        "Dog",
        "3 years",
        "Male (neutered)",
        "Golden Retriever",
        "Owner reports 2 days of lethargy, vomiting (3x/day), loss of appetite, and bloody diarrhea.",
    ],
    [
        "Cat",
        "7 years",
        "Female (spayed)",
        "Domestic Shorthair",
        "Increased thirst and urination for the past 3 weeks, weight loss despite good appetite, occasional vomiting.",
    ],
    [
        "Cattle",
        "2 years",
        "Female (intact)",
        "Holstein",
        "Cow found down in the field, unable to rise. Dropped milk production noted for 2 days. Calved 3 days ago.",
    ],
    [
        "Sheep",
        "4 months",
        "Male (intact)",
        "Merino",
        "Lamb showing incoordination, circling, head pressing. Flock recently moved to new pasture.",
    ],
]

DISCLAIMER = (
    "**Medical Disclaimer:** VetQwen is an AI-assisted tool for educational and research purposes only. "
    "It is **not** a substitute for professional veterinary examination, diagnosis, or treatment. "
    "Always consult a licensed veterinarian for clinical decisions."
)

# ---------------------------------------------------------------------------
# Model loading (lazy — loaded on first inference)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def load_model(adapter_path: str, base_model: str, use_cpu: bool = False):
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

    log.info(f"Loading base model: {base_model}")

    if use_cpu:
        # CPU mode: no quantization
        from peft import PeftModel  # type: ignore

        _model = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=True, torch_dtype=torch.float32
        )
        if Path(adapter_path).exists():
            _model = PeftModel.from_pretrained(_model, adapter_path)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        _model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )

        if Path(adapter_path).exists():
            from peft import PeftModel  # type: ignore

            log.info(f"Loading LoRA adapter from: {adapter_path}")
            _model = PeftModel.from_pretrained(_model, adapter_path)
        else:
            log.warning(
                f"Adapter path '{adapter_path}' not found — running base model only (no fine-tuning applied)."
            )

    _tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "left"

    _model.eval()
    log.info("Model ready.")
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def format_prompt(species: str, age: str, sex: str, breed: str, complaint: str) -> str:
    parts = [f"Species: {species}"]
    if age:
        parts.append(f"Age: {age}")
    if sex:
        parts.append(f"Sex: {sex}")
    if breed:
        parts.append(f"Breed: {breed}")
    parts.append(f"Presenting complaint: {complaint}")
    return "\n".join(parts)


def diagnose(
    species: str,
    age: str,
    sex: str,
    breed: str,
    complaint: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    if not complaint.strip():
        return "Please enter a presenting complaint / symptoms."

    model, tokenizer = _model, _tokenizer
    if model is None or tokenizer is None:
        return "Model not loaded. Please restart the demo with a valid --adapter path."

    user_content = format_prompt(species, age, sex, breed, complaint)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_demo(adapter_path: str, base_model: str, use_cpu: bool):
    import gradio as gr  # type: ignore

    # Pre-load model
    load_model(adapter_path, base_model, use_cpu)

    def _diagnose(species, age, sex, breed, complaint, temperature, max_new_tokens):
        return diagnose(
            species, age, sex, breed, complaint, temperature, max_new_tokens
        )

    with gr.Blocks(title="VetQwen — Veterinary Diagnostic Assistant") as demo:
        gr.Markdown("# VetQwen — Veterinary Diagnostic Assistant")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Patient Information")
                species_input = gr.Dropdown(
                    choices=["Dog", "Cat", "Cattle", "Pig", "Sheep"],
                    value="Dog",
                    label="Species",
                )
                age_input = gr.Textbox(
                    label="Age (e.g. '3 years', '6 months')", placeholder="3 years"
                )
                sex_input = gr.Radio(
                    choices=[
                        "Male (intact)",
                        "Male (neutered)",
                        "Female (intact)",
                        "Female (spayed)",
                    ],
                    value="Male (neutered)",
                    label="Sex / Reproductive Status",
                )
                breed_input = gr.Textbox(
                    label="Breed", placeholder="e.g. Golden Retriever"
                )
                complaint_input = gr.Textbox(
                    label="Presenting Complaint / Symptoms",
                    lines=5,
                    placeholder="Describe the owner's chief complaint and clinical signs...",
                )

                with gr.Accordion("Generation Settings", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="Temperature (lower = more focused)",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=100,
                        maximum=600,
                        value=350,
                        step=50,
                        label="Max new tokens",
                    )

                submit_btn = gr.Button(
                    "Generate Diagnostic Assessment", variant="primary"
                )

            with gr.Column(scale=2):
                gr.Markdown("### Diagnostic Assessment")
                output_box = gr.Textbox(
                    label="VetQwen Response",
                    lines=20,
                )
                gr.Markdown(
                    "_VetQwen is powered by Qwen2.5-3B-Instruct fine-tuned with QLoRA on veterinary diagnostic data._"
                )

        # Example cases
        gr.Examples(
            examples=EXAMPLES,
            inputs=[species_input, age_input, sex_input, breed_input, complaint_input],
            label="Example Cases",
        )

        submit_btn.click(
            fn=_diagnose,
            inputs=[
                species_input,
                age_input,
                sex_input,
                breed_input,
                complaint_input,
                temperature_slider,
                max_tokens_slider,
            ],
            outputs=output_box,
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VetQwen Gradio Demo")
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help=f"Path to LoRA adapter directory (default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model HuggingFace ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve Gradio on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run in CPU mode (no quantization, very slow)",
    )
    args = parser.parse_args()

    demo = build_demo(args.adapter, args.base_model, args.cpu)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
