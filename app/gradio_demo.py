"""
gradio_demo.py — VetQwen Gradio inference UI

Single-page demo app for the VetQwen veterinary diagnostic assistant.
Loads the fine-tuned LoRA adapter on top of Qwen2.5-3B-Instruct.

Usage:
    python app/gradio_demo.py --adapter ./adapter
    python app/gradio_demo.py --adapter ./adapter --base-model Qwen/Qwen2.5-3B-Instruct --share
    python app/gradio_demo.py --adapter ./adapter --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vetqwen_core.constants import DEFAULT_ADAPTER, DEFAULT_BASE_MODEL
from vetqwen_core.inference import (
    generate_chat_response,
    load_inference_model,
    resolve_device,
)
from vetqwen_core.records import build_patient_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

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

_model: Any | None = None
_tokenizer: Any | None = None
_resolved_device = None


def load_model(
    adapter_path: str,
    base_model: str,
    device: str = "auto",
) -> tuple[Any, Any]:
    """Load the inference model once and cache it for the demo session."""
    global _model, _tokenizer, _resolved_device

    if _model is not None:
        return _model, _tokenizer

    _resolved_device = resolve_device(device)
    model_path = adapter_path
    model_base = base_model
    if not Path(adapter_path).exists():
        log.warning(
            "Adapter path '%s' not found — running base model only (no fine-tuning applied).",
            adapter_path,
        )
        model_path = base_model
        model_base = None

    log.info("Loading demo model on device=%s", _resolved_device)
    _model, _tokenizer, _resolved_device = load_inference_model(
        model_path,
        model_base,
        device=_resolved_device,
        enforce_supported_python=False,
    )
    log.info("Model ready on device=%s.", _resolved_device)
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def diagnose(
    species: str,
    age: str,
    sex: str,
    breed: str,
    complaint: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Generate a structured diagnostic response for the current form state."""
    if not complaint.strip():
        return "Please enter a presenting complaint / symptoms."

    model, tokenizer = _model, _tokenizer
    if model is None or tokenizer is None:
        return "Model not loaded. Please restart the demo with a valid model configuration."

    user_content = build_patient_prompt(
        species=species,
        complaint=complaint,
        age=age or None,
        sex=sex or None,
        breed=breed or None,
    )
    return generate_chat_response(
        model,
        tokenizer,
        user_content,
        do_sample=True,
        temperature=float(temperature),
        top_p=0.9,
        max_new_tokens=int(max_new_tokens),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_demo(adapter_path: str, base_model: str, device: str) -> Any:
    """Build the Gradio interface and warm the model before launch."""
    import gradio as gr  # type: ignore

    # Surface startup issues before the UI is served.
    load_model(adapter_path, base_model, device)

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
    parser = argparse.ArgumentParser(description="Launch the VetQwen Gradio demo.")
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help=(
            "Inference device to use (default: auto). On Apple Silicon Macs, "
            "use --device mps or keep --device auto."
        ),
    )
    args = parser.parse_args()

    demo = build_demo(args.adapter, args.base_model, args.device)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
