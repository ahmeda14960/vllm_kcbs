"""
Minimal example for running KCBS (k-Chunked Beam Search) with vLLM.

Note: This uses a regular `LLM.generate` loop under the hood and does not
require any sampler changes. For best fidelity to top-k decoding semantics,
avoid penalties/min_p/logit_bias and keep temperature near 1.0.
"""

from __future__ import annotations

from vllm import LLM
from vllm.sampling_params import KCBSParams


def main():
    # Choose a small model for quick runs; replace as needed.
    # Ensure the model is available locally or accessible in your environment.
    llm = LLM(model="Qwen/Qwen3-0.6B")

    prompts = [
        {"prompt": "Write a short haiku about the sea."},
        {"prompt": "Name three colors starting with the letter B."},
    ]

    params = KCBSParams(
        beam_width=3,
        max_tokens=2,   # T: suffix length
        final_prune=True,  # keep only top-k finals; set False to keep k^2
        temperature=1.0,
    )

    outputs = llm.kcbs(prompts, params)

    for i, out in enumerate(outputs):
        print(f"\nPrompt {i} results:")
        for j, seq in enumerate(out.sequences):
            text = seq.text or ""
            print(f"  [{j}] cum_logprob={seq.cum_logprob:.4f} -> {text}")


if __name__ == "__main__":
    main()
