Here’s a detailed Markdown document suitable for a coding agent to pick up:

⸻

Issue Report: Difference in Top-K Sampling Between vLLM and Hugging Face Transformers

Overview

This document describes a behavioral difference between vLLM and Hugging Face Transformers in their implementations of top-k sampling.
Although both libraries aim to perform stochastic token selection from the most likely candidates, the order of operations (softmax → top-k vs. top-k → softmax) leads to significantly different outcomes.

⸻

Summary of the Problem

When top_k > 0 and sampling is enabled, vLLM and Transformers produce diverging token distributions—even with identical random seeds—because they differ in when softmax normalization is applied relative to the top-k truncation.

⸻

vLLM Implementation Details

File: vllm/model_executor/layers/sampler.py

Process:
	1.	Softmax first:

probs = torch.softmax(logits, dim=-1, dtype=torch.float)
logprobs = torch.log(probs)


	2.	Then apply top-k/top-p truncation:

probs = _apply_top_p_top_k(probs, top_ps, top_ks)


	3.	Sample directly from truncated probabilities:

next_token = torch.multinomial(probs, num_samples=1)



Key Observation:
After truncation, the probs tensor is not renormalized — i.e., its total probability mass is < 1.0. Sampling from this unnormalized distribution effectively over-weights the largest probabilities and under-weights smaller surviving tokens.

⸻

Transformers Implementation Details

Files:
	•	generation/utils.py
	•	generation/logits_process.py (class TopKLogitsWarper)

Process:
	1.	Top-k truncation on logits (before softmax):

indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
scores = scores.masked_fill(indices_to_remove, -float("Inf"))


	2.	Then compute softmax on truncated logits:

probs = nn.functional.softmax(next_token_scores, dim=-1)
next_tokens = torch.multinomial(probs, num_samples=1)



Key Observation:
By performing softmax after truncation, the truncated logits yield a normalized probability distribution where surviving tokens sum to 1.0. This is the standard top-k sampling definition used in most language generation frameworks.

⸻

Behavioral Consequences

Aspect	vLLM	Transformers
Truncation order	After softmax	Before softmax
Probability normalization	Not renormalized	Renormalized
Sampling bias	Over-emphasizes high-probability tokens	Preserves proper stochastic balance
Determinism with same seed	Divergent results	Consistent with standard definition

Even when both use the same random seed and logits, sampled tokens will likely differ because vLLM samples from a scaled-down, non-normalized distribution.

⸻

Reproduction Example
	1.	Run inference in both frameworks with:

top_k=50
temperature=1.0
do_sample=True
seed=42


	2.	Observe that the first few generated tokens differ even though the logits are identical.

⸻

Potential Root Cause

vLLM’s _apply_top_p_top_k() operates after softmax rather than on logits.
Because it directly zeroes out probabilities instead of logits, the remaining probabilities are not renormalized before sampling.

⸻

Possible Fix / Alignment Suggestion

To align vLLM with Hugging Face Transformers:
	1.	Move truncation before softmax, or
	2.	Re-normalize probabilities after truncation:

probs = _apply_top_p_top_k(probs, top_ps, top_ks)
probs = probs / probs.sum(dim=-1, keepdim=True)



This ensures the distribution sums to 1 and matches the expected semantics of top-k sampling.

⸻

Conclusion

This discrepancy is not necessarily a bug, but rather a design divergence.
However, since most frameworks—including Hugging Face Transformers—apply top-k on logits before softmax, vLLM’s behavior may confuse developers expecting consistent results or reproducibility across toolchains.

If uniform behavior across frameworks is desired, vLLM should adopt or provide an option for pre-softmax top-k truncation or post-truncation renormalization.

⸻

Author: Generated analysis by ChatGPT (GPT-5)
Date: 2025-11-07
Purpose: Documentation for coding agent to align vLLM sampling logic with Hugging Face Transformers.

⸻

Would you like me to include a ready-to-apply code patch or PR-style diff section at the end?

⸻

Update: Current vLLM (v1) Behavior

The latest vLLM sampler has been refactored to run top-k/top-p filtering **before** the softmax step, eliminating the normalization gap described above. Key changes:
	•	File: vllm/v1/sample/sampler.py – `Sampler.sample()` now hands logits (after penalties, temperature, and logits processors) directly to `TopKTopPSampler` without taking a softmax first.
	•	File: vllm/v1/sample/ops/topk_topp_sampler.py – `apply_top_k_top_p()` masks logits in-place (setting trimmed entries to `-inf`). Only afterward does `forward_native`/`forward_cpu` call `logits.softmax(...)`, so the remaining candidates are renormalized automatically.
	•	Sampling uses a Gumbel-style `random_sample()` (or FlashInfer kernels when enabled) on the normalized probabilities, keeping behavior aligned with Hugging Face’s logits-first truncation semantics.

Effectively, vLLM v1 now matches the standard “top-k/top-p → softmax → sample” ordering, so the bias outlined earlier only applies to historical versions (e.g., v0’s `vllm/model_executor/layers/sampler.py`). Use the new sampler when consistency with Transformers is required.
