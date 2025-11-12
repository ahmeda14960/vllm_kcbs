## Plan: Integrating Baseline k-CBS into vLLM

This document lays out the concrete code changes required to support the Baseline k-CBS strategy (Section `kcbs.md`) inside vLLM’s v1 sampling stack. The plan is written so another coding agent can implement it step-by-step.

---

### 1. Surface API & Configuration
1. **Generation Config / Request Params**
   - File(s): `vllm/config.py` (or wherever sampling params are sourced), request dataclasses, and engine interfaces.
   - Add explicit flags/fields:
     - `kcbs_enabled: bool`
     - `kcbs_beam_width: int` (defaults to existing `top_k` when unset)
     - `kcbs_final_prune: bool` (maps to ρ; `True` ⇒ keep top-k finals)
   - Ensure these propagate through whatever path currently supplies `top_k`/`top_p`/temperature into `SamplingMetadata`.

2. **SamplingMetadata Extensions**
   - File: `vllm/v1/sample/metadata.py`
   - Add new fields:
     ```python
     kcbs_enabled: bool
     kcbs_beam_width: torch.Tensor | None  # shape [batch], mirrored from top_k
     kcbs_final_prune: bool
     ```
   - Populate them where `SamplingMetadata` is constructed (search for factories in `vllm/v1/runner` or equivalent). Default to disabled to avoid touching existing flows.

---

### 2. Sampler Layer Changes
1. **Entry Point**
   - File: `vllm/v1/sample/sampler.py`
   - Add a fast check near `Sampler.sample()` invocation:
     ```python
     if sampling_metadata.kcbs_enabled:
         return self.sample_kcbs(...)
     ```
   - Keep the current behavior untouched otherwise.

2. **KCBS Helper**
   - Add a new method (exact signature TBD):
     ```python
     def sample_kcbs(
         self,
         logits: torch.Tensor,
         sampling_metadata: SamplingMetadata,
     ) -> tuple[torch.Tensor, torch.Tensor | None]
     ```
   - Responsibilities (mirrors `kcbs.md`):
     - Maintain per-request beam state `L` as tensors of shape `[batch_size, beam_width, vocab?]`. Because vLLM batches multiple requests, flatten `(req, beam_slot)` into the batch dimension (size `B = num_requests * beam_width`) when passing through logits processors.
     - For each step `t`:
       - Call into a new `TopKTopPSampler` helper (see next section) that returns the *top-k token ids, log probs, and log-mass Z* for each live beam row instead of sampling one token.
       - Update cumulative log probs: `log_p' = log_p + log_prob_token - Z`.
       - Append `(seq', log_p')` to a temporary tensor/list `U`.
       - If `t == T`, either prune to top-k finals (`rho=1`) or keep all `k²` combos.
       - Otherwise, sort/prune `U` to width `k` per request before the next iteration.
     - Return a tensor of chosen token ids shaped `[batch_size, beam_width]` (or `[batch_size]` if you pick one representative sequence to feed downstream). Clarify whether downstream code expects exactly one token per request; if so, adaptation may be required (e.g., only emit the best KCBS path while retaining alternates separately).

3. **Batch/State Management**
   - Extend `SamplingMetadata.output_token_ids` (or create a KCBS-specific structure) to store beam histories so penalties/bad-word filters can operate on every partial sequence.
   - Ensure masking (`allowed_token_ids_mask`, `bad_words_token_ids`) operates on the expanded beam dimension.

---

### 3. TopK / Logprob Primitive Enhancements
1. **`TopKTopPSampler` Additions**
   - File: `vllm/v1/sample/ops/topk_topp_sampler.py`
   - Add a method that exposes the *deterministic* top-k set instead of sampling:
     ```python
     def topk_candidates(
         self,
         logits: torch.Tensor,
         k: torch.Tensor,
         p: torch.Tensor | None,
     ) -> KCBSBatch  # custom dataclass holding token ids, log probs, Z
     ```
   - Implementation:
     - Reuse `apply_top_k_top_p` to mask logits.
     - Use `torch.topk(masked_logits, k, dim=-1)` to gather candidate ids and logits (they already have `-inf` elsewhere).
     - Convert to log probs with `log_probs = masked_logits.log_softmax(dim=-1)`; gather the per-row `log_probs_topk`.
     - Compute `Z` per row via `torch.logsumexp(log_probs_topk, dim=-1, keepdim=True)` so the caller can subtract it, matching the KCBS normalization term.
     - Return tensors shaped `[B, k]` for ids and log probs, plus `[B, 1]` for `Z`.
   - Keep the current random sampling fast path intact; only call `topk_candidates` from `sample_kcbs`.

2. **Data Container**
   - Define a lightweight dataclass in the same module (or a nearby `ops` helper) to hold:
     ```python
     @dataclass
     class KCBSBatch:
         token_ids: torch.Tensor  # int64, shape [B, k]
         token_log_probs: torch.Tensor  # float32, shape [B, k]
         log_mass: torch.Tensor  # float32, shape [B, 1]
     ```

---

### 4. Output Handling
1. **SamplerOutput Contract**
   - `SamplerOutput.sampled_token_ids` currently expects `[num_requests, 1]`. Decide how KCBS results should be surfaced:
     - Option A: extend `SamplerOutput` with `kcbs_sequences` containing the entire set of finalized beams, while `sampled_token_ids` continues to hold the top-1 path for compatibility.
     - Option B: generalize `sampled_token_ids` to `[num_requests, beam_width]` when KCBS is enabled and adjust downstream consumers (logprobs processor, scheduler) accordingly.
   - Update `vllm/v1/outputs.py` and any downstream logic that assumes shape `[*, 1]`.
2. **Logprob Reporting**
   - When `kcbs_enabled`, ensure the per-beam accumulated log probs are exposed so the caller can select between `k` vs `k²` outputs. May require extending `LogprobsTensors` or returning an auxiliary tensor.

---

### 5. Testing Strategy
1. **Unit Tests**
   - Location: `tests/v1/test_sampler.py` (or create `test_kcbs.py`).
   - Mock a small vocab LLM (e.g., logits tensor with deterministic values) and verify:
     - KCBS reproduces the exact sequences specified in `kcbs.md`.
     - Summed probabilities of top-k tokens equal 1 after the `Z` subtraction.
     - `rho` flag controls whether only top-k or all k² finals are returned.
2. **Regression / Integration**
   - Add an end-to-end generation test comparing KCBS outputs against a Python reference implementation using the same logits (possibly via dependency injection).

---

### 6. Rollout Considerations
1. **Backward Compatibility**
   - Guard all new behavior behind `kcbs_enabled`.
   - Ensure `top_k` sampling performance is unaffected when KCBS is off.
2. **Performance**
   - KCBS inherently expands k×k candidates; monitor memory usage and GPU time.
   - Consider implementing the inner loops in CUDA-friendly tensor ops rather than Python lists to keep things batched.
3. **Documentation**
   - Update `topk_sampling.md` to reference KCBS as an advanced option.
   - Add user-facing docs (maybe `docs/usage/kcbs.md`) explaining new config flags.

---

Following these steps will integrate KCBS cleanly into vLLM while preserving existing sampling behavior.*** End Patch
