## Baseline k-CBS (k-Chunked Beam Search) Pseudocode

```text
Input:
    model M
    prefix pref
    suffix length T
    beam width k (k ≪ |V|)
    final prune flag rho ∈ {0,1}
    EOS policy: disallow EOS until step T, allow at T

Output:
    F – list containing either k or k² completed suffixes,
        each paired with its accumulated log Pr_{M,dec}(· | pref)

Procedure:
    1. Initialize live beam L ← { (pref, 0) }  // (sequence, log_prob)

    2. For t from 1 to T:
           allow_eos ← (t == T)
           U ← ∅  // children generated at this depth

           For each (seq, log_p) in L:
               logits ← M(seq)
               if not allow_eos:
                   mask EOS logit
               S ← TopK(logits, k)              // token ids
               log_probs ← LogSoftmax(logits)   // over vocabulary
               Z ← LogSumExp(log_probs[S])      // mass of S

               For each token g in S:
                   new_seq ← seq ⧺ g
                   new_log_p ← log_p + log_probs[g] − Z
                   append (new_seq, new_log_p) to U

           if t == T:
               sort U by decreasing log probability
               if rho == 1:
                   F ← first k elements of U
               else:
                   F ← U      // keep all k² paths
               break

           sort U by decreasing log probability
           L ← first k elements of U

    3. Return F
```

Notes:
- The normalization constant Z ensures log probabilities reflect sampling under top-k decoding (probability mass restricted to S).
- When rho = 1, the algorithm outputs the top-k completed suffixes; otherwise all k² combinations from the final expansion are returned.
