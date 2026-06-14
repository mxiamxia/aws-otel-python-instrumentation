## 2026-06-14 - [Optimizing Sampler Matcher]
**Learning:** In the sampling hot path, iterating over rule attributes ((R)$) rather than all span attributes ((A)$) significantly reduces overhead, especially as  \ll A$ in most production scenarios. Combining this with  for regex compilation and avoiding redundant loops for wildcard detection provides a compound performance benefit.
**Action:** Always prefer iterating over the set of constraints/rules rather than the full data set when performing filtering or matching in performance-critical paths.
## 2025-05-15 - [Optimizing Sampler Matcher]
**Learning:** In the sampling hot path, iterating over rule attributes (O(R)) rather than all span attributes (O(A)) significantly reduces overhead, especially as R << A in most production scenarios. Combining this with lru_cache for regex compilation and avoiding redundant loops for wildcard detection provides a compound performance benefit.
**Action:** Always prefer iterating over the set of constraints/rules rather than the full data set when performing filtering or matching in performance-critical paths.
