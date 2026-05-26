## 2026-05-26 - Optimized Attribute Matching in Sampler
**Learning:** In performance-critical paths like sampling or attribute filtering, iterating over the smaller set of keys (e.g., rule attributes O(R)) instead of the larger set (e.g., span attributes O(A)) significantly reduces complexity and allows for efficient early returns.
**Action:** Always check the relative sizes of data sets being matched and iterate over the more constrained one to optimize complexity.
