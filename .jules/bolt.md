## 2026-06-20 - [Optimized _Matcher in Sampler]
**Learning:** In the sampling hot path, iterating over span attributes (O(A)) to match against rules is inefficient if the span has many attributes and rules are few. Switching to rule-based iteration (O(R)) with early return significantly improves performance.
**Action:** Always prefer iterating over the smaller set of keys and use early returns in matching/validation logic.
