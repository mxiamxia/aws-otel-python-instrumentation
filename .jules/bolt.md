## 2026-06-12 - Optimization of _Matcher in Sampling Path
**Learning:** Switching iteration from span attributes (O(A)) to rule attributes (O(R)) in `attribute_match` significantly improves performance because R is typically much smaller than A. Additionally, regex matching can be avoided for common wildcard patterns (prefix, suffix, contains) by implementing string-based fast-paths.
**Action:** Always prefer iterating over the smaller set of keys in matching logic and use `lru_cache` for compiled regex or repetitive string parsing in hot paths.
