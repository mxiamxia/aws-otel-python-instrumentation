## 2025-05-14 - Optimized Wildcard Matching in Sampler
**Learning:** Iterating over rule attributes (O(R)) instead of span attributes (O(A)) and using early returns significantly improves performance when R << A. Additionally, fast-paths for common wildcard patterns (prefix, suffix, contains) and regex caching avoid the heavy overhead of the regex engine for the majority of telemetry matching.
**Action:** Always check if a set of rules can be iterated over instead of the data being evaluated, and look for string-based fast-paths before falling back to regex.
