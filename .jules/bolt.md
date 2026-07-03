## 2026-07-03 - Optimize Sampler Matcher Performance
**Learning:** Iterating over span attributes in `attribute_match` is inefficient when the number of rule attributes is much smaller. Also, regex compilation and wildcard matching without fast-paths significantly slow down the sampling path.
**Action:** Use `lru_cache` for regex compilation, implement string-based fast-paths for common wildcards (prefix, suffix, contains), and iterate over rule attributes with early return in `attribute_match`.
