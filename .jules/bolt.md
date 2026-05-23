## 2026-05-23 - Optimized Wildcard Matching in Sampler
**Learning:** Wildcard matching using regular expressions is a significant overhead in the sampling hot path, especially when patterns are repeatedly compiled.
**Action:** Implement fast-paths for common wildcard patterns (exact match, `prefix*`, `*suffix`, `*middle*`) using string methods, and use `functools.lru_cache` for complex patterns to avoid redundant regex compilation.
