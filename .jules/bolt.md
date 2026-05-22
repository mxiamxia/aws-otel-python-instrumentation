## 2025-05-22 - Optimize Wildcard Matching in Sampler
**Learning:** Wildcard matching using regex in hot paths (like sampling) is a significant performance bottleneck. Simple string operations (startswith, endswith, in) are much faster than the regex engine for common patterns like "prefix*", "*suffix", or "*contains*".
**Action:** Always check for simple wildcard patterns and use fast-path string methods before falling back to regex. Use functools.lru_cache to avoid redundant regex compilation for complex patterns.
