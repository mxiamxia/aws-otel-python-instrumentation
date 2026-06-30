## 2024-05-23 - Optimize Sampler Wildcard Matching
**Learning:** Wildcard matching using regular expressions in a hot path (sampling) is a significant bottleneck. Implementing fast-paths for common patterns (prefix, suffix, middle-contains) using native string operations can improve performance by nearly 10x.
**Action:** Always check for common string patterns before falling back to regex in high-frequency matching logic. Use `lru_cache` for compiled regex to avoid redundant processing.
