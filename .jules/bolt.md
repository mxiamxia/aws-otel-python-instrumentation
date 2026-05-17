## 2025-05-15 - Optimize wildcard matching and SQL keyword regex
**Learning:** In performance-critical paths like sampling and attribute generation, regex compilation and matching can be a significant bottleneck.
**Action:** Pre-compile regex patterns at the module level and implement fast-paths for common string patterns (prefix, suffix) to bypass the regex engine entirely. Use `functools.lru_cache` for dynamic regex patterns.
