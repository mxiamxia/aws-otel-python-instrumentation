## 2025-05-14 - Optimized Wildcard Matching and Attribute Filtering

**Learning:** Sampling is a hot path where every microsecond matters. Standard regex-based wildcard matching is expensive due to repeated compilation and engine overhead. Additionally, spans often contain many attributes, while sampling rules only target a few.

**Action:**
1. Implement fast-paths for common wildcard patterns (`prefix*`, `*suffix`, `*middle*`) using native string methods to bypass the regex engine.
2. Use `functools.lru_cache` to cache compiled regex objects for complex patterns.
3. In `attribute_match` logic, iterate over the (usually smaller) set of rule attributes O(R) instead of span attributes O(A) and use early returns to optimize performance.
