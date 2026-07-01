## 2026-07-01 - Wildcard Matching Optimization
**Learning:** Glob-style wildcard matching (`*`, `?`) in the sampling path was a major bottleneck due to redundant regex compilation and execution. Implementing O(N) fast-paths for common patterns (`prefix*`, `*suffix`, `*middle*`) and using `lru_cache` for compiled regex objects significantly reduces CPU overhead.
**Action:** Always prefer specialized string operations over regex for simple glob patterns in high-frequency paths.
