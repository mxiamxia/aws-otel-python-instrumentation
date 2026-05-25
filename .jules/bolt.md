## 2026-05-25 - [Optimize telemetry processing hot paths]
**Learning:** Implementing fast-paths for common wildcard patterns (prefix, suffix, middle-wildcard) and using `functools.lru_cache` for complex regexes can significantly improve performance in high-frequency paths like sampling and span processing. Explicit regex pre-compilation also provides a measurable gain for large patterns.
**Action:** Always look for opportunities to bypass the regex engine for simple string matching and cache expensive regex objects in performance-critical code.
