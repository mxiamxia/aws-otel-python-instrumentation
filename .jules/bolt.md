## 2026-05-21 - [Optimizing Wildcard Matching and SQL Keyword Patterns]
**Learning:** Performance-critical paths like sampling and metric attribute generation benefit significantly from bypassing regex engines with fast-paths for common patterns and using `functools.lru_cache` to avoid redundant regex compilation.

**Action:** Always implement fast-paths for simple string matches (prefix, suffix, contains) in wildcard logic and ensure large regular expressions are pre-compiled and reused.
