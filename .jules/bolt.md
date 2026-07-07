## 2026-07-07 - Optimized wildcard matching and attribute matching in sampler
**Learning:** Wildcard matching using regex is expensive for common patterns like prefix (`*suffix`), suffix (`prefix*`), and substring (`*substring*`). Reversing the attribute matching loop to iterate over rules instead of span attributes is also more efficient when span attributes outnumber rules.
**Action:** Use string methods (`startswith`, `endswith`, `in`) for common wildcard patterns and `functools.lru_cache` for regex compilation in hot paths.
