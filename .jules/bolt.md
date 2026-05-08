## 2025-05-15 - [Regex Pre-compilation]
**Learning:** Pre-compiling a large regex (like SQL_KEYWORD_PATTERN with 600+ keywords) provides a ~15-20% speedup in the hot path of telemetry processing compared to letting `re.match` handle it with its internal cache.
**Action:** Always pre-compile regex patterns that are defined at the module level and used in telemetry processing paths.
