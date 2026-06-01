## 2025-05-14 - Optimizing AWS Resource Identifier Parsing
**Learning:** AWS resource identifiers like ARNs and SQS URLs are frequently reused across spans. Repeatedly parsing these strings using `.split()`, `.replace()`, and manual validation loops in the telemetry hot path (metric attribute generation) introduces significant latency.
**Action:** Always apply `@functools.lru_cache` to utility functions that parse AWS identifiers. Use string slicing instead of `.replace()` for known prefix removal, and prefer pre-compiled regex over manual character loops for validation.
