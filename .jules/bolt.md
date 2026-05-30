## 2025-05-14 - Optimized SQS URL and ARN Parsing
**Learning:** Caching parsing results of highly repetitive resource identifiers (ARNs, SQS URLs) and using efficient string slicing instead of repeated replacements significantly reduces overhead in the telemetry processing path.
**Action:** Always look for repetitive string parsing in hot paths and apply memoization and efficient string operations.

## 2025-05-14 - Safter Caching with Tuples
**Learning:** Returning a `tuple` instead of a `list` from cached methods ensures that the cached objects are immutable, preventing accidental side effects if a consumer attempts to modify the returned value.
**Action:** Return immutable types (tuples, frozensets) from methods decorated with `@lru_cache` when possible.
