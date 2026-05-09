
## 2026-05-09 - [Sampling & Attribute Matching Optimization]
**Learning:** In high-frequency telemetry paths, iterating over span attributes O(A) is much more expensive than iterating over rule attributes O(R). Also, bypassing the regex engine for exact matches and using `lru_cache` for wildcard-to-regex conversion provides significant gains.
**Action:** Always prefer O(Rule) iteration over O(Span) iteration in attribute matching logic. Pre-compile regex or use caching for dynamic regex generation.
