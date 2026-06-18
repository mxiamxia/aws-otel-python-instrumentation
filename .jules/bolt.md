## 2026-06-18 - [Optimization of Sampler _Matcher]
**Learning:** In high-frequency paths like OpenTelemetry sampling, iterating over the smaller set (rule attributes, O(R)) instead of the larger set (span attributes, O(A)) significantly reduces overhead. Additionally, bypassing the regex engine with string-based fast-paths for common wildcard patterns (prefix, suffix, contains) provides substantial speedups (~5x-20x).
**Action:** Always prefer iterating over the defined rule set and use string methods for simple pattern matching before falling back to regex.
