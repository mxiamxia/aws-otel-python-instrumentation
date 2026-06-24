## 2026-06-24 - [Optimizing Matcher for High-Performance Sampling]
**Learning:** In hot paths like OpenTelemetry samplers, iterating over the smaller set of rules (O(R)) instead of the larger set of span attributes (O(A)) can yield significant performance gains (~4x-10x) by reducing unnecessary dictionary lookups and allowing for early returns. Additionally, regex compilation is a major bottleneck; implementing fast-paths for common wildcard patterns (prefix, suffix, contains) using native string methods bypasses the regex engine entirely for a ~3x-5x speedup.

**Action:** Always check if iteration order can be reversed to favor the smaller collection, and prioritize native string operations over regex for common patterns in performance-critical sections.
