# Bolt's Performance Journal

## 2025-06-07 - Sampler Matcher Optimization
**Learning:** In performance-critical paths like OpenTelemetry sampling, algorithmic improvements often yield higher returns than micro-optimizations. Flipping the iteration logic in `attribute_match` from $O(A)$ (span attributes) to $O(R)$ (rule attributes) and implementing early-return significantly reduced overhead. Additionally, string-based fast-paths for common wildcard patterns (`prefix*`, `*suffix`, `*middle*`) provide a much larger speedup (~5x-20x) than just caching regex compilation.
**Action:** Always check if the set of requirements (rules) is smaller than the set of data (attributes) and optimize iteration direction accordingly. Prioritize string methods over regex for simple pattern matching.

## 2025-06-07 - Type Safety in Sampling Path
**Learning:** Span attributes can be of various types (int, bool, list), but the AWS X-Ray sampling rules expect string patterns. Failing to convert attribute values to strings before matching can lead to `TypeError`.
**Action:** Ensure all attribute values are explicitly cast to `str` before performing wildcard or regex matching in the sampler.
