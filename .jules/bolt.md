## 2026-07-06 - Optimized Sampler Matcher
**Learning:** Iterating over rule attributes instead of span attributes in `_Matcher.attribute_match` significantly reduces overhead in hot paths, especially when spans have many attributes. Combining this with early returns and fast-paths for common wildcard patterns (prefix/suffix/substring) can lead to a >4x speedup.
**Action:** Always consider the relative size of input collections when designing matching logic, and prioritize early returns and string-based fast-paths over regex where possible.
