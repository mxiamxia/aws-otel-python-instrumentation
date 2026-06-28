## 2026-06-28 - Optimizing Samer Wildcard and Attribute Matching
**Learning:** Matching span attributes against sampling rules is a hot path where O(SpanAttributes) iteration can be a bottleneck when many attributes are present. Regex compilation is also expensive and often redundant for static rules.
**Action:** Use 'lru_cache' for regex patterns and implement string-based fast-paths for common wildcards (prefix, suffix, middle). Always iterate over rule attributes instead of span attributes for better performance and early exit.
