## 2026-06-29 - Optimized _Matcher for high-frequency sampling

**Learning:** Sampling decisions happen for every span, making `_Matcher.attribute_match` and `_Matcher.wild_card_match` extremely hot paths. Iterating over span attributes (which can be numerous) when matching a small set of rule attributes is inefficient. Additionally, re-compiling regex or using regex for simple prefix/suffix matches adds significant overhead.

**Action:** Always optimize matching logic to iterate over the smaller set (usually rules), implement early-exit predicates, and provide fast-paths for common string patterns (prefix/suffix) to bypass the regex engine. Use `lru_cache` for regex compilation when patterns are reused.
