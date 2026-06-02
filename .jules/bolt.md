## 2026-06-02 - Optimize _Matcher performance
**Learning:** Flipping the iteration logic in `attribute_match` from O(Attributes) to O(RuleAttributes) provides a massive performance win for spans with many attributes, especially when combined with early-return and `lru_cache` for regex.
**Action:** Always check if the set being iterated over can be switched to a smaller, fixed set of requirements.
