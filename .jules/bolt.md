## 2026-07-05 - [Optimized _Matcher in Sampling Path]
**Learning:** The sampling path is a hot path where even small overheads like regex compilation or iterating over large attribute dictionaries add up. Iterating over rule attributes (few) instead of span attributes (many) and using early-return significantly reduces match time.
**Action:** Always check the relative sizes of data structures being compared and iterate over the smaller one when possible. Use `lru_cache` for regex compilation in repetitive matching tasks.
