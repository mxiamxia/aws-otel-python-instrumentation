## 2025-05-15 - [Wildcard Matching & Metadata Parsing Optimizations]
**Learning:** Wildcard matching using regular expressions is expensive for simple prefix/suffix/substring patterns which are common in sampling rules. Implementing fast-paths with `startswith`, `endswith`, and `in` can yield ~6x speedups. Additionally, telemetry metadata like ARNs and SQS URLs are highly repetitive, making `lru_cache` extremely effective for parsing logic.

**Action:** Always look for fast-paths in string matching logic before falling back to regex. Apply caching to parsing functions that handle repetitive identifier strings.

## 2025-05-15 - [Regex & String Allocation Optimizations]
**Learning:** Repeatedly calling `.upper()` or `.casefold()` on long strings (like SQL statements) in hot paths creates unnecessary allocations. Using `re.IGNORECASE` in a pre-compiled regex and only uppercasing the match result is significantly more efficient.

**Action:** Pre-compile regexes with appropriate flags and avoid broad string transformations on inputs when a targeted transformation on the match result suffices.
