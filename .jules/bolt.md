## 2026-05-15 - Fast-paths for wildcard matching and regex pre-compilation
**Learning:** Re-compiling regular expressions or converting wildcard patterns to regex on every call in high-frequency telemetry paths (like sampling or attribute generation) is a significant bottleneck. Fast-paths for common patterns ('*', 'prefix*', '*suffix', '*middle*') using native string methods can provide up to 80% speedup.
**Action:** Always implement fast-paths for wildcard matches and pre-compile large static regular expressions at the module level.
