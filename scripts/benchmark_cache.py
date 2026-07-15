"""Print the naive-vs-LMCache KV cache memory comparison referenced in
the README, for the configured model geometry and workload shape.

Usage:
    python scripts/benchmark_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from core.cache_manager import CacheManager  # noqa: E402


def main() -> None:
    settings = get_settings()
    cm = CacheManager(settings)

    bpt = cm.kv_bytes_per_token()
    print(f"Model geometry: {settings.num_layers} layers, {settings.num_kv_heads} kv heads, "
          f"head_dim={settings.head_dim}, dtype_bytes={settings.kv_dtype_bytes}")
    print(f"KV bytes/token: {bpt} bytes ({bpt/1024:.2f} KB)\n")

    naive = cm.estimate_naive_kv_gb(num_sessions=100, num_agents=4)
    shared = cm.estimate_shared_prefix_kv_gb(num_sessions=100, num_agents=4, active_concurrency=8)

    print(f"Naive (no sharing, all 100 sessions' full prompts resident):  {naive:>8.2f} GB")
    print(f"LMCache (shared prefixes + 8-session active working set):    {shared:>8.2f} GB")
    print(f"Configured GPU KV-cache budget:                              {settings.kv_cache_budget_gb:>8.2f} GB")
    print(f"\nReduction factor: {naive / shared:.1f}x")
    print(f"Fits in budget: {shared <= settings.kv_cache_budget_gb}")


if __name__ == "__main__":
    main()
