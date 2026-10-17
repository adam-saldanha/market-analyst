"""CLI: python -m market_analyst.main AAPL [--stub]"""
import argparse
import json
import time

from .graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="Generate an investment memo for a ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--stub", action="store_true", help="run with canned data (no API calls)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = build_graph().invoke({"ticker": args.ticker.upper(), "stub": args.stub, "timings": {}})
    elapsed = time.perf_counter() - t0

    print(json.dumps(result["memo"].model_dump(), indent=2))
    print(f"\nStep timings: {result.get('timings', {})}")
    print(f"Total: {elapsed:.2f}s  (goal: < 15s)")


if __name__ == "__main__":
    main()
