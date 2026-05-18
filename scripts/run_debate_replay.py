"""Run PARA-DEBATE historical replay — validates AI decision-making on past data.

This runs actual Claude API calls and costs money (~$10-30 per full run).
Use --dry-run to see what WOULD be debated without making API calls.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.historical_replay import HistoricalReplay


def main():
    parser = argparse.ArgumentParser(description="Run PARA-DEBATE on historical data")
    parser.add_argument("--sample-interval", type=int, default=30,
                        help="Trading days between debate samples (default: 30)")
    parser.add_argument("--lookback-years", type=int, default=3,
                        help="Years back from today to sample (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be debated without calling Claude")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-debate output")
    args = parser.parse_args()

    replay = HistoricalReplay()

    if args.dry_run:
        prices_df = replay._load_or_fetch_prices()
        prices_df = replay._compute_factor_scores(prices_df)
        all_dates = sorted(prices_df["date"].unique())
        import pandas as pd
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=args.lookback_years)
        recent_dates = [d for d in all_dates if pd.Timestamp(d) >= cutoff]
        sample_dates = recent_dates[::args.sample_interval]

        print(f"DRY RUN — {len(sample_dates)} debates would run (no API calls)")
        print(f"Date range: {sample_dates[0] if sample_dates else 'N/A'} → {sample_dates[-1] if sample_dates else 'N/A'}")
        print(f"Estimated cost: ~${len(sample_dates) * 5.5 * 0.00025 + len(sample_dates) * 2 * 0.003:.2f}")
        print(f"  ({len(sample_dates)} debates × 5.5 Haiku calls + 2 Sonnet calls each)")
        print()

        for i, date_str in enumerate(sample_dates[:10]):
            day_data = prices_df[prices_df["date"] == date_str]
            signals = replay._compute_historical_signals(day_data)
            if not signals.empty:
                top = signals.head(1).iloc[0]
                print(f"  [{i+1}] {str(date_str)[:10]} | {top['ticker']} | score={top['composite_score']:.3f}")
        if len(sample_dates) > 10:
            print(f"  ... and {len(sample_dates) - 10} more")
    else:
        print("Running PARA-DEBATE historical replay (this will cost money)...")
        print(f"Sample interval: {args.sample_interval} days, Lookback: {args.lookback_years} years")
        print("Ctrl+C to cancel within 3 seconds...")
        import time
        time.sleep(3)

        results = replay.run_debate_replay(
            sample_interval=args.sample_interval,
            lookback_years=args.lookback_years,
            verbose=not args.quiet,
        )

        if results:
            from learning.historical_replay import DEBATE_RESULTS_FILE
            print(f"\nSaved {len(results)} debate results to {DEBATE_RESULTS_FILE}")
        else:
            print("\nNo debate results produced.")


if __name__ == "__main__":
    main()
