"""Smoke test all Phase 4 learning modules with mock trade data."""
from learning.feedback import FeedbackLoop
from learning.decay import DecayDetector
from learning.attribution import attribute_trade, compute_factor_ic, compute_hit_rate
from learning.weight_optimizer import FactorWeights, OnlineWeightOptimizer

# Mock closed trades
trades = [
    {"trade_id": "t1", "ticker": "AAPL", "pnl_pct": 0.03, "composite_score": 0.72,
     "quality_pct": 0.68, "momentum_pct": 0.81, "value_pct": 0.55, "low_vol_pct": 0.62,
     "regime_id": 1, "exit_reason": "take_profit", "status": "CLOSED", "pnl_usd": 150.0},
    {"trade_id": "t2", "ticker": "MSFT", "pnl_pct": -0.02, "composite_score": 0.58,
     "quality_pct": 0.75, "momentum_pct": 0.45, "value_pct": 0.52, "low_vol_pct": 0.70,
     "regime_id": 2, "exit_reason": "stop_loss", "status": "CLOSED", "pnl_usd": -100.0},
    {"trade_id": "t3", "ticker": "GOOGL", "pnl_pct": 0.05, "composite_score": 0.85,
     "quality_pct": 0.72, "momentum_pct": 0.90, "value_pct": 0.48, "low_vol_pct": 0.55,
     "regime_id": 1, "exit_reason": "take_profit", "status": "CLOSED", "pnl_usd": 250.0},
    {"trade_id": "t4", "ticker": "NVDA", "pnl_pct": -0.04, "composite_score": 0.62,
     "quality_pct": 0.60, "momentum_pct": 0.88, "value_pct": 0.35, "low_vol_pct": 0.40,
     "regime_id": 3, "exit_reason": "stop_loss", "status": "CLOSED", "pnl_usd": -200.0},
    {"trade_id": "t5", "ticker": "TSLA", "pnl_pct": 0.01, "composite_score": 0.55,
     "quality_pct": 0.45, "momentum_pct": 0.52, "value_pct": 0.60, "low_vol_pct": 0.48,
     "regime_id": 2, "exit_reason": "trailing_stop", "status": "CLOSED", "pnl_usd": 50.0},
]

print("=== Attribution ===")
attr = attribute_trade(trades[0], {})
print(f"  {attr['ticker']} pnl={attr['pnl_pct']:.2%}")
contribs = {k: round(v, 3) for k, v in attr["factor_contributions"].items()}
print(f"  Contributions: {contribs}")

print("\n=== Factor IC ===")
ic = compute_factor_ic(trades, None)
print(f"  IC: {dict((k, round(v, 3)) for k, v in ic.items())}")

print("\n=== Hit Rate ===")
hit = compute_hit_rate(trades)
print(f"  Hit rate: {dict((k, round(v, 3)) for k, v in hit.items())}")

print("\n=== Online Weight Optimizer ===")
opt = OnlineWeightOptimizer()
opt.update(ic)
print(f"  Updated weights: {opt.weights_dict}")

print("\n=== Feedback Loop ===")
fb = FeedbackLoop()
result = fb.process_closed_trades(trades)
print(f"  Trades: {result['trades_processed']}, Decay: {result['decay_level']}")
print(f"  Final weights: {result['updated_weights']}")

print("\n=== Decay Detector ===")
detector = DecayDetector()
decay = detector.check(trades, signal_scores=[0.72, 0.58, 0.85, 0.62, 0.55])
print(f"  Level: {decay.level.value} ({decay.alert_count}/3 detectors)")
print(f"  CUSUM: {decay.cusum_triggered} ({decay.cusum_value:.2f})")
print(f"  Sharpe: {decay.sharpe_triggered} ({decay.sharpe_value:.3f})")
print(f"  PSI: {decay.psi_triggered} ({decay.psi_value:.3f})")
if decay.details:
    for d in decay.details:
        print(f"  -> {d}")

print("\n=== State Snapshot ===")
state = fb.get_current_state()
print(f"  Total trades: {state['total_trades']}")
print(f"  Total attributions: {state['total_attributions']}")
print(f"  Optimizer updates: {state['optimizer_updates']}")

print("\nAll smoke tests passed!")
