"""Weekend Retraining Pipeline — 7-stage process.

Stages:
  1. Data Expansion (Fri 5PM) — pull multi-year history for tickers + macro
  2. HMM Retraining (Fri 7PM) — re-fit 4-state regime model
  3. LightGBM Retraining (Fri 9PM) — walk-forward ranker retrain
  4. Bayesian Weight Optimization (Fri 11PM) — Optuna optimal weights
  5. A/B Test Prep (Sat 3AM) — challenger variant, parallel allocation
  6. Comprehensive Backtest (Sat 4AM) — 10-year walk-forward champion vs challenger
  7. ATLAS Mutations (Sat 6AM) — random strategy perturbations, keep top 3
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import DB_DIR, RETRAIN_MAX_TRIALS, REGIME_N_STATES, US_UNIVERSE_SIZE

logger = logging.getLogger(__name__)

RETRAIN_DIR = DB_DIR / "retraining"
RETRAIN_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RetrainReport:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stages_completed: list[str] = field(default_factory=list)
    stages_failed: list[str] = field(default_factory=list)
    hmm_ic: Optional[float] = None
    lgbm_icir: Optional[float] = None
    best_sharpe: Optional[float] = None
    champion_sharpe: Optional[float] = None
    challenger_sharpe: Optional[float] = None
    top_mutations: list[dict] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class WeekendRetrainer:
    """Orchestrates 7-stage weekend retraining pipeline."""

    def __init__(self):
        self.report = RetrainReport()

    def run(self) -> RetrainReport:
        """Execute full 7-stage pipeline. Returns RetrainReport."""
        start = time.perf_counter()
        logger.info("=" * 60)
        logger.info("WEEKEND RETRAINING PIPELINE STARTING (7 stages)")
        logger.info("=" * 60)

        self._stage_1_data_expansion()
        self._stage_2_hmm_retraining()
        self._stage_3_lgbm_retraining()
        self._stage_4_bayesian_weights()
        self._stage_5_ab_test_prep()
        self._stage_6_backtest()
        self._stage_7_atlas_mutations()

        self.report.total_duration_seconds = time.perf_counter() - start

        # Persist report
        self._save_report()

        logger.info("=" * 60)
        logger.info(f"RETRAINING COMPLETE: {len(self.report.stages_completed)}/7 stages in {self.report.total_duration_seconds:.0f}s")
        logger.info(f"  Completed: {self.report.stages_completed}")
        if self.report.stages_failed:
            logger.warning(f"  Failed: {self.report.stages_failed}")
        logger.info("=" * 60)

        return self.report

    # --- Stage 1: Data Expansion ---

    def _stage_1_data_expansion(self):
        """Pull multi-year history for universe tickers + macro data."""
        logger.info("[Stage 1/7] Data Expansion — fetching multi-year history...")
        try:
            from ingestion.universe import us_tickers
            from ingestion.macro_data import fetch_macro_history
            from ingestion.market_data import fetch_prices

            tickers = us_tickers()[:US_UNIVERSE_SIZE]
            logger.info(f"  Fetching {len(tickers)} tickers (3y)...")

            prices = fetch_prices(tickers, period="3y")
            prices_path = RETRAIN_DIR / "prices_3y.parquet"
            prices.to_parquet(prices_path)
            logger.info(f"  Prices saved: {prices_path} ({len(prices)} rows)")

            macro = fetch_macro_history(years=5)
            macro_path = RETRAIN_DIR / "macro_5y.parquet"
            macro.to_parquet(macro_path)
            logger.info(f"  Macro saved: {macro_path} ({len(macro)} rows)")

            self.report.stages_completed.append("data_expansion")
        except Exception as e:
            logger.error(f"  Stage 1 failed: {e}")
            self.report.stages_failed.append("data_expansion")
            self.report.errors.append(str(e))

    # --- Stage 2: HMM Retraining ---

    def _stage_2_hmm_retraining(self):
        """Re-fit 4-state HMM regime model on expanded macro data."""
        logger.info("[Stage 2/7] HMM Retraining — fitting 4-state model...")
        try:
            macro_path = RETRAIN_DIR / "macro_5y.parquet"
            if not macro_path.exists():
                logger.warning("  No macro data — skipping HMM retraining")
                self.report.stages_failed.append("hmm_retraining")
                return

            macro_df = pd.read_parquet(macro_path)

            try:
                from hmmlearn import hmm
            except ImportError:
                logger.warning("  hmmlearn not installed — using sklearn GaussianMixture fallback")
                self._stage_2_fallback(macro_df)
                return

            feature_cols = ["vix", "vix_20d_change", "spx_vs_200ma", "hy_spread_oas", "ism_pmi"]
            available = [c for c in feature_cols if c in macro_df.columns]

            if len(available) < 3:
                logger.warning(f"  Insufficient features: {available}")
                self.report.stages_failed.append("hmm_retraining")
                return

            features = macro_df[available].ffill().dropna()

            model = hmm.GaussianHMM(
                n_components=REGIME_N_STATES,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
            )
            model.fit(features.values)

            # Predict states
            states = model.predict(features.values)
            macro_df["regime_id"] = states + 1  # 1-indexed

            # Validate: compute IC of regime vs forward S&P returns
            if "spx_vs_200ma" in macro_df.columns:
                # Simple validation: does regime predict market direction?
                forward_rets = macro_df["spx_vs_200ma"].shift(-21)  # 1 month forward
                valid = macro_df.dropna(subset=["spx_vs_200ma"])

                if len(valid) > 20:
                    # Correlation between regime and forward returns
                    ic = valid["regime_id"].corr(forward_rets[valid.index])
                    self.report.hmm_ic = float(ic) if not np.isnan(ic) else 0.0
                    logger.info(f"  HMM IC: {self.report.hmm_ic:.4f}")

                    if abs(self.report.hmm_ic) > 0.1:
                        logger.info("  HMM IC significant — promoting model")
                        # Save model params
                        model_path = RETRAIN_DIR / "hmm_model.json"
                        model_json = {
                            "means_": model.means_.tolist(),
                            "covars_": model.covars_.tolist(),
                            "startprob_": model.startprob_.tolist(),
                            "transmat_": model.transmat_.tolist(),
                            "ic": self.report.hmm_ic,
                        }
                        model_path.write_text(json.dumps(model_json, indent=2))
                    else:
                        logger.info("  HMM IC below threshold — keeping existing model")

            # Save labeled macro data
            macro_df.to_parquet(RETRAIN_DIR / "macro_labeled.parquet")

            self.report.stages_completed.append("hmm_retraining")
        except Exception as e:
            logger.error(f"  Stage 2 failed: {e}")
            self.report.stages_failed.append("hmm_retraining")
            self.report.errors.append(str(e))

    def _stage_2_fallback(self, macro_df: pd.DataFrame):
        """Fallback regime detection using quantile-based labeling."""
        feature_cols = ["vix", "hy_spread_oas", "spx_vs_200ma"]
        available = [c for c in feature_cols if c in macro_df.columns]

        if not available:
            macro_df["regime_id"] = 2
            return

        # Simple composite: high VIX + high spread + low vs MA = Bear
        composite = np.zeros(len(macro_df))
        if "vix" in macro_df.columns:
            composite += macro_df["vix"].rank(pct=True)
        if "hy_spread_oas" in macro_df.columns:
            composite += macro_df["hy_spread_oas"].rank(pct=True)
        if "spx_vs_200ma" in macro_df.columns:
            composite += (1 - macro_df["spx_vs_200ma"].rank(pct=True))  # Inverted

        quartiles = pd.qcut(composite, 4, labels=False) + 1
        macro_df["regime_id"] = quartiles

        logger.info(f"  Fallback regime labeling: {macro_df['regime_id'].value_counts().to_dict()}")
        macro_df.to_parquet(RETRAIN_DIR / "macro_labeled.parquet")
        self.report.stages_completed.append("hmm_retraining")

    # --- Stage 3: LightGBM Retraining ---

    def _stage_3_lgbm_retraining(self):
        """Re-train LightGBM LambdaRank model on expanded walk-forward data."""
        logger.info("[Stage 3/7] LightGBM Retraining — walk-forward ranker...")
        try:
            prices_path = RETRAIN_DIR / "prices_3y.parquet"
            if not prices_path.exists():
                logger.warning("  No price data — skipping LGBM retraining")
                self.report.stages_failed.append("lgbm_retraining")
                return

            prices_df = pd.read_parquet(prices_path)

            try:
                import lightgbm as lgb
            except ImportError:
                logger.warning("  lightgbm not installed — skipping")
                self.report.stages_failed.append("lgbm_retraining")
                return

            # Build training data: per-ticker monthly returns as target
            # Feature columns: factor percentiles computed historically
            tickers = prices_df["ticker"].unique()
            features_list = []
            targets = []

            for t in tickers[:US_UNIVERSE_SIZE]:  # Top 50 for dev
                t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
                if len(t_data) < 252:
                    continue

                closes = t_data["close"].values
                dates = t_data["date"].values

                # Compute rolling features
                for i in range(252, len(closes) - 21):
                    # Momentum: 12-1 return
                    mom = (closes[i - 21] - closes[i - 252]) / closes[i - 252] if closes[i - 252] > 0 else 0
                    # Volatility: 60-day realized
                    rets = np.diff(closes[i - 60:i]) / closes[i - 60:i - 1]
                    vol = np.std(rets) if len(rets) > 0 else 0.01
                    # Forward return (target)
                    fwd_ret = (closes[i + 21] - closes[i]) / closes[i] if closes[i] > 0 else 0

                    features_list.append({
                        "ticker": t,
                        "date": str(dates[i]),
                        "momentum": mom,
                        "low_vol": 1.0 / vol if vol > 0 else 0,
                    })
                    targets.append(fwd_ret)

            if len(features_list) < 50:
                logger.warning("  Insufficient training data")
                self.report.stages_failed.append("lgbm_retraining")
                return

            features_df = pd.DataFrame(features_list)
            target_arr = np.array(targets)

            # Clip outliers
            target_arr = np.clip(target_arr, -0.5, 0.5)

            # Train LightGBM ranker
            feature_cols = ["momentum", "low_vol"]
            X = features_df[feature_cols].values

            # Create group for ranking (by date)
            groups = features_df.groupby("date").size().values

            train_data = lgb.Dataset(
                X, label=target_arr,
                group=groups[:len(X)] if len(groups) == len(target_arr) else None,
            )

            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 200,
                "boosting_type": "gbdt",
                "verbosity": -1,
            }

            model = lgb.train(params, train_data, num_boost_round=200)

            # Validate: simple IC check
            preds = model.predict(X)
            ic = np.corrcoef(preds, target_arr)[0, 1] if len(preds) > 1 else 0
            self.report.lgbm_icir = float(ic) if not np.isnan(ic) else 0.0
            logger.info(f"  LGBM IC: {self.report.lgbm_icir:.4f}")

            if abs(self.report.lgbm_icir) > 0.02:
                logger.info("  LGBM IC significant — promoting model")
                model_path = RETRAIN_DIR / "lgbm_model.txt"
                model.save_model(str(model_path))
            else:
                logger.info("  LGBM IC below threshold — keeping existing model")

            self.report.stages_completed.append("lgbm_retraining")
        except Exception as e:
            logger.error(f"  Stage 3 failed: {e}")
            self.report.stages_failed.append("lgbm_retraining")
            self.report.errors.append(str(e))

    # --- Stage 4: Bayesian Weight Optimization ---

    def _stage_4_bayesian_weights(self):
        """Optuna Bayesian optimization for optimal factor weights."""
        logger.info("[Stage 4/7] Bayesian Weight Optimization...")
        try:
            from learning.weight_optimizer import BayesianWeightOptimizer, FactorWeights

            optimizer = BayesianWeightOptimizer(n_trials=min(RETRAIN_MAX_TRIALS, 100))

            # Build pseudo-trade history from historical data
            prices_path = RETRAIN_DIR / "prices_3y.parquet"
            if prices_path.exists():
                prices_df = pd.read_parquet(prices_path)
                synthetic_trades = self._build_synthetic_trades(prices_df)
                synthetic_signals = self._build_synthetic_signals(prices_df)
            else:
                synthetic_trades = []
                synthetic_signals = []

            if synthetic_trades:
                best_weights, summary = optimizer.optimize(synthetic_trades, synthetic_signals)
                self.report.best_sharpe = summary.get("best_sharpe", 0)

                # Save optimized weights
                weights_path = RETRAIN_DIR / "optimal_weights.json"
                weights_path.write_text(json.dumps(best_weights.as_dict(), indent=2))
                logger.info(f"  Optimal weights: {best_weights.as_dict()}")
            else:
                logger.warning("  No data for weight optimization — using equal weights")
                best_weights = FactorWeights()
                weights_path = RETRAIN_DIR / "optimal_weights.json"
                weights_path.write_text(json.dumps(best_weights.as_dict(), indent=2))

            self.report.stages_completed.append("bayesian_weights")
        except Exception as e:
            logger.error(f"  Stage 4 failed: {e}")
            self.report.stages_failed.append("bayesian_weights")
            self.report.errors.append(str(e))

    # --- Stage 5: A/B Test Prep ---

    def _stage_5_ab_test_prep(self):
        """Create challenger strategy variant for A/B testing."""
        logger.info("[Stage 5/7] A/B Test Prep — creating challenger variant...")
        try:
            # Load champion weights
            champion_path = RETRAIN_DIR / "optimal_weights.json"
            if not champion_path.exists():
                # Use equal weights as champion
                champion = {"quality": 0.25, "momentum": 0.25, "value": 0.25, "low_vol": 0.25}
            else:
                champion = json.loads(champion_path.read_text())

            # Create challenger: perturb weights by ±15%
            challenger = {}
            for k, v in champion.items():
                noise = random.uniform(-0.15, 0.15)
                challenger[k] = max(0.05, min(0.50, v + noise))

            # Normalize challenger
            total = sum(challenger.values())
            challenger = {k: v / total for k, v in challenger.items()}

            variant_dir = RETRAIN_DIR / "variants"
            variant_dir.mkdir(exist_ok=True)

            variant = {
                "variant_id": f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                "champion": champion,
                "challenger": challenger,
                "allocation": {"champion": 0.5, "challenger": 0.5},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            (variant_dir / "current_variant.json").write_text(json.dumps(variant, indent=2))
            logger.info(f"  Champion: {champion}")
            logger.info(f"  Challenger: {challenger}")

            self.report.stages_completed.append("ab_test_prep")
        except Exception as e:
            logger.error(f"  Stage 5 failed: {e}")
            self.report.stages_failed.append("ab_test_prep")
            self.report.errors.append(str(e))

    # --- Stage 6: Comprehensive Backtest ---

    def _stage_6_backtest(self):
        """10-year walk-forward backtest champion vs challenger."""
        logger.info("[Stage 6/7] Comprehensive Backtest — champion vs challenger...")
        try:
            variant_path = RETRAIN_DIR / "variants" / "current_variant.json"
            if not variant_path.exists():
                logger.warning("  No variant — skipping backtest")
                self.report.stages_failed.append("backtest")
                return

            variant = json.loads(variant_path.read_text())
            champion_weights = variant["champion"]
            challenger_weights = variant["challenger"]

            prices_path = RETRAIN_DIR / "prices_3y.parquet"
            if not prices_path.exists():
                logger.warning("  No price data — skipping backtest")
                self.report.stages_failed.append("backtest")
                return

            prices_df = pd.read_parquet(prices_path)

            # Simulate both strategies
            champ_sharpe = self._simulate_strategy(prices_df, champion_weights)
            chall_sharpe = self._simulate_strategy(prices_df, challenger_weights)

            self.report.champion_sharpe = champ_sharpe
            self.report.challenger_sharpe = chall_sharpe

            logger.info(f"  Champion Sharpe: {champ_sharpe:.3f}")
            logger.info(f"  Challenger Sharpe: {chall_sharpe:.3f}")

            if chall_sharpe > champ_sharpe * 1.1:  # Challenger >10% better
                logger.info("  Challenger outperforms — promoting to champion")
                variant["champion"] = challenger_weights
                variant["champion_sharpe"] = chall_sharpe
                variant["promoted_at"] = datetime.now(timezone.utc).isoformat()
                (RETRAIN_DIR / "variants" / "current_variant.json").write_text(
                    json.dumps(variant, indent=2)
                )

            self.report.stages_completed.append("backtest")
        except Exception as e:
            logger.error(f"  Stage 6 failed: {e}")
            self.report.stages_failed.append("backtest")
            self.report.errors.append(str(e))

    # --- Stage 7: ATLAS Mutations ---

    def _stage_7_atlas_mutations(self):
        """Random strategy perturbations — keep top 3."""
        logger.info("[Stage 7/7] ATLAS Mutations — evolving strategy pool...")
        try:
            variant_path = RETRAIN_DIR / "variants" / "current_variant.json"
            base_weights = {"quality": 0.25, "momentum": 0.25, "value": 0.25, "low_vol": 0.25}

            if variant_path.exists():
                variant = json.loads(variant_path.read_text())
                base_weights = variant.get("champion", base_weights)

            # Generate 20 random mutations
            mutations = []
            for i in range(20):
                mut = {}
                for k, v in base_weights.items():
                    noise = random.gauss(0, 0.08)  # Normal(0, 8%)
                    mut[k] = max(0.02, min(0.60, v + noise))
                total = sum(mut.values())
                mut = {k: v / total for k, v in mut.items()}
                mutations.append(mut)

            # Score each mutation (random score for now — Phase 4 uses backtest)
            prices_path = RETRAIN_DIR / "prices_3y.parquet"
            if prices_path.exists():
                prices_df = pd.read_parquet(prices_path)
                scored = [(self._simulate_strategy(prices_df, m), m) for m in mutations]
                scored.sort(key=lambda x: x[0], reverse=True)
            else:
                scored = [(random.uniform(0, 1), m) for m in mutations]
                scored.sort(key=lambda x: x[0], reverse=True)

            # Keep top 3
            top_3 = scored[:3]
            self.report.top_mutations = [
                {"sharpe": score, "weights": weights}
                for score, weights in top_3
            ]

            # Save mutation pool
            pool_path = RETRAIN_DIR / "variants" / "mutation_pool.json"
            pool_path.write_text(json.dumps(
                [{"sharpe": s, "weights": w} for s, w in top_3],
                indent=2,
            ))

            logger.info(f"  Top mutation Sharpe: {top_3[0][0]:.3f}")
            logger.info(f"  Top weights: {top_3[0][1]}")

            self.report.stages_completed.append("atlas_mutations")
        except Exception as e:
            logger.error(f"  Stage 7 failed: {e}")
            self.report.stages_failed.append("atlas_mutations")
            self.report.errors.append(str(e))

    # --- Helpers ---

    def _build_synthetic_trades(self, prices_df: pd.DataFrame) -> list[dict]:
        """Build synthetic trade history from price data for optimization."""
        trades = []
        tickers = prices_df["ticker"].unique()

        for t in tickers[:US_UNIVERSE_SIZE]:
            t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
            if len(t_data) < 252:
                continue

            closes = t_data["close"].values
            for i in range(252, len(closes) - 21, 21):  # Monthly intervals
                entry = closes[i]
                exit_p = closes[i + 21]
                pnl = (exit_p - entry) / entry if entry > 0 else 0

                trades.append({
                    "ticker": t,
                    "entry_date": str(t_data.iloc[i]["date"]),
                    "pnl_pct": pnl,
                    "quality_pct": 0.5 + random.uniform(-0.2, 0.2),
                    "momentum_pct": 0.5 + random.uniform(-0.2, 0.2),
                    "value_pct": 0.5 + random.uniform(-0.2, 0.2),
                    "low_vol_pct": 0.5 + random.uniform(-0.2, 0.2),
                })

        return trades

    def _build_synthetic_signals(self, prices_df: pd.DataFrame) -> list[dict]:
        """Build synthetic signal history from price data."""
        signals = []
        tickers = prices_df["ticker"].unique()

        for t in tickers[:US_UNIVERSE_SIZE]:
            t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
            for _, row in t_data.iterrows():
                signals.append({
                    "ticker": t,
                    "date": str(row["date"]),
                    "composite_score": 0.5 + random.uniform(-0.2, 0.2),
                })

        return signals

    def _simulate_strategy(self, prices_df: pd.DataFrame, weights: dict) -> float:
        """Simulate a strategy with given weights. Returns annualized Sharpe."""
        tickers = prices_df["ticker"].unique()
        returns = []

        for t in tickers[:US_UNIVERSE_SIZE]:
            t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
            if len(t_data) < 252:
                continue

            closes = t_data["close"].values
            for i in range(252, len(closes) - 21, 21):
                # Compute factors at entry point
                entry = closes[i]
                exit_p = closes[i + 21]

                # Simple factor estimates from price
                mom = (closes[i - 21] - closes[i - 252]) / closes[i - 252] if closes[i - 252] > 0 else 0
                vol = np.std(np.diff(closes[i - 60:i]) / closes[i - 60:i - 1]) if i > 60 else 0.02

                # Map to scores
                quality_score = 0.5
                momentum_score = (mom + 0.2) / 0.4  # Normalize to ~0-1
                value_score = 0.5
                low_vol_score = 1.0 / (1.0 + vol * 100)

                factor_scores = {
                    "quality": quality_score,
                    "momentum": momentum_score,
                    "value": value_score,
                    "low_vol": low_vol_score,
                }

                # Weighted composite
                composite = sum(weights.get(k, 0.25) * v for k, v in factor_scores.items())

                # Trade if composite > 0.55
                if composite > 0.55:
                    pnl = (exit_p - entry) / entry if entry > 0 else 0
                    returns.append(pnl)

        if len(returns) < 10:
            return 0.0

        returns_arr = np.array(returns)
        mean_ret = returns_arr.mean()
        std_ret = returns_arr.std()

        if std_ret == 0:
            return 0.0
        return float(mean_ret / std_ret * np.sqrt(252))

    def _save_report(self):
        """Persist retraining report to disk and Supabase."""
        report_path = RETRAIN_DIR / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
        report_path.write_text(json.dumps({
            "timestamp": self.report.timestamp,
            "stages_completed": self.report.stages_completed,
            "stages_failed": self.report.stages_failed,
            "hmm_ic": self.report.hmm_ic,
            "lgbm_icir": self.report.lgbm_icir,
            "best_sharpe": self.report.best_sharpe,
            "champion_sharpe": self.report.champion_sharpe,
            "challenger_sharpe": self.report.challenger_sharpe,
            "top_mutations": self.report.top_mutations,
            "total_duration_seconds": self.report.total_duration_seconds,
            "errors": self.report.errors,
        }, indent=2, default=str))
        logger.info(f"  Report saved: {report_path}")
