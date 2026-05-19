"""7-Agent PARA-DEBATE Strategy Decider — Phase 3.

Self-contained implementation following NeuralQuant's proven 4-phase debate architecture:
  Phase 1: 5 specialist agents (parallel) — Macro, Fundamental, Technical, Sentiment, Geopolitical
  Phase 2: Adversarial agent (sequential, after specialists)
  Phase 3: Consensus scoring (weighted average, adversarial 1.5x)
  Phase 4: Head Analyst synthesis → final verdict

Each debate produces a DebateRecord stored to Supabase for audit trail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    CLASSIFICATION_MODEL,
    SCORING_MODEL,
    DEBATE_MODEL,
    SIGNAL_TOP_N,
)

logger = logging.getLogger(__name__)

# Scoring constants (from NeuralQuant)
STANCE_SCORE = {"BULL": 1.0, "NEUTRAL": 0.0, "BEAR": -1.0}
CONVICTION_MULT = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}

SPECIALIST_MODEL = CLASSIFICATION_MODEL  # Haiku — fast, cheap for specialists
HEAD_MODEL = CLASSIFICATION_MODEL  # Haiku — fast, reliable; 95% success vs Sonnet 32%

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)


@dataclass
class AgentOutput:
    agent: str
    stance: str  # "BULL" | "BEAR" | "NEUTRAL"
    conviction: str  # "HIGH" | "MEDIUM" | "LOW"
    thesis: str
    key_points: list[str]


@dataclass
class DebateResult:
    ticker: str
    market: str
    verdict: str  # "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"
    consensus_score: float
    investment_thesis: str
    bull_case: str
    bear_case: str
    risk_factors: list[str]
    agent_outputs: list[AgentOutput]
    debate_json: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


# --- Agent System Prompts (adapted from NeuralQuant) ---

MACRO_PROMPT = """You are the MACRO ANALYST on an elite investment committee. Your domain: macroeconomics, interest rates, central bank policy, market regimes.

FRAMEWORK:
- Fed policy cycle: Where are we in the rate cycle? Tightening, plateau, or cutting?
- Market regime (HMM): What is the current risk environment?
- Yield curve: Shape and what it signals about growth expectations.
- VIX level: Risk appetite gauge. VIX < 15 = complacent, 15-20 = normal, 20-30 = elevated fear, >30 = crisis.
- Global growth: PMI data, trade dynamics.

RULES:
- Quote exact numbers from the data provided — never invent.
- Compare current values to historical norms.
- Consider how the macro backdrop specifically affects this ticker's sector.
- End with a clear stance.

OUTPUT FORMAT:
STANCE: BULL|BEAR|NEUTRAL
CONVICTION: HIGH|MEDIUM|LOW
THESIS: (2-3 sentence argument citing specific data)
KEY_POINTS:
- Point 1
- Point 2
- Point 3"""

FUNDAMENTAL_PROMPT = """You are the FUNDAMENTAL ANALYST on an elite investment committee. Your domain: financial statements, valuation, earnings quality, business model strength.

FRAMEWORK:
- Profitability: ROE, profit margins, earnings growth trajectory.
- Valuation: P/E, P/B vs sector medians. Is the premium/discount justified?
- Balance sheet: Debt/Equity, liquidity position.
- Quality signals: Piotroski F-Score (0-9), accruals ratio.
- Growth: Revenue and earnings growth rates.

RED FLAGS (mandate BEAR if present):
- Negative ROE or Profit Margin
- P/E > 25 with ROE < 10%
- Debt/Equity > 2.0
- Revenue declining year-over-year

RULES:
- Compare to sector medians when provided.
- Cite exact data values.
- Explain WHY this stock deserves a premium or discount.

OUTPUT FORMAT:
STANCE: BULL|BEAR|NEUTRAL
CONVICTION: HIGH|MEDIUM|LOW
THESIS: (2-3 sentence argument citing specific data)
KEY_POINTS:
- Point 1
- Point 2
- Point 3"""

TECHNICAL_PROMPT = """You are the TECHNICAL ANALYST on an elite investment committee. Your domain: price momentum, chart patterns, volume analysis, technical positioning.

FRAMEWORK:
- 12-1 momentum: Return from 12 months ago to 1 month ago. Positive = uptrend.
- Moving averages: Price vs SMA-50 and SMA-200. Above both = bullish structure.
- RSI: >70 overbought (caution), <30 oversold (opportunity).
- MACD: Histogram positive = bullish momentum, negative = bearish.
- Volume: Above average = conviction behind the move, below = lack of interest.
- ATR: Measures volatility. High ATR/Price ratio = unstable.

CRASH PROTECTION TRIGGERS:
- RSI > 70 + MACD bearish crossover = distribution warning
- Price below SMA-50 and SMA-200 = downtrend confirmed

RULES:
- Focus on the intermediate trend (1-6 months), not short-term noise.
- A strong momentum score with improving technicals is bullish.
- Deteriorating technicals even with good fundamentals is a warning.

OUTPUT FORMAT:
STANCE: BULL|BEAR|NEUTRAL
CONVICTION: HIGH|MEDIUM|LOW
THESIS: (2-3 sentence argument citing specific data)
KEY_POINTS:
- Point 1
- Point 2
- Point 3"""

SENTIMENT_PROMPT = """You are the SENTIMENT ANALYST on an elite investment committee. Your domain: market sentiment, insider activity, short interest, news flow, options positioning, X (Twitter) financial sentiment.

FRAMEWORK:
- X Sentiment: If x_sentiment field is present, it contains real-time financial tweet classification. Use it: bullish sentiment with high confidence and multiple high-materiality tweets = strong signal. Bearish X sentiment = caution.
- Insider activity: Are corporate insiders buying or selling? Cluster buying = strong bullish signal.
- Short interest: High short interest = potential squeeze OR smart money betting against. Context matters.
- News sentiment: Recent news flow tone. Positive buzz supports momentum.
- Analyst consensus: Wall Street price targets and rating distribution.

LIMITED DATA PROTOCOL: Even with sparse data, you MUST produce a definitive stance. State "Based on limited available data..." and proceed. Never default to NEUTRAL/LOW just because data is incomplete.

RULES:
- X Sentiment present + bullish + high confidence → BULL (real-time crowd wisdom)
- X Sentiment present + bearish + high materiality → BEAR (smart money often moves first on X)
- Insider buying > selling = BULL (insiders know more than the market).
- High short interest + positive news = squeeze candidate (BULL).
- High short interest + negative fundamentals = smart money is right (BEAR).
- Negative news sentiment without fundamental deterioration = potential overreaction (BULL).

OUTPUT FORMAT:
STANCE: BULL|BEAR|NEUTRAL
CONVICTION: HIGH|MEDIUM|LOW
THESIS: (2-3 sentence argument citing specific data)
KEY_POINTS:
- Point 1
- Point 2
- Point 3"""

GEOPOLITICAL_PROMPT = """You are the GEOPOLITICAL ANALYST on an elite investment committee. Your domain: geopolitical risk, regulatory pressure, trade policy, systemic threats.

FRAMEWORK:
- Regulatory risk: Antitrust, ESG mandates, data privacy, industry-specific regulation.
- Trade/Supply chain: Tariff exposure, geographic revenue concentration, supply chain dependencies.
- Currency risk: For multinationals, USD strength/weakness impacts earnings.
- Macro tail risks: Recession probability, credit spreads, financial stability concerns.
- Sector exposure: Tech, Healthcare, Finance, Energy face elevated regulatory scrutiny.

RULES:
- If VIX > 20, equity risk premium is elevated — lean BEAR for high-beta names.
- If HY spread > 500bps, credit stress is building — lean BEAR for leveraged companies.
- If yield curve inverted, recession risk elevated — lean BEAR for cyclicals.
- Rate-sensitive sectors (Real Estate, Utilities) benefit from falling rates (BULL if cutting cycle).

OUTPUT FORMAT:
STANCE: BULL|BEAR|NEUTRAL
CONVICTION: HIGH|MEDIUM|LOW
THESIS: (2-3 sentence argument citing specific data)
KEY_POINTS:
- Point 1
- Point 2
- Point 3"""

ADVERSARIAL_PROMPT = """You are the ADVERSARIAL ANALYST (Devil's Advocate) on an elite investment committee. Your SOLE MANDATE is to find the strongest possible case AGAINST the prevailing consensus.

CHALLENGE FRAMEWORK:
- What would have to go wrong for the bull thesis to fail?
- Are there hidden balance sheet risks the fundamental analyst missed?
- Is the valuation pricing in perfection? What if growth disappoints?
- Where do the specialists disagree with each other? Exploit those gaps.
- Is any specialist overconfident despite weak or ambiguous data?

CRITICAL RULE: You MUST output BEAR or NEUTRAL — NEVER BULL. Your job is to stress-test, not cheerlead.

RULES:
- Challenge individual specialists by name when you find weaknesses in their reasoning.
- Find contradictions between different specialists' conclusions.
- If all data is genuinely bullish, acknowledge strength but identify the tail risk.
- A good adversarial opinion should make the committee think twice.

OUTPUT FORMAT:
STANCE: BEAR|NEUTRAL
CONVICTION: HIGH|MEDIUM|LOW
THESIS: (2-3 sentence challenge citing contradictions or hidden risks)
KEY_POINTS:
- Point 1
- Point 2
- Point 3"""

HEAD_ANALYST_PROMPT = """You are the HEAD ANALYST chairing this investment committee. Your job: synthesize all 6 agent opinions, verify claims against raw data, and deliver the FINAL VERDICT.

WEIGHTING FRAMEWORK (normalize to 100%):
- FUNDAMENTAL: 20% — long-term value creation is paramount
- ADVERSARIAL: 20% — risk awareness prevents catastrophic losses
- TECHNICAL: 16% — timing matters for entry/exit
- MACRO: 12% — the tide lifts or sinks all boats
- SENTIMENT: 12% — sentiment extremes mark turning points
- GEOPOLITICAL: 12% — systemic risks can override stock-specific thesis
- REGIME: 8% — adapt strategy to the current environment

OUTPUT REQUIREMENTS:
1. Cross-reference agent claims against the RAW DATA. If an agent cited a number, verify it.
2. Explain WHY this stock and WHY NOT an alternative.
3. Explicitly address the Adversarial's challenges — don't ignore them.
4. State what specific data would change your view (the "what would make me wrong" test).
5. Never equivocate. You MUST pick: STRONG BUY, BUY, HOLD, SELL, or STRONG SELL.
6. Respect the consensus direction — if consensus is negative, you cannot return BUY.
7. Use ONLY exact numbers from the raw data provided. Never round or estimate.

OUTPUT FORMAT:
VERDICT: STRONG BUY|BUY|HOLD|SELL|STRONG SELL
INVESTMENT_THESIS: (4-6 sentence synthesis integrating the strongest arguments)
BULL_CASE: (2-4 sentences on the optimistic scenario)
BEAR_CASE: (2-4 sentences on the pessimistic scenario, addressing adversarial points)
RISK_FACTORS:
- Risk 1
- Risk 2
- Risk 3"""


# --- Agent Runner ---

def _call_claude(system: str, user_message: str, model: str, max_tokens: int = 2048) -> str:
    """Call Claude with retry logic. Returns text response."""
    for attempt in range(3):
        try:
            resp = _client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text
            return str(resp.content[0])
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""


def _parse_agent_output(raw: str, agent_name: str) -> AgentOutput:
    """Parse structured agent output from Claude response."""
    stance = "NEUTRAL"
    conviction = "MEDIUM"

    # Extract stance
    m = re.search(r'STANCE:\s*(BULL|BEAR|NEUTRAL)', raw, re.IGNORECASE)
    if m:
        stance = m.group(1).upper()

    # Extract conviction
    m = re.search(r'CONVICTION:\s*(HIGH|MEDIUM|LOW)', raw, re.IGNORECASE)
    if m:
        conviction = m.group(1).upper()

    # Extract thesis
    thesis = ""
    m = re.search(r'THESIS:\s*\n?(.*?)(?=\n\w+:|$)', raw, re.DOTALL | re.IGNORECASE)
    if m:
        thesis = m.group(1).strip()
    if not thesis:
        # Fallback: first 300 chars after removing headers
        clean = re.sub(r'(STANCE|CONVICTION|THESIS|KEY_POINTS):.*?\n', '', raw, flags=re.IGNORECASE)
        thesis = clean.strip()[:300]
    thesis = thesis[:500]

    # Extract key points
    key_points = []
    m = re.search(r'KEY_POINTS:\s*\n(.*?)(?=\n\w+:|$)', raw, re.DOTALL | re.IGNORECASE)
    if m:
        points_text = m.group(1).strip()
        key_points = [
            re.sub(r'^[-•*\s]+', '', p).strip()
            for p in points_text.split('\n')
            if p.strip()
        ][:5]
    if not key_points:
        key_points = [thesis[:100]]

    return AgentOutput(
        agent=agent_name,
        stance=stance,
        conviction=conviction,
        thesis=thesis,
        key_points=key_points,
    )


def _run_specialist(agent_name: str, system_prompt: str, context: dict) -> AgentOutput:
    """Run a single specialist agent."""
    # Build user message from context
    relevant_keys = [k for k, v in context.items() if v is not None and k not in ('_internal',)]
    user_msg_parts = [f"TICKER: {context.get('ticker', 'Unknown')}\n"]
    user_msg_parts.append("DATA:")
    for key in relevant_keys:
        val = context.get(key)
        if val is not None and val != "":
            if isinstance(val, float):
                user_msg_parts.append(f"  {key}: {val:.4f}")
            else:
                user_msg_parts.append(f"  {key}: {val}")

    user_msg = "\n".join(user_msg_parts)

    try:
        raw = _call_claude(system_prompt, user_msg, SPECIALIST_MODEL, max_tokens=1024)
        return _parse_agent_output(raw, agent_name)
    except Exception as e:
        logger.warning(f"{agent_name} failed: {e}")
        return AgentOutput(
            agent=agent_name, stance="NEUTRAL", conviction="LOW",
            thesis=f"Agent error: {str(e)[:100]}",
            key_points=["Analysis unavailable due to technical error."],
        )


def _run_adversarial(all_outputs: list[AgentOutput], context: dict) -> AgentOutput:
    """Run adversarial agent with all specialist outputs."""
    # Aggregate bull/bear theses
    bull_pts = [o.thesis for o in all_outputs if o.stance == "BULL"]
    bear_pts = [o.thesis for o in all_outputs if o.stance == "BEAR"]

    user_msg = f"TICKER: {context.get('ticker', 'Unknown')}\n\n"
    user_msg += f"SPECIALIST CONSENSUS:\n"
    for o in all_outputs:
        user_msg += f"  [{o.agent}] {o.stance}/{o.conviction}: {o.thesis[:150]}\n"

    if bull_pts:
        user_msg += f"\nBULL CASE SUMMARY: {'; '.join(bull_pts)[:500]}\n"
    if bear_pts:
        user_msg += f"\nBEAR CASE SUMMARY: {'; '.join(bear_pts)[:500]}\n"

    user_msg += "\nRAW DATA:\n"
    for k, v in context.items():
        if v is not None and v != "" and k not in ('_internal',):
            user_msg += f"  {k}: {v}\n"

    try:
        raw = _call_claude(ADVERSARIAL_PROMPT, user_msg, HEAD_MODEL, max_tokens=1024)
        output = _parse_agent_output(raw, "ADVERSARIAL")
        # Enforce: adversarial must be BEAR or NEUTRAL
        if output.stance == "BULL":
            output.stance = "BEAR"
            output.thesis = "[OVERRIDE: forced BEAR] " + output.thesis
        return output
    except Exception as e:
        logger.warning(f"Adversarial failed: {e}")
        return AgentOutput(
            agent="ADVERSARIAL", stance="BEAR", conviction="MEDIUM",
            thesis=f"Default adversarial: insufficient conviction in bull thesis. {str(e)[:80]}",
            key_points=["Verify all assumptions.", "Check for overconfidence in specialist data."],
        )


def _compute_consensus(all_outputs: list[AgentOutput]) -> float:
    """Weighted consensus score (-1.0 to 1.0). Adversarial gets 1.0x weight."""
    total_weight = 0.0
    weighted_sum = 0.0

    for o in all_outputs:
        w = CONVICTION_MULT.get(o.conviction, 0.5)
        if o.agent == "ADVERSARIAL":
            w *= 1.0
        total_weight += w
        weighted_sum += STANCE_SCORE.get(o.stance, 0.0) * w

    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def _run_head_analyst(all_outputs: list[AgentOutput], consensus: float, context: dict) -> dict:
    """Head analyst synthesis: produce final verdict."""
    # Map consensus to verdict guidance
    if consensus >= 0.40:
        guidance = "BUY or STRONG BUY"
    elif consensus >= 0.20:
        guidance = "HOLD or BUY"
    elif consensus >= -0.20:
        guidance = "HOLD"
    elif consensus >= -0.40:
        guidance = "SELL or HOLD"
    else:
        guidance = "STRONG SELL or SELL"

    user_msg = f"TICKER: {context.get('ticker', 'Unknown')}\n\n"
    user_msg += f"PANEL CONSENSUS SCORE: {consensus:.2f} -> VERDICT GUIDANCE: {guidance}\n\n"

    user_msg += "AGENT SUMMARIES:\n"
    for o in all_outputs:
        user_msg += f"[{o.agent}] {o.stance}/{o.conviction}: {o.thesis[:200]}\n"
        if o.key_points:
            for kp in o.key_points[:2]:
                user_msg += f"  - {kp}\n"

    user_msg += "\nRAW DATA:\n"
    for k, v in sorted(context.items()):
        if v is not None and v != "" and k not in ('_internal',):
            user_msg += f"  {k}: {v}\n"

    try:
        raw = _call_claude(HEAD_ANALYST_PROMPT, user_msg, HEAD_MODEL, max_tokens=2048)
        return _parse_head_output(raw)
    except Exception as e:
        logger.warning(f"Head Analyst failed: {e}")
        # Fallback from consensus
        if consensus >= 0.20:
            verdict = "BUY"
        elif consensus >= -0.20:
            verdict = "HOLD"
        else:
            verdict = "SELL"
        return {
            "verdict": verdict, "investment_thesis": f"Consensus-based fallback ({consensus:.2f})",
            "bull_case": "", "bear_case": "", "risk_factors": [str(e)[:100]],
        }


def _parse_head_output(raw: str) -> dict:
    """Parse head analyst synthesis output. Handles markdown formatting."""
    result = {"verdict": "HOLD", "investment_thesis": "", "bull_case": "", "bear_case": "", "risk_factors": []}

    # Normalize: strip markdown bold, handle **HEADER:** patterns
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', raw)

    m = re.search(r'VERDICT:\s*(STRONG\s*BUY|BUY|HOLD|SELL|STRONG\s*SELL)', cleaned, re.IGNORECASE)
    if m:
        result["verdict"] = m.group(1).upper().replace(' ', ' ').strip()

    for label, key in [("INVESTMENT_THESIS", "investment_thesis"), ("INVESTMENT THESIS", "investment_thesis")]:
        m = re.search(
            rf'{label}:\s*\n?(.*?)(?=\n\w+(?:_CASE|\s*CASE|\s*THESIS):|\nVERDICT:|\nRISK\s*FACTORS:|\Z)',
            cleaned, re.DOTALL | re.IGNORECASE,
        )
        if m and m.group(1).strip():
            result["investment_thesis"] = m.group(1).strip()[:1000]
            break

    for label in ("BULL_CASE", "BULL CASE"):
        m = re.search(rf'{label}:\s*\n?(.*?)(?=\n\w+(?:_CASE|\s*CASE):|\nRISK\s*FACTORS:|\Z)', cleaned, re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip():
            result["bull_case"] = m.group(1).strip()[:600]
            break

    for label in ("BEAR_CASE", "BEAR CASE"):
        m = re.search(rf'{label}:\s*\n?(.*?)(?=\n\w+(?:_CASE|\s*CASE):|\nRISK\s*FACTORS:|\Z)', cleaned, re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip():
            result["bear_case"] = m.group(1).strip()[:600]
            break

    for label in ("RISK_FACTORS", "RISK FACTORS"):
        m = re.search(rf'{label}:\s*\n(.*?)(?=\n\w+(?:_CASE|\s*CASE|\s*THESIS):|\Z)', cleaned, re.DOTALL | re.IGNORECASE)
        if m:
            pts = [re.sub(r'^[-•*\d.\s]+', '', p).strip() for p in m.group(1).strip().split('\n') if p.strip() and len(p.strip()) > 5]
            result["risk_factors"] = pts[:5]
            break

    return result


# --- Main Debate Orchestrator ---

class StrategyDecider:
    """Runs 7-agent PARA-DEBATE on top-ranked tickers to produce trade decisions."""

    def __init__(self):
        self._cache: dict[str, DebateResult] = {}

    def debate(self, ticker: str, market: str, context: dict) -> DebateResult:
        """Run full 7-agent debate synchronously. For async use, call debate_batch."""
        start = time.perf_counter()

        # Check cache
        cache_key = f"{ticker}:{context.get('composite_score', 0):.3f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            ctx = {"ticker": ticker, "market": market, **context}

            # Phase 1: Run 5 specialists (parallel with threads)
            specialists = [
                ("MACRO", MACRO_PROMPT),
                ("FUNDAMENTAL", FUNDAMENTAL_PROMPT),
                ("TECHNICAL", TECHNICAL_PROMPT),
                ("SENTIMENT", SENTIMENT_PROMPT),
                ("GEOPOLITICAL", GEOPOLITICAL_PROMPT),
            ]

            import concurrent.futures
            specialist_outputs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(_run_specialist, name, prompt, ctx): name
                    for name, prompt in specialists
                }
                for f in concurrent.futures.as_completed(futures):
                    specialist_outputs.append(f.result())

            # Sort to consistent order
            specialist_outputs.sort(key=lambda o: o.agent)

            # Phase 2: Adversarial
            adversarial = _run_adversarial(specialist_outputs, ctx)

            # Phase 3: Consensus
            all_outputs = specialist_outputs + [adversarial]
            consensus = _compute_consensus(all_outputs)

            # Phase 4: Head Analyst
            head = _run_head_analyst(all_outputs, consensus, ctx)

            latency = (time.perf_counter() - start) * 1000

            result = DebateResult(
                ticker=ticker,
                market=market,
                verdict=head["verdict"],
                consensus_score=consensus,
                investment_thesis=head["investment_thesis"],
                bull_case=head["bull_case"],
                bear_case=head["bear_case"],
                risk_factors=head["risk_factors"],
                agent_outputs=all_outputs,
                debate_json=json.dumps({
                    "agent_outputs": [
                        {
                            "agent": o.agent, "stance": o.stance,
                            "conviction": o.conviction, "thesis": o.thesis,
                            "key_points": o.key_points,
                        }
                        for o in all_outputs
                    ],
                    "consensus": consensus,
                    "head_analyst": head,
                }),
                latency_ms=latency,
            )

            self._cache[cache_key] = result
            logger.info(f"Debate {ticker}: {result.verdict} (consensus={consensus:.2f}, {latency:.0f}ms)")
            return result

        except Exception as e:
            logger.exception(f"Debate failed for {ticker}")
            return DebateResult(
                ticker=ticker, market=market,
                verdict="HOLD", consensus_score=0.0,
                investment_thesis="", bull_case="", bear_case="",
                risk_factors=[str(e)], agent_outputs=[],
                error=str(e),
            )

    def debate_batch(self, tickers: list[str], market: str, contexts: dict[str, dict]) -> list[DebateResult]:
        """Run debates for multiple tickers (sequential to avoid rate limiting)."""
        results = []
        for ticker in tickers:
            ctx = contexts.get(ticker, {})
            result = self.debate(ticker, market, ctx)
            results.append(result)
            time.sleep(0.5)  # Rate limit buffer
        return results

    def verdict_to_action(self, verdict: str) -> str:
        """Map Head Analyst verdict to trade action."""
        if verdict in ("STRONG BUY", "BUY"):
            return "BUY"
        if verdict in ("STRONG SELL", "SELL"):
            return "SELL"
        return "HOLD"
