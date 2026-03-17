"""
Polymarket Trading Bot v2 — Main Orchestrator
==============================================
Pipeline: scan → ELO features → sentiment → edge detect → execute → manage positions.
PAPER_TRADING=true is the hardcoded default fallback.
Real trading requires PAPER_TRADING=false + explicit CONFIRM prompt at startup.
Tests must pass before the bot starts.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path("/etc/secrets/.env"))          # Render secret file location
load_dotenv(Path(__file__).parent / ".env")    # local fallback

# ── Safety check: PAPER_TRADING default ───────────────────────────────────────
# Hardcoded fallback regardless of env
_ENV_PAPER = os.getenv("PAPER_TRADING", "true").strip().lower()
PAPER_TRADING: bool = _ENV_PAPER != "false"  # anything other than explicit "false" = paper


def _confirm_live_trading():
    """Require typed CONFIRM before enabling live trading (interactive only)."""
    if PAPER_TRADING:
        return
    print("\n" + "=" * 60)
    print("WARNING: Real wallet trading is ENABLED.")
    print("   PAPER_TRADING=false in your .env file.")
    print("   Real USDC will be used for orders.")
    print("=" * 60)
    if not sys.stdin.isatty():
        # Non-interactive environment (e.g. Render, Docker). Require explicit
        # opt-in via LIVE_TRADING_CONFIRMED=true env var instead of stdin.
        if os.getenv("LIVE_TRADING_CONFIRMED", "").strip().lower() != "true":
            print("Non-interactive environment detected and LIVE_TRADING_CONFIRMED != true.")
            print("Aborting. Set PAPER_TRADING=true or LIVE_TRADING_CONFIRMED=true to proceed.")
            sys.exit(1)
        print("Live trading confirmed via LIVE_TRADING_CONFIRMED env var.\n")
        return
    answer = input("Type CONFIRM to proceed with live trading, or anything else to abort: ").strip()
    if answer != "CONFIRM":
        print("Aborting. Set PAPER_TRADING=true to run safely.")
        sys.exit(0)
    print("Live trading confirmed.\n")


def _run_tests() -> bool:
    """Run pytest. Bot refuses to start if tests fail."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("TESTS FAILED — bot will not start:")
        print(result.stderr)
        return False
    return True


def _scan_for_hardcoded_secrets():
    """Scan all .py files for potential secrets."""
    import re
    secret_pattern = re.compile(
        r'(api[_-]?key|secret|passphrase|private[_-]?key|password)\s*=\s*["\'][^"\']{6,}["\']',
        re.IGNORECASE,
    )
    issues = []
    for py_file in Path(__file__).parent.rglob("*.py"):
        if ".pytest_cache" in str(py_file):
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for match in secret_pattern.finditer(content):
                # Ignore placeholder/env/test patterns
                val = match.group(0)
                if any(x in val.lower() for x in ["your_", "placeholder", "os.getenv", "os.environ", "env.", "test_", "example"]):
                    continue
                issues.append(f"{py_file.name}: {val[:60]}")
        except Exception:
            pass
    if issues:
        print("POTENTIAL HARDCODED SECRETS FOUND:")
        for i in issues:
            print(f"  {i}")
    return len(issues) == 0


def _seed_paper_trades(markets, trade_logger, n: int, logger):
    """Place n random paper trades across top markets to seed the system."""
    import random
    eligible = [m for m in markets if 0.02 < m.yes_price < 0.98 and m.liquidity > 50]
    selected = random.sample(eligible, min(n, len(eligible)))
    count = 0
    for market in selected:
        side = random.choice(["YES", "NO"])
        price = market.yes_price if side == "YES" else market.no_price
        if price <= 0 or price >= 1:
            continue
        trade_logger.log_trade(
            market_id=market.market_id,
            market_question=market.question,
            category=market.category,
            side=side,
            entry_price=price,
            size=10.0,
            fair_value=0.5,
            edge_at_entry=0.0,
            elo_diff=0.0,
            sentiment_score=0.0,
            is_paper=True,
            status="open",
            slippage=0.0,
        )
        count += 1
    logger.info(f"Seeded {count} random paper trades across {len(selected)} markets.")
    return count


def _ingest_historical_snapshots(snapshot_store, logger):
    """Paginate through ALL Polymarket markets and store snapshots."""
    import requests as req
    from scanner.market_scanner import Market

    session = req.Session()
    session.headers.update({"User-Agent": "polymarket-bot/2.0", "Accept": "application/json"})

    total = 0
    offset = 0
    limit = 25000
    empty_pages = 0

    logger.info("Starting full historical market ingestion (all pages)...")
    while True:
        try:
            resp = session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"limit": limit, "offset": offset,
                        "order": "volume24hr", "ascending": "false"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            markets_data = data if isinstance(data, list) else data.get("markets", [])

            if not markets_data:
                empty_pages += 1
                if empty_pages >= 2:
                    break
                offset += limit
                continue

            batch = []
            for item in markets_data:
                try:
                    m = Market(item)
                    batch.append(m)
                except Exception:
                    pass

            if batch:
                snapshot_store.save_snapshots(batch)
                total += len(batch)
                logger.info(f"Ingested {total} snapshots so far (offset={offset})...")

            if len(markets_data) < limit:
                break  # last page

            offset += limit
            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"Ingestion error at offset {offset} with limit {limit}: {e} — retrying with limit=1000")
            limit = 1000
            time.sleep(1.0)
            continue

    logger.info(f"Historical ingestion complete — {total} total market snapshots stored.")


def main():
    _requested = os.getenv("DATA_DIR", "data")
    try:
        os.makedirs(f"{_requested}/logs", exist_ok=True)
        data_dir = _requested
    except PermissionError:
        data_dir = "data"
        print(f"WARNING: DATA_DIR={_requested!r} is not writable — falling back to local 'data/' directory.")
        os.makedirs(f"{data_dir}/logs", exist_ok=True)

    # ── Logging ────────────────────────────────────────────────────────────────
    from logger.log_setup import setup_logging
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    import logging
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("Polymarket Bot v2 starting...")
    logger.info(f"Mode: {'PAPER TRADING' if PAPER_TRADING else '[!!] LIVE TRADING'}")
    logger.info("=" * 60)

    # ── Safety self-review ─────────────────────────────────────────────────────
    clean = _scan_for_hardcoded_secrets()
    if not clean:
        logger.warning("Hardcoded secret scan flagged issues — review above before going live.")

    # ── Confirmation for live mode ─────────────────────────────────────────────
    _confirm_live_trading()

    # ── Run tests before starting ──────────────────────────────────────────────
    if os.getenv("SKIP_STARTUP_TESTS", "false").strip().lower() != "true":
        logger.info("Running test suite...")
        if not _run_tests():
            logger.error("Tests failed — fix issues before running the bot.")
            sys.exit(1)
        logger.info("All tests passed.")
    else:
        logger.info("SKIP_STARTUP_TESTS=true — skipping test suite.")

    # ── Config ─────────────────────────────────────────────────────────────────
    min_edge = float(os.getenv("MIN_EDGE_PCT", os.getenv("MIN_EDGE_THRESHOLD", "0.08")))
    max_pos_size = float(os.getenv("MAX_POSITION_SIZE", "100"))
    max_pos_pct = float(os.getenv("MAX_POSITION_PCT", "0.05"))
    profit_target = float(os.getenv("PROFIT_TARGET", "0.15"))
    stop_loss_pct = float(os.getenv("STOP_LOSS", "0.10"))
    max_daily_drawdown = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.05"))
    bankroll = float(os.getenv("BANKROLL", "1000"))
    webhook_url = os.getenv("WEBHOOK_URL", "")
    dashboard_port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "8000")))
    news_api_key = os.getenv("NEWS_API_KEY", "")
    poll_interval = int(os.getenv("POLL_INTERVAL", "7"))

    # ── Check WALLET_PRIVATE_KEY → force paper if missing ─────────────────────
    wallet_pk = os.getenv("WALLET_PRIVATE_KEY", "")
    effective_paper = PAPER_TRADING or not wallet_pk
    if not wallet_pk and not PAPER_TRADING:
        logger.warning("WALLET_PRIVATE_KEY missing — forcing PAPER_TRADING=true")

    # ── Initialize modules ─────────────────────────────────────────────────────
    from logger.trade_logger import TradeLogger
    from scanner.market_scanner import MarketScanner, SnapshotStore
    from model.elo import ELOSystem
    from model.fair_value_model import FairValueModel
    from model.edge_detector import EdgeDetector
    from model.sentiment import SentimentAnalyzer
    from model.momentum import MomentumScorer
    from executor.trade_executor import TradeExecutor
    from position_manager.position_manager import PositionManager
    from dashboard.cli_dashboard import CLIDashboard
    from dashboard.web_dashboard import create_app, update_signals, update_status
    from logger.log_setup import log_trade_event, log_edge, log_model

    trade_logger = TradeLogger(db_path=f"{data_dir}/trades.db")
    snapshot_store = SnapshotStore(db_path=f"{data_dir}/snapshots.db")
    elo_system = ELOSystem(save_path=Path(data_dir) / "elo_state.json")
    model = FairValueModel(elo_system=elo_system)
    edge_detector = EdgeDetector(
        model=model,
        elo_system=elo_system,
        min_edge=min_edge,
    )
    sentiment = SentimentAnalyzer(api_key=news_api_key)
    momentum = MomentumScorer(elo_system=elo_system)
    scanner = MarketScanner(
        api_key=os.getenv("POLYMARKET_API_KEY", ""),
        poll_interval=poll_interval,
        snapshot_store=snapshot_store,
    )
    executor = TradeExecutor(
        trade_logger=trade_logger,
        paper_trading=effective_paper,
        max_position_size=max_pos_size,
        max_position_pct=max_pos_pct,
        bankroll=bankroll,
        api_key=os.getenv("POLYMARKET_API_KEY", ""),
        api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
        api_passphrase=os.getenv("POLYMARKET_PASSPHRASE", ""),
        wallet_private_key=wallet_pk,
    )
    pos_manager = PositionManager(
        trade_logger=trade_logger,
        profit_target=profit_target,
        stop_loss=stop_loss_pct,
        max_daily_drawdown_pct=max_daily_drawdown,
        elo_system=elo_system,
        paper_trading=effective_paper,
        bankroll=bankroll,
    )
    dashboard = CLIDashboard(
        trade_logger=trade_logger,
        paper_trading=effective_paper,
        elo_system=elo_system,
        bankroll=bankroll,
    )

    # ── Train/load model ───────────────────────────────────────────────────────
    logger.info("Preparing fair value model...")
    model.ensure_ready(trade_logger=trade_logger, snapshot_store=snapshot_store)
    dashboard.set_model_info(model._last_mae, model._last_n_samples)
    log_model("Model ready", mae=model._last_mae, n=model._last_n_samples)

    # ── Ingest historical snapshots on startup ─────────────────────────────────
    logger.info("Ingesting historical market snapshots...")
    _ingest_historical_snapshots(snapshot_store, logger)

    # ── Seed random paper trades if requested ──────────────────────────────────
    seed_n = int(os.getenv("SEED_TRADES", "0"))
    if seed_n > 0 and trade_logger.get_stats(paper=True)["total_trades"] == 0:
        logger.info(f"Seeding {seed_n} random paper trades...")
        seed_markets = scanner.fetch_markets(limit=200)
        if seed_markets:
            _seed_paper_trades(seed_markets, trade_logger, seed_n, logger)

    # ── Web dashboard thread ───────────────────────────────────────────────────
    web_app = create_app(trade_logger, elo_system=elo_system, paper_trading=effective_paper)

    def _run_web():
        import uvicorn
        uvicorn.run(web_app, host="0.0.0.0", port=dashboard_port, log_level="warning")

    threading.Thread(target=_run_web, daemon=True, name="web").start()
    logger.info(f"Web dashboard: http://localhost:{dashboard_port}")

    # ── Position manager thread ────────────────────────────────────────────────
    _stop = threading.Event()

    def _pos_loop():
        while not _stop.is_set():
            try:
                pos_manager.run_cycle()
            except Exception as e:
                logger.error(f"Position manager error: {e}")
            time.sleep(30)

    threading.Thread(target=_pos_loop, daemon=True, name="positions").start()

    # ── Auto-retrain thread ────────────────────────────────────────────────────
    def _retrain_loop():
        while not _stop.is_set():
            time.sleep(3600)  # check every hour
            if model.should_retrain():
                logger.info("Triggering scheduled model retraining...")
                try:
                    mae = model.retrain_from_resolved(snapshot_store, trade_logger)
                    dashboard.set_model_info(model._last_mae, model._last_n_samples)
                    log_model("Scheduled retrain complete", mae=mae)
                except Exception as e:
                    logger.error(f"Auto-retrain error: {e}")

    threading.Thread(target=_retrain_loop, daemon=True, name="retrain").start()

    # ── Daily paper report thread ──────────────────────────────────────────────
    def _report_loop():
        import datetime as dt
        while not _stop.is_set():
            now = dt.datetime.utcnow()
            if now.hour == 0 and now.minute < 2:
                report = trade_logger.daily_paper_report()
                logger.info(f"DAILY PAPER REPORT: {report}")
            time.sleep(60)

    threading.Thread(target=_report_loop, daemon=True, name="daily-report").start()

    # ── Build summary ──────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("BUILD SUMMARY")
    logger.info("  Modules: scanner, ELO, fair_value_model, edge_detector,")
    logger.info("           sentiment, executor, position_manager, dashboard")
    logger.info(f"  Model features: 20 | MAE: {model._last_mae:.4f}")
    logger.info(f"  Paper trading: {effective_paper}")
    logger.info(f"  Min edge: {min_edge:.0%} | Max position: ${max_pos_size}")
    logger.info("─" * 60)

    # ── Main trading loop ──────────────────────────────────────────────────────
    cycle = 0
    logger.info("Main trading loop started. Press Ctrl+C to stop.")

    try:
        while True:
            cycle += 1
            dashboard.set_cycle(cycle)

            status = f"Cycle #{cycle} | scanning {len(scanner.get_cached_markets())} cached markets..."
            dashboard.set_status(status)
            update_status({
                "cycle": cycle,
                "paper": effective_paper,
                "status": status,
                "snapshot_count": snapshot_store.get_snapshot_count(),
            })
            dashboard.set_snapshot_count(snapshot_store.get_snapshot_count())

            # 1. Scan markets
            markets = scanner.scan_once()
            if not markets:
                logger.warning("No markets returned. Retrying in 15s...")
                time.sleep(15)
                continue

            # 2. Sentiment (with score + momentum)
            try:
                sentiment_data = sentiment.batch_analyze(markets[:40])
            except Exception as e:
                logger.warning(f"Sentiment analysis error: {e}")
                sentiment_data = {}

            # 3. Detect edges (ELO primary signal)
            signals = edge_detector.scan_markets(markets, sentiment_data=sentiment_data)
            dashboard.set_signals(signals)
            update_signals(signals)

            if signals:
                for s in signals[:3]:
                    log_edge(
                        "Edge detected",
                        market_id=s.market.market_id[:8],
                        side=s.trade_side,
                        elo_edge=f"{s.elo_edge:.1%}",
                        confidence=f"{s.confidence:.0%}",
                    )

            # 4. Check daily drawdown before executing
            if trade_logger.is_trading_halted_today(paper=effective_paper):
                logger.warning("Trading halted for today (daily drawdown limit). Monitoring only.")
                dashboard.set_status(f"Cycle #{cycle} | TRADING HALTED (daily drawdown)")
            else:
                # Execute top signals (skip already-open markets)
                open_ids = {t.market_id for t in trade_logger.get_open_trades(paper=effective_paper)}
                executed = 0
                for signal in signals[:3]:
                    if signal.market.market_id in open_ids:
                        continue
                    trade = executor.execute(signal, webhook_url=webhook_url)
                    if trade:
                        open_ids.add(signal.market.market_id)
                        executed += 1
                        log_trade_event(
                            "Trade opened",
                            id=trade.id,
                            side=trade.side,
                            price=trade.entry_price,
                            size=trade.size,
                            elo_diff=trade.elo_diff,
                        )
                if executed:
                    logger.info(f"Executed {executed} trade(s) this cycle")

            # 5. CLI dashboard (TTY only — suppressed on Render/Docker)
            if sys.stdout.isatty():
                dashboard.render_once()

            import random
            time.sleep(poll_interval + random.uniform(-1.5, 1.5))

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down gracefully...")
        _stop.set()

        # Save final state
        elo_system.save()
        logger.info("ELO state saved.")

        # Print final stats
        stats = trade_logger.get_stats(paper=effective_paper)
        logger.info(
            f"Final stats: {stats['total_trades']} trades | "
            f"Win rate: {stats['win_rate']:.1f}% | "
            f"Total P&L: ${stats['total_pnl']:.2f}"
        )
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
