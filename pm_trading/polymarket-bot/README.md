# Polymarket Trading Bot v2

An autonomous Python trading bot for [Polymarket](https://polymarket.com) prediction markets.
Scans markets every ~7 seconds, estimates fair value using an ELO-powered XGBoost model,
detects mispriced edges, and executes paper or live trades.

**Default mode: PAPER TRADING** — your real wallet is never touched until you explicitly
set `PAPER_TRADING=false` and type `CONFIRM` at the startup prompt.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        main.py                                 │
│  Orchestrates all threads + main trading loop                  │
└───────────────────────────┬────────────────────────────────────┘
                            │
       ┌────────────────────┼──────────────────────┐
       │                    │                      │
       ▼                    ▼                      ▼
┌─────────────┐    ┌────────────────┐    ┌─────────────────────┐
│   Scanner   │    │   ELO System   │    │  Position Manager   │
│  (Gamma API)│    │  model/elo.py  │    │  (30s cycle thread) │
│  7s polling │    │  3 dimensions: │    │  trailing stop,     │
│  snapshots  │    │  category/     │    │  profit target,     │
│  → SQLite   │    │  event_type/   │    │  daily drawdown     │
└──────┬──────┘    │  time_of_day   │    └─────────────────────┘
       │           └────────┬───────┘
       ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feature Engineering                        │
│  20 features including ELO-implied probability,             │
│  ELO edge (PRIMARY signal), rolling volume windows,         │
│  sentiment score/momentum, whale activity flag              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              XGBoost Fair Value Model                        │
│  Predicts true YES probability                               │
│  Auto-retrains every 24h on resolved markets                 │
│  Logs feature importance after each retrain                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Edge Detector                              │
│  Gate 1: ELO edge > MIN_EDGE_PCT (8%)                        │
│  Gate 2: Sentiment direction agrees                          │
│  Gate 3: Model does not strongly contradict ELO             │
│  Output: EdgeSignal ranked by CONFIDENCE, not just size      │
│  Kelly Criterion position sizing                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                Trade Executor                                │
│  Paper: simulates slippage 0.1–0.3%, order book impact      │
│  Live: py-clob-client CLOB orders (requires CONFIRM prompt)  │
│  PAPER_TRADING guard before every real order                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  Trade Logger (SQLite)                        │
│  Separate tables: paper vs live (NEVER mixed)                │
│  Records: entry/exit, P&L, hold time, ELO diff, sentiment,  │
│  exit reason, Sharpe, Sortino, daily P&L tracking            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                 ┌─────────┴──────────┐
                 ▼                    ▼
         ┌──────────────┐    ┌──────────────────┐
         │ CLI Dashboard│    │  Web Dashboard   │
         │   (Rich)     │    │  FastAPI :8000   │
         │  5s refresh  │    │  5s auto-refresh │
         └──────────────┘    └──────────────────┘
```

---

## The ELO System (Primary Edge Signal)

**This is the core of the strategy**, inspired by tennis prediction research where
ELO momentum outperformed raw win rates as the most predictive feature.

### How it works

Three separate ELO rating dimensions, each starting at 1500:

| Dimension | Keys | Weight in blended probability |
|-----------|------|-------------------------------|
| `category` | politics, crypto, sports, economics… | 60% |
| `event_type` | election, fed_decision, earnings… | 30% |
| `time_of_day` | morning, afternoon, evening, overnight | 10% |

**ELO updates after every market resolution:**
- YES resolution → ELO goes **up**
- NO resolution → ELO goes **down**
- Volume-weighted: $1M market swing = 3× the ELO delta of a $1k market

**Converting ELO to probability:**
```
P(YES) = 1 / (1 + 10^((1500 - rating) / 400))
```

**The primary edge signal:**
```
elo_edge = elo_implied_probability - current_market_price
```

If `elo_edge > 8%` (configurable), the market is considered mispriced and a trade is flagged.

### Why ELO > raw stats

Raw yes/no rates are noisy and don't adjust for recency or volume.
The ELO system naturally weights recent high-volume resolutions more,
giving the model a "momentum signal" that is forward-looking rather than
just a historical average.

---

## XGBoost Model

The model predicts the true probability of YES resolution using **20 engineered features**:

| # | Feature | Why it matters |
|---|---------|----------------|
| 1 | `market_age_hours` | Older markets are better calibrated |
| 2 | `time_to_resolution_hours` | More time = more uncertainty |
| 3 | `current_price` | Raw market signal |
| 4 | `volume_1h` | Short-term interest |
| 5 | `volume_6h` | Medium-term trend |
| 6 | `volume_24h` | Main volume metric |
| 7 | `volume_7d` | Weekly context |
| 8 | `volume_acceleration` | d(volume)/dt — momentum |
| 9 | `price_velocity` | d(price)/dt — directional momentum |
| 10 | `category_elo` | ELO of the market's topic |
| 11 | `event_type_elo` | ELO of the event type |
| 12 | `elo_implied_probability` | Blended ELO → probability |
| **13** | **`elo_edge`** | **THE primary signal: elo_prob - price** |
| 14 | `liquidity_depth` | Thin markets = harder to trade |
| 15 | `bid_ask_spread` | Tight spread = efficient = harder edge |
| 16 | `age_ratio` | age / (age + time_to_resolution) |
| 17 | `sentiment_score` | NewsAPI headline sentiment |
| 18 | `sentiment_momentum` | Direction of sentiment change |
| 19 | `hour_of_day` | Temporal pattern |
| 20 | `whale_activity_flag` | Large order detected in last hour |

**Auto-retraining:** Every 24 hours, the model retrains using:
1. All resolved markets stored in `data/snapshots.db`
2. Synthetic data to pad the training set early on

Feature importance is logged after every retraining event to `data/logs/model.log`.

---

## Quickstart

### 1. Clone and install
```bash
cd polymarket-bot
pip install -r requirements.txt
```

### 2. Add API keys (optional for paper trading)
```bash
cp .env.example .env
# Edit .env — all fields are optional for paper trading
```

### 3. Run in paper mode (safe — default)
```bash
python main.py
```

The bot will:
1. Run the test suite (91 tests) — **refuses to start if any fail**
2. Scan for hardcoded secrets
3. Train the XGBoost model on synthetic data (~5 seconds)
4. Start the web dashboard at `http://localhost:8000`
5. Begin scanning Polymarket every ~7 seconds
6. Display the Rich terminal dashboard

---

## How to Add API Keys

Edit `.env`:

```env
# For paper trading — only NEWS_API_KEY matters (optional)
NEWS_API_KEY=your_newsapi_key_from_newsapi.org

# For LIVE trading — fill all four:
POLYMARKET_API_KEY=your_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_PASSPHRASE=your_passphrase
WALLET_PRIVATE_KEY=your_polygon_wallet_private_key
PAPER_TRADING=false
```

Get Polymarket CLOB credentials from your account at polymarket.com.
Get a free NewsAPI key at newsapi.org (optional — improves sentiment).

---

## Enabling Live Trading

> **Warning**: Start with very small `MAX_POSITION_SIZE` (e.g., `5` USDC) to verify
> everything works before scaling up.

1. Fill in all four wallet fields in `.env`
2. Set `PAPER_TRADING=false`
3. Run `python main.py`
4. Type `CONFIRM` at the prompt

The bot implements multiple safety layers:
- Missing `WALLET_PRIVATE_KEY` → forces paper trading regardless of flag
- `PAPER_TRADING` guard asserted before every real CLOB order
- Daily drawdown halt: stops trading if today's P&L drops below `MAX_DAILY_DRAWDOWN_PCT`
- Maximum position: `min(kelly_size, MAX_POSITION_PCT * bankroll, MAX_POSITION_SIZE)`

---

## How to Backtest

The bot accumulates real market snapshots in `data/snapshots.db` as it runs.
Once you have resolved markets stored:

```python
from scanner.market_scanner import SnapshotStore
from model.fair_value_model import FairValueModel
from model.elo import ELOSystem

store = SnapshotStore("data/snapshots.db")
elo = ELOSystem()
model = FairValueModel(elo_system=elo)

# Retrain on real data
model.retrain_from_resolved(store)

# Inspect resolved markets
resolved = store.get_resolved_for_training()
print(f"{len(resolved)} resolved markets available")
```

For deeper backtesting, export the snapshots table to a DataFrame and replay
the edge detector logic with different `min_edge` thresholds.

---

## Dashboard Guide

### Terminal Dashboard (CLI)
Refreshes every 5 seconds. Columns:

| Section | What to look at |
|---------|-----------------|
| **Performance** | Win Rate (50) = last 50 trades — more reliable than all-time |
| **Top Category ELOs** | Categories above 1500 = historically YES-biased |
| **Top Edges** | `ELO d` = raw ELO signal; `Edge` = blended with model; `Conf` = combined score |
| **Open Positions** | `ELO d` column shows the edge that triggered the trade |

### Web Dashboard (`http://localhost:8000`)
- **Edges tab**: All current signals with Kelly fraction
- **Positions tab**: Open trades
- **Trades tab**: Paginated history with hold time and exit reason
- **ELO tab**: Full ELO rating table, all categories

---

## Performance Expectations & Limitations

**Realistic expectations:**
- This bot operates on a semi-efficient market. Detectable edges are rare (typically 2–8 per scan).
- Most edges come from newly opened markets before liquidity stabilizes.
- ELO cold start: ratings start at 1500 (neutral). The signal gets stronger after ~50+ resolved markets.
- Slippage (0.1–0.3%) and CLOB fees eat into edge. Only trade edges > 8% to stay profitable.

**Limitations:**
- The XGBoost model is initially trained on synthetic data. Accuracy improves as real resolved markets accumulate in `data/snapshots.db`.
- Sentiment analysis without a NewsAPI key relies on question-text lexicon only.
- Polymarket restricts access from certain jurisdictions. Check Terms of Service before use.
- This is not financial advice. Past performance of prediction market bots does not guarantee future results.

---

## Running Tests

```bash
pytest tests/ -v
```

91 tests covering: ELO calculator (19), trade logger (13), market scanner (12),
fair value model (10), edge detector (5), Kelly criterion (5), sentiment (9),
position manager (9), executor (8), and 2 integration tests.

The bot **refuses to start** if any test fails.

---

## REST API Reference

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web dashboard |
| `GET /health` | Heartbeat, uptime, paper/live mode |
| `GET /stats` | Win rate, P&L, Sharpe, Sortino, model MAE |
| `GET /positions` | Open positions |
| `GET /trades?limit=N&offset=N` | Paginated trade history |
| `GET /edges` | Current top edges with all signal components |
| `GET /elo` | Full ELO table (category, event_type, time_bucket) |

---

## .env Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPER_TRADING` | `true` | **Master safety switch.** Set `false` for live trading |
| `WALLET_PRIVATE_KEY` | `` | Polygon wallet private key — missing = forced paper mode |
| `POLYMARKET_API_KEY` | `` | CLOB API key (live only) |
| `POLYMARKET_API_SECRET` | `` | CLOB API secret (live only) |
| `POLYMARKET_PASSPHRASE` | `` | CLOB passphrase (live only) |
| `NEWS_API_KEY` | `` | NewsAPI key for headline sentiment (optional) |
| `MIN_EDGE_PCT` | `0.08` | Minimum ELO edge to flag a trade (8%) |
| `MAX_POSITION_SIZE` | `100` | Max USDC per trade (absolute cap) |
| `MAX_POSITION_PCT` | `0.05` | Max % of bankroll per trade (Kelly cap) |
| `BANKROLL` | `1000` | Your starting bankroll for Kelly sizing |
| `PROFIT_TARGET` | `0.15` | Exit when position gains 15% |
| `STOP_LOSS` | `0.10` | Trailing stop: exit when down 10% from peak |
| `MAX_DAILY_DRAWDOWN_PCT` | `0.05` | Halt trading if today's P&L < -5% of bankroll |
| `WEBHOOK_URL` | `` | Slack/Discord webhook for large edge alerts |
| `DASHBOARD_PORT` | `8000` | Web dashboard port |
| `POLL_INTERVAL` | `7` | Seconds between market scans |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG/INFO/WARNING/ERROR) |

---

## Log Files

All logs in `data/logs/`, rotated daily, 30-day retention:

| File | Contents |
|------|----------|
| `bot.log` | All bot activity |
| `errors.log` | Errors only |
| `trades.log` | Trade open/close events |
| `edges.log` | Every detected edge signal |
| `model.log` | Model training events + feature importance |

---

## Project Structure

```
polymarket-bot/
├── .env                          Your config (never commit)
├── .env.example                  Template
├── requirements.txt
├── main.py                       Orchestrator
├── secret_scan.py                Runs at startup — checks for hardcoded secrets
│
├── scanner/
│   └── market_scanner.py         Gamma API polling + SnapshotStore (SQLite)
│
├── model/
│   ├── elo.py                    ELO system (PRIMARY signal) — 3 dimensions
│   ├── fair_value_model.py       XGBoost 20-feature model + auto-retrain
│   ├── edge_detector.py          3-gate signal detection + Kelly sizing
│   ├── sentiment.py              NewsAPI + lexicon sentiment + momentum
│   └── momentum.py               Legacy wrapper (delegates to ELOSystem)
│
├── executor/
│   └── trade_executor.py         Paper (slippage sim) + live CLOB orders
│
├── position_manager/
│   └── position_manager.py       Trailing stop, profit target, drawdown halt
│
├── logger/
│   ├── trade_logger.py           SQLAlchemy trades DB (paper/live separate)
│   └── log_setup.py              Rotating file handlers, 4 log channels
│
├── dashboard/
│   ├── cli_dashboard.py          Rich terminal dashboard
│   └── web_dashboard.py          FastAPI web UI
│
├── tests/                        91 tests — bot refuses to start if any fail
│   ├── test_elo.py               19 ELO system tests
│   ├── test_logger.py            13 trade logger tests
│   ├── test_scanner.py           12 scanner + snapshot tests
│   ├── test_model.py             24 model + edge + sentiment + Kelly tests
│   ├── test_executor.py          8 executor + slippage tests
│   ├── test_position_manager.py  9 position management tests
│   └── test_integration.py       2 full end-to-end cycle tests
│
└── data/                         Auto-created at runtime
    ├── trades.db                 SQLite trade log
    ├── snapshots.db              Market price history (grows over time)
    ├── elo_state.json            Persisted ELO ratings
    ├── fair_value_model.pkl      Trained XGBoost model
    └── logs/                     Rotating daily log files
```
