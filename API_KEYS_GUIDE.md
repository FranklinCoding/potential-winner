# API Keys Guide

## Paper Trading (no real money)

You need zero API keys. The bot already works right now — it pulls market data from
Polymarket's public API and simulates everything. The only optional one:

| Key | Where to get it | Why |
|-----|----------------|-----|
| `NEWS_API_KEY` | newsapi.org → Sign Up (free) | Better sentiment signals. Without it the bot uses keyword matching on market titles instead |

---

## Live Trading ($100 real account)

You need 4 things from two places:

### 1. Polymarket CLOB API (3 keys)
Go to polymarket.com → connect a wallet → go to your profile → API Keys section:
- `POLYMARKET_API_KEY`
- `POLYMARKET_API_SECRET`
- `POLYMARKET_PASSPHRASE`

### 2. Polygon Wallet Private Key (1 key)
- `WALLET_PRIVATE_KEY` — the private key of the wallet you connect to Polymarket
- You fund this wallet with USDC on Polygon (not ETH mainnet)
- To get USDC on Polygon: buy USDC on Coinbase/Kraken → withdraw to Polygon network to your wallet address

---

## Important notes for the $100 live account

Once you have the keys, set in .env:

    PAPER_TRADING=false
    MAX_POSITION_SIZE=5        <- start small, e.g. $5 max per trade
    MAX_POSITION_PCT=0.03      <- 3% of bankroll per trade
    BANKROLL=100
    WALLET_PRIVATE_KEY=your_key
    POLYMARKET_API_KEY=your_key
    POLYMARKET_API_SECRET=your_secret
    POLYMARKET_PASSPHRASE=your_passphrase

The bot will prompt you to type CONFIRM before any real money moves.
Recommended: run paper mode for a few days first to see what signals it finds
before switching to live.
