"""
FastAPI web dashboard.
Endpoints:
  GET /          - HTML dashboard
  GET /stats     - performance metrics
  GET /positions - open positions
  GET /trades    - paginated trade history
  GET /edges     - current top edges
  GET /elo       - ELO table by category
  GET /health    - heartbeat + uptime
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

_trade_logger = None
_elo_system = None
_latest_signals: list = []
_bot_status: dict = {}
_start_time: float = time.time()
_paper_trading: bool = True


def create_app(trade_logger, elo_system=None, paper_trading: bool = True) -> FastAPI:
    global _trade_logger, _elo_system, _paper_trading
    _trade_logger = trade_logger
    _elo_system = elo_system
    _paper_trading = paper_trading

    app = FastAPI(title="Polymarket Bot", version="2.0")

    @app.get("/health")
    def health():
        uptime = time.time() - _start_time
        return {
            "status": "running",
            "uptime_seconds": round(uptime),
            "uptime_human": _fmt_uptime(uptime),
            "paper_trading": _paper_trading,
            "timestamp": datetime.utcnow().isoformat(),
            **_bot_status,
        }

    @app.get("/stats")
    def stats():
        if not _trade_logger:
            return JSONResponse({"error": "not initialized"}, status_code=503)
        s = _trade_logger.get_stats(paper=_paper_trading)
        s50 = _trade_logger.get_stats(paper=_paper_trading, last_n=50)
        halted = _trade_logger.is_trading_halted_today(paper=_paper_trading)
        today_pnl = _trade_logger.get_today_pnl(paper=_paper_trading)
        retrain = _trade_logger.get_last_retrain()
        return {
            **s,
            "win_rate_last50": s50["win_rate"],
            "sharpe_last50": s50["sharpe"],
            "sortino_last50": s50["sortino"],
            "today_pnl": today_pnl,
            "trading_halted": halted,
            "model_mae": retrain.mae if retrain else None,
            "model_samples": retrain.n_samples if retrain else None,
            "model_last_train": str(retrain.retrained_at) if retrain else None,
        }

    @app.get("/positions")
    def positions():
        if not _trade_logger:
            return []
        return [_t(t) for t in _trade_logger.get_open_trades(paper=_paper_trading)]

    @app.get("/trades")
    def trades(limit: int = Query(default=20, le=200), offset: int = Query(default=0)):
        if not _trade_logger:
            return []
        all_trades = _trade_logger.get_recent_trades(limit=limit + offset, paper=_paper_trading)
        return [_t(t) for t in all_trades[offset:offset + limit]]

    @app.get("/edges")
    def edges():
        return [
            {
                "market_id": s.market.market_id,
                "question": s.market.question,
                "category": s.market.category,
                "side": s.trade_side,
                "elo_edge": round(s.elo_edge, 4),
                "final_edge": round(s.final_edge, 4),
                "price": round(s.trade_price, 4),
                "elo_implied_prob": round(s.elo_implied_prob, 4),
                "fair_value": round(s.fair_value, 4),
                "confidence": round(s.confidence, 3),
                "kelly_fraction": round(s.kelly_fraction, 4),
                "sentiment_score": round(s.sentiment_score, 3),
                "sentiment_agrees": s.sentiment_agrees,
                "liquidity": s.market.liquidity,
                "whale_activity": s.market.whale_activity,
            }
            for s in _latest_signals[:20]
        ]

    @app.get("/elo")
    def elo():
        if not _elo_system:
            return {"error": "ELO system not available"}
        return _elo_system.get_ratings_table()

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return HTMLResponse(_DASHBOARD_HTML)

    return app


def update_signals(signals: list):
    global _latest_signals
    _latest_signals = signals


def update_status(status: dict):
    global _bot_status
    _bot_status = status


def _t(trade) -> dict:
    return {
        "id": trade.id,
        "market_id": trade.market_id,
        "question": trade.market_question or "",
        "category": trade.category or "",
        "side": trade.side,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "size": trade.size,
        "pnl": trade.pnl,
        "pnl_pct": trade.pnl_pct,
        "status": trade.status,
        "is_paper": trade.is_paper,
        "edge_at_entry": trade.edge_at_entry,
        "elo_diff": trade.elo_diff,
        "sentiment_score": trade.sentiment_score,
        "hold_time_minutes": trade.hold_time_minutes,
        "exit_reason": trade.exit_reason,
        "opened_at": str(trade.opened_at) if trade.opened_at else None,
        "closed_at": str(trade.closed_at) if trade.closed_at else None,
    }


def _fmt_uptime(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    return f"{h}h {m}m"


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Polymarket Bot</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',sans-serif;background:#0d0f1a;color:#dde}
    header{background:#131627;padding:14px 24px;border-bottom:1px solid #1e2240;display:flex;gap:16px;align-items:center}
    h1{font-size:18px;color:#7c83ff}
    .badge{padding:3px 10px;border-radius:10px;font-size:11px;font-weight:700}
    .badge.paper{background:#2b2200;color:#ffcc00}
    .badge.live{background:#330000;color:#ff4444}
    #uptime{font-size:11px;color:#555;margin-left:auto}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;padding:16px}
    .card{background:#131627;border:1px solid #1e2240;border-radius:8px;padding:14px}
    .card .lbl{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
    .card .val{font-size:22px;font-weight:700}
    .g{color:#4caf50}.r{color:#f44336}.y{color:#ffc107}.c{color:#00bcd4}.p{color:#b39ddb}
    section{padding:0 16px 16px}
    h2{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
    table{width:100%;border-collapse:collapse;background:#131627;border-radius:8px;overflow:hidden}
    th{text-align:left;padding:8px 10px;font-size:10px;color:#666;text-transform:uppercase;border-bottom:1px solid #1e2240}
    td{padding:7px 10px;font-size:12px;border-bottom:1px solid #181b2c}
    tr:last-child td{border-bottom:none}
    tr:hover td{background:#181b2c}
    #refresh{font-size:10px;color:#444}
    .halted{color:#ff4444;font-weight:bold}
    .tabs{display:flex;gap:2px;padding:0 16px 8px}
    .tab{padding:5px 14px;background:#131627;border:1px solid #1e2240;border-radius:6px 6px 0 0;cursor:pointer;font-size:12px;color:#888}
    .tab.active{background:#1e2240;color:#dde;border-bottom-color:#1e2240}
    .tab-content{display:none}.tab-content.active{display:block}
  </style>
</head>
<body>
<header>
  <h1>⚡ Polymarket Bot</h1>
  <span id="mode-badge" class="badge paper">PAPER</span>
  <span id="halted-badge" style="display:none" class="halted">⛔ HALTED TODAY</span>
  <span id="uptime">—</span>
  <span id="refresh">—</span>
</header>

<div class="grid" id="stats-grid">
  <div class="card"><div class="lbl">Today P&L</div><div class="val" id="today-pnl">—</div></div>
  <div class="card"><div class="lbl">Total P&L</div><div class="val" id="total-pnl">—</div></div>
  <div class="card"><div class="lbl">Win Rate (50)</div><div class="val" id="win-rate">—</div></div>
  <div class="card"><div class="lbl">Open Positions</div><div class="val c" id="open-pos">—</div></div>
  <div class="card"><div class="lbl">Sharpe</div><div class="val" id="sharpe">—</div></div>
  <div class="card"><div class="lbl">Model MAE</div><div class="val p" id="model-mae">—</div></div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('edges')">Edges</div>
  <div class="tab" onclick="switchTab('positions')">Positions</div>
  <div class="tab" onclick="switchTab('trades')">Trades</div>
  <div class="tab" onclick="switchTab('elo')">ELO Table</div>
</div>

<div id="tab-edges" class="tab-content active">
  <section>
    <table><thead><tr><th>Market</th><th>Side</th><th>ELO Δ</th><th>Edge</th><th>Price</th><th>ELO Prob</th><th>Conf</th><th>Kelly</th></tr></thead>
    <tbody id="edges-body"><tr><td colspan="8" style="color:#444">Loading...</td></tr></tbody></table>
  </section>
</div>

<div id="tab-positions" class="tab-content">
  <section>
    <table><thead><tr><th>ID</th><th>Market</th><th>Side</th><th>Entry</th><th>Size</th><th>ELO Δ</th><th>Opened</th></tr></thead>
    <tbody id="pos-body"><tr><td colspan="7" style="color:#444">Loading...</td></tr></tbody></table>
  </section>
</div>

<div id="tab-trades" class="tab-content">
  <section>
    <table><thead><tr><th>ID</th><th>Market</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Hold</th><th>Exit</th></tr></thead>
    <tbody id="trades-body"><tr><td colspan="8" style="color:#444">Loading...</td></tr></tbody></table>
  </section>
</div>

<div id="tab-elo" class="tab-content">
  <section>
    <h2>Category ELO Ratings</h2>
    <table id="elo-table"><thead><tr><th>Category</th><th>ELO</th><th>Δ vs Baseline</th><th>Implied P(YES)</th></tr></thead>
    <tbody id="elo-body"><tr><td colspan="4" style="color:#444">Loading...</td></tr></tbody></table>
  </section>
</div>

<script>
let activeTab='edges';
function switchTab(name){
  document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(e=>e.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  document.querySelectorAll('.tab')[['edges','positions','trades','elo'].indexOf(name)].classList.add('active');
  activeTab=name;
}
async function j(url){const r=await fetch(url);return r.json();}
function fp(v){return v!=null?v.toFixed(3):'—';}
function pnl(v){if(v==null)return'—';const c=v>=0?'g':'r';return`<span class="${c}">${v>=0?'+':''}$${v.toFixed(2)}</span>`;}
function pct(v){if(v==null)return'—';const c=v>=0?'g':'r';return`<span class="${c}">${v>=0?'+':''}${v.toFixed(1)}%</span>`;}
function tr(s,n=48){return s&&s.length>n?s.slice(0,n)+'…':(s||'—');}

async function refresh(){
try{
  const[health,stats,edges,pos,trades,elo]=await Promise.all([
    j('/health'),j('/stats'),j('/edges'),j('/positions'),j('/trades?limit=15'),j('/elo')]);

  document.getElementById('today-pnl').innerHTML=pnl(stats.today_pnl);
  document.getElementById('total-pnl').innerHTML=pnl(stats.total_pnl);
  const wr=stats.win_rate_last50||0;
  document.getElementById('win-rate').innerHTML=`<span class="${wr>=55?'g':wr>=45?'y':'r'}">${wr.toFixed(1)}%</span>`;
  document.getElementById('open-pos').textContent=stats.open_positions;
  const sh=stats.sharpe_last50||0;
  document.getElementById('sharpe').innerHTML=`<span class="${sh>=0?'g':'r'}">${sh.toFixed(2)}</span>`;
  document.getElementById('model-mae').textContent=stats.model_mae?stats.model_mae.toFixed(4):'—';
  document.getElementById('uptime').textContent='Up '+health.uptime_human;
  document.getElementById('mode-badge').textContent=health.paper_trading?'PAPER':'⚠ LIVE';
  document.getElementById('mode-badge').className='badge '+(health.paper_trading?'paper':'live');
  document.getElementById('halted-badge').style.display=stats.trading_halted?'':'none';
  document.getElementById('refresh').textContent='Updated '+new Date().toLocaleTimeString();

  // Edges
  document.getElementById('edges-body').innerHTML=edges.length?edges.map(s=>`<tr>
    <td>${tr(s.question)}</td><td><span class="c">${s.side}</span></td>
    <td><span class="g">${(s.elo_edge*100).toFixed(1)}%</span></td>
    <td><span class="g">${(s.final_edge*100).toFixed(1)}%</span></td>
    <td>${fp(s.price)}</td><td>${fp(s.elo_implied_prob)}</td>
    <td>${(s.confidence*100).toFixed(0)}%</td><td>${(s.kelly_fraction*100).toFixed(1)}%</td>
  </tr>`).join(''):'<tr><td colspan="8" style="color:#444">No signals</td></tr>';

  // Positions
  document.getElementById('pos-body').innerHTML=pos.length?pos.map(t=>`<tr>
    <td>${t.id}</td><td>${tr(t.question,40)}</td><td><span class="c">${t.side}</span></td>
    <td>${fp(t.entry_price)}</td><td>$${(t.size||0).toFixed(0)}</td>
    <td>${t.elo_diff?(t.elo_diff*100).toFixed(1)+'%':'—'}</td>
    <td>${t.opened_at?t.opened_at.slice(0,16):'—'}</td>
  </tr>`).join(''):'<tr><td colspan="7" style="color:#444">No open positions</td></tr>';

  // Trades
  document.getElementById('trades-body').innerHTML=trades.length?trades.map(t=>`<tr>
    <td>${t.id}</td><td>${tr(t.question,36)}</td><td><span class="c">${t.side}</span></td>
    <td>${fp(t.entry_price)}</td><td>${fp(t.exit_price)}</td><td>${pnl(t.pnl)}</td>
    <td>${t.hold_time_minutes?Math.round(t.hold_time_minutes)+'m':'—'}</td>
    <td>${t.exit_reason||t.status}</td>
  </tr>`).join(''):'<tr><td colspan="8" style="color:#444">No trades</td></tr>';

  // ELO
  if(elo.category){
    const cats=Object.entries(elo.category).sort((a,b)=>b[1]-a[1]);
    document.getElementById('elo-body').innerHTML=cats.map(([cat,rating])=>{
      const delta=rating-1500;const prob=1/(1+Math.pow(10,(1500-rating)/400));
      const cls=delta>=0?'g':'r';
      return`<tr><td>${cat}</td><td><span class="${cls}">${rating.toFixed(0)}</span></td>
        <td><span class="${cls}">${delta>=0?'+':''}${delta.toFixed(0)}</span></td>
        <td>${(prob*100).toFixed(1)}%</td></tr>`;
    }).join('');
  }
}catch(e){console.error(e);}
}
refresh();setInterval(refresh,5000);
</script>
</body></html>"""
