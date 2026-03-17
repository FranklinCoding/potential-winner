"""
Rich CLI dashboard — refreshes every 5 seconds.
Shows: bankroll, today P&L, win rate (last 50), open positions,
top 5 edges, top 5 category ELOs, model accuracy.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from logger.trade_logger import Trade, TradeLogger

# Force UTF-8 safe output on Windows (avoids cp1252 UnicodeEncodeError).
# On Linux/Render, let Rich respect whether stdout is a real TTY so that
# ANSI escape codes are not emitted into non-interactive log streams.
import sys
if sys.platform == "win32":
    console = Console(
        highlight=False, force_terminal=True,
        file=open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False),
    )
else:
    console = Console(highlight=False)


def _c(value: Optional[float], fmt: str = "+.2f", suffix: str = "") -> Text:
    if value is None:
        return Text("—", style="dim")
    color = "green" if value >= 0 else "red"
    return Text(f"{value:{fmt}}{suffix}", style=color)


def _pct(v: Optional[float]) -> Text:
    return _c(v, fmt="+.1f", suffix="%")


def _fp(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _trunc(s: Optional[str], n: int = 42) -> str:
    if not s:
        return "—"
    return s[:n] + ("…" if len(s) > n else "")


class CLIDashboard:
    def __init__(
        self,
        trade_logger: TradeLogger,
        paper_trading: bool = True,
        elo_system=None,
        bankroll: float = 1000.0,
    ):
        self.trade_logger = trade_logger
        self.paper_trading = paper_trading
        self.elo_system = elo_system
        self.bankroll = bankroll
        self._signals: list = []
        self._status: str = "Initializing..."
        self._model_mae: float = 0.0
        self._model_samples: int = 0
        self._snapshot_count: int = 0
        self._cycle: int = 0

    def set_signals(self, signals: list):
        self._signals = signals

    def set_status(self, msg: str):
        self._status = msg

    def set_model_info(self, mae: float, samples: int):
        self._model_mae = mae
        self._model_samples = samples

    def set_snapshot_count(self, n: int):
        self._snapshot_count = n

    def set_cycle(self, n: int):
        self._cycle = n

    # ── Panels ────────────────────────────────────────────────────────────────

    def _header(self) -> Panel:
        mode = "[bold yellow]PAPER[/]" if self.paper_trading else "[bold red]!! LIVE[/]"
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        return Panel(
            f"[bold cyan]Polymarket Bot[/] | {mode} | {now} | Cycle #{self._cycle}\n"
            f"[dim]{self._status}[/]",
            style="bold",
        )

    def _stats(self) -> Panel:
        stats = self.trade_logger.get_stats(paper=self.paper_trading)
        stats50 = self.trade_logger.get_stats(paper=self.paper_trading, last_n=50)
        today_pnl = self.trade_logger.get_today_pnl(paper=self.paper_trading)
        halted = self.trade_logger.is_trading_halted_today(paper=self.paper_trading)

        g = Table.grid(padding=(0, 2))
        g.add_column(style="dim", width=20)
        g.add_column(style="bold")

        wr = stats50["win_rate"]
        wr_color = "green" if wr >= 55 else ("yellow" if wr >= 45 else "red")
        pnl_color = "green" if stats["total_pnl"] >= 0 else "red"
        today_color = "green" if today_pnl >= 0 else "red"

        g.add_row("Bankroll", f"[cyan]${self.bankroll:.2f}[/]")
        g.add_row("Today P&L", f"[{today_color}]{today_pnl:+.2f}[/]" + (" [red](HALTED)[/]" if halted else ""))
        g.add_row("Total P&L", f"[{pnl_color}]{stats['total_pnl']:+.2f}[/]")
        g.add_row("Win Rate (50)", f"[{wr_color}]{wr:.1f}%[/]")
        g.add_row("Wins/Losses", f"[green]{stats['wins']}[/]/[red]{stats['losses']}[/]")
        g.add_row("Open Positions", f"[cyan]{stats['open_positions']}[/]")
        g.add_row("Sharpe", f"{stats['sharpe']:+.2f}")
        g.add_row("Sortino", f"{stats['sortino']:+.2f}")
        g.add_row("Snapshots DB", str(self._snapshot_count))
        g.add_row("Model MAE", f"{self._model_mae:.4f} (n={self._model_samples})")
        return Panel(g, title="[bold]Performance[/]", border_style="cyan")

    def _elo_panel(self) -> Panel:
        t = Table(show_header=True, header_style="bold yellow", show_lines=False)
        t.add_column("Category", width=14)
        t.add_column("ELO", width=8)
        t.add_column("P(YES)", width=8)

        if self.elo_system:
            from model.elo import _elo_to_probability
            top = self.elo_system.get_top_categories(5)
            for cat, rating in top:
                prob = _elo_to_probability(rating)
                diff = rating - 1500
                color = "green" if diff > 0 else "red"
                t.add_row(
                    cat.title(),
                    f"[{color}]{rating:.0f}[/]",
                    f"{prob:.1%}",
                )
        else:
            t.add_row("[dim]ELO system not available[/]", "", "")
        return Panel(t, title="[bold]Top Category ELOs[/]", border_style="yellow")

    def _signals_panel(self) -> Panel:
        t = Table(show_header=True, header_style="bold magenta", show_lines=False)
        t.add_column("Market", max_width=38, no_wrap=True)
        t.add_column("Side", width=5)
        t.add_column("ELO d", width=8)
        t.add_column("Edge", width=8)
        t.add_column("Conf", width=6)
        t.add_column("Kelly%", width=7)

        for s in self._signals[:5]:
            e_color = "bold green" if s.final_edge > 0.15 else "green"
            t.add_row(
                _trunc(s.market.question, 36),
                f"[cyan]{s.trade_side}[/]",
                f"[{e_color}]{s.elo_edge:.1%}[/]",
                f"[{e_color}]{s.final_edge:.1%}[/]",
                f"{s.confidence:.0%}",
                f"{s.kelly_fraction:.1%}",
            )
        if not self._signals:
            t.add_row("[dim]No signals[/]", "", "", "", "", "")
        return Panel(t, title=f"[bold]Top Edges ({len(self._signals)} total)[/]", border_style="magenta")

    def _open_positions(self) -> Panel:
        trades = self.trade_logger.get_open_trades(paper=self.paper_trading)
        t = Table(show_header=True, header_style="bold blue", show_lines=False)
        t.add_column("ID", width=4)
        t.add_column("Market", max_width=36, no_wrap=True)
        t.add_column("Side", width=5)
        t.add_column("Entry", width=7)
        t.add_column("$Size", width=7)
        t.add_column("ELO d", width=7)
        for tr in trades[:8]:
            t.add_row(
                str(tr.id),
                _trunc(tr.market_question, 34),
                f"[cyan]{tr.side}[/]",
                _fp(tr.entry_price),
                f"${tr.size:.0f}",
                f"{tr.elo_diff:.1%}" if tr.elo_diff else "—",
            )
        if not trades:
            t.add_row("—", "[dim]No open positions[/]", "", "", "", "")
        return Panel(t, title=f"[bold]Open Positions ({len(trades)})[/]", border_style="blue")

    def _recent_trades(self) -> Panel:
        trades = self.trade_logger.get_recent_trades(limit=8, paper=self.paper_trading)
        t = Table(show_header=True, header_style="bold green", show_lines=False)
        t.add_column("ID", width=4)
        t.add_column("Market", max_width=34, no_wrap=True)
        t.add_column("Side", width=5)
        t.add_column("Entry", width=7)
        t.add_column("Exit", width=7)
        t.add_column("P&L", width=10)
        t.add_column("Reason", width=12)
        for tr in trades:
            s_color = {"open": "blue", "closed": "green", "cancelled": "red"}.get(tr.status, "dim")
            t.add_row(
                str(tr.id),
                _trunc(tr.market_question, 32),
                f"[cyan]{tr.side}[/]",
                _fp(tr.entry_price),
                _fp(tr.exit_price),
                _c(tr.pnl, fmt="+.2f", suffix=""),
                f"[{s_color}]{tr.exit_reason or tr.status}[/]",
            )
        if not trades:
            t.add_row("—", "[dim]No trades[/]", "", "", "", "", "")
        return Panel(t, title="[bold]Recent Trades[/]", border_style="green")

    # ── Render ────────────────────────────────────────────────────────────────

    def render_once(self):
        console.clear()
        console.print(self._header())
        console.print(Columns([self._stats(), self._elo_panel()], equal=True))
        console.print(Columns([self._signals_panel(), self._open_positions()], equal=True))
        console.print(self._recent_trades())

    def run_live(self, refresh_interval: float = 5.0, stop_event: Optional[threading.Event] = None):
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                layout = Layout()
                layout.split_column(
                    Layout(self._header(), size=4),
                    Layout(name="row2", size=18),
                    Layout(name="row3", size=12),
                    Layout(self._recent_trades(), size=12),
                )
                layout["row2"].split_row(
                    Layout(self._stats(), ratio=1),
                    Layout(self._elo_panel(), ratio=1),
                    Layout(self._signals_panel(), ratio=2),
                )
                layout["row3"].update(self._open_positions())
                live.update(layout)
                if stop_event and stop_event.is_set():
                    break
                time.sleep(refresh_interval)
