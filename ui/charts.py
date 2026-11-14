"""Plotly charts with one consistent, colorblind-checked look in light and dark mode.

Colors are assigned by *role* (series 1, 2, 3 …) in a fixed order, never by
rank, so a stock keeps its color. Text always uses ink colors, never series
colors. Every chart on the site also has a "See the numbers" table nearby.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass(frozen=True)
class Palette:
    series: tuple[str, ...]
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    surface: str
    neutral: str  # diverging midpoint
    good: str
    critical: str


LIGHT = Palette(
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    surface="#fcfcfb",
    neutral="#f0efec",
    good="#0ca30c",
    critical="#d03b3b",
)
DARK = Palette(
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"),
    ink="#f0efec",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    surface="#1a1a19",
    neutral="#383835",
    good="#0ca30c",
    critical="#d03b3b",
)
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
FONT = '"Source Sans 3", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", sans-serif'
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def palette() -> Palette:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _style(fig: go.Figure, height: int, x_title: str | None, y_title: str | None, y_format: str | None, legend: bool):
    p = palette()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=40 if legend else 16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=14, color=p.ink2),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)", font=dict(color=p.ink2)),
        hoverlabel=dict(font=dict(family=FONT, size=13), bgcolor=p.surface, bordercolor=p.axis, font_color=p.ink),
    )
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(color=p.muted, size=13)) if x_title else None,
        showgrid=False, linecolor=p.axis, tickcolor=p.axis, tickfont=dict(color=p.muted), zeroline=False,
        automargin=True, title_standoff=8,
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=p.muted, size=13)) if y_title else None,
        gridcolor=p.grid, gridwidth=1, linecolor=p.axis, tickfont=dict(color=p.muted), zeroline=False,
        tickformat=y_format, automargin=True, title_standoff=8,
    )
    return fig


def show(fig: go.Figure, key: str | None = None):
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def _thin(x, *ys, max_points: int = 800):
    """Downsample long series for snappy charts (keeps the last point)."""
    n = len(x)
    if n <= max_points:
        return (x, *ys)
    idx = np.unique(np.r_[np.arange(0, n, int(np.ceil(n / max_points))), n - 1])
    return (x[idx], *(y[idx] for y in ys))


# ------------------------------------------------------------------ Monte Carlo


def fan_chart(dates: pd.DatetimeIndex, pct: np.ndarray, invested: np.ndarray, samples: np.ndarray | None = None) -> go.Figure:
    p = palette()
    blue = p.series[0]
    a_out, a_in = (0.22, 0.45) if p is DARK else (0.14, 0.30)
    x = np.asarray(dates)
    x, p5, p25, p50, p75, p95, inv = _thin(x, pct[:, 0], pct[:, 1], pct[:, 2], pct[:, 3], pct[:, 4], invested)
    fig = go.Figure()
    if samples is not None:
        for i, path in enumerate(samples):
            xs, ys = _thin(np.asarray(dates), path, max_points=300)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", line=dict(color=rgba(p.muted, 0.35), width=1),
                name="Single simulated future", legendgroup="paths", showlegend=i == 0, hoverinfo="skip",
            ))
    fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", line=dict(width=0), showlegend=False, name="Great case (95th pct.)",
                             hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=p5, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=rgba(blue, a_out),
                             name="9 in 10 futures land in here", hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=p75, mode="lines", line=dict(width=0), showlegend=False, name="Good case (75th pct.)",
                             hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=p25, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=rgba(blue, a_in),
                             name="Half of futures land in here", hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", line=dict(color=blue, width=2.5), name="Typical future (median)",
                             hovertemplate="%{y:$,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=inv, mode="lines", line=dict(color=p.ink2, width=1.5, dash="dot"),
                             name="Money you put in", hovertemplate="%{y:$,.0f}"))
    fig.update_layout(hovermode="x unified", legend=dict(traceorder="normal"))
    return _style(fig, 440, None, "Value of your investment", "$,.0f", True)


def outcome_histogram(final_values: np.ndarray, invested: float, median: float) -> go.Figure:
    p = palette()
    hi = float(np.percentile(final_values, 99))
    vals = final_values[final_values <= hi]
    fig = go.Figure(go.Histogram(
        x=vals, nbinsx=60, marker=dict(color=p.series[0], line=dict(color=p.surface, width=1)),
        hovertemplate="Ending between %{x}<br>%{y} simulated futures<extra></extra>", name="Futures",
    ))
    fig.add_vline(x=invested, line=dict(color=p.ink2, width=1.5, dash="dot"))
    fig.add_vline(x=median, line=dict(color=p.ink, width=1.5))
    fig.add_annotation(x=invested, y=1, yref="paper", text="Money put in", showarrow=False, xanchor="right",
                       xshift=-4, yanchor="top", font=dict(color=p.ink2, size=12))
    fig.add_annotation(x=median, y=1, yref="paper", text="Typical", showarrow=False, xanchor="left",
                       xshift=4, yanchor="top", font=dict(color=p.ink, size=12))
    fig.update_layout(bargap=0.02)
    fig = _style(fig, 300, "Final value (top 1% of outcomes trimmed for readability)", "Number of futures", None, False)
    fig.update_xaxes(tickformat="$,.0f")
    return fig


# ------------------------------------------------------------------ generic


def line_chart(series: dict[str, pd.Series], y_title: str, y_format: str = "$,.0f", height: int = 380,
               hover_format: str | None = None, reference: float | None = None) -> go.Figure:
    """Multi-line chart; colors follow insertion order; names labeled at line ends."""
    p = palette()
    hover_format = hover_format or y_format
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        x, y = _thin(np.asarray(s.index), s.values)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=p.series[i % 8], width=2),
                                 hovertemplate=f"%{{y:{hover_format}}}"))
    if reference is not None:
        fig.add_hline(y=reference, line=dict(color=p.axis, width=1))
    fig.update_layout(hovermode="x unified")
    return _style(fig, height, None, y_title, y_format, len(series) > 1)


def hbar(values: pd.Series, labels: dict[str, str] | None = None, x_format: str = ".0%", height: int | None = None,
         color_index: int = 0, x_title: str | None = None) -> go.Figure:
    """Ranked horizontal bars (largest on top). `labels` maps a row name to the text at its bar end."""
    p = palette()
    values = values.sort_values()
    bar = go.Bar(
        x=values.values, y=list(values.index), orientation="h",
        marker=dict(color=p.series[color_index], line=dict(color=p.surface, width=2), cornerradius=4),
        hovertemplate=f"%{{y}}: %{{x:{x_format}}}<extra></extra>",
    )
    if labels:
        bar.text = [labels.get(k, "") for k in values.index]
        bar.textposition = "outside"
        bar.textfont = dict(color=p.ink2, size=13)
        bar.cliponaxis = False
    fig = go.Figure(bar)
    fig = _style(fig, height or max(220, 34 * len(values) + 60), x_title, None, None, False)
    fig.update_xaxes(tickformat=x_format, showgrid=True, gridcolor=p.grid, range=[0, float(values.max()) * 1.35 if len(values) else 1])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=p.ink2))
    return fig


def grouped_hbar(df: pd.DataFrame, x_format: str = ".0%", x_title: str | None = None) -> go.Figure:
    """Two or three measures per row, as side-by-side horizontal bars."""
    p = palette()
    df = df.iloc[::-1]
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Bar(
            x=df[col].values, y=list(df.index), orientation="h", name=col,
            marker=dict(color=p.series[i], line=dict(color=p.surface, width=2), cornerradius=4),
            hovertemplate=f"%{{y}} · {col}: %{{x:{x_format}}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, legend=dict(traceorder="normal"))
    fig = _style(fig, max(260, 46 * len(df) + 80), x_title, None, None, True)
    fig.update_xaxes(tickformat=x_format, showgrid=True, gridcolor=p.grid)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=p.ink2))
    return fig


def grouped_vbar(df: pd.DataFrame, y_format: str = ".0%", y_title: str | None = None, hover_extra: pd.Series | None = None) -> go.Figure:
    p = palette()
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        custom = hover_extra.values if hover_extra is not None else None
        fig.add_trace(go.Bar(
            x=list(df.index), y=df[col].values, name=col, customdata=custom,
            marker=dict(color=p.series[i], line=dict(color=p.surface, width=2), cornerradius=4),
            hovertemplate=f"%{{x}} · {col}: %{{y:{y_format}}}" + ("<br>%{customdata} predictions" if custom is not None else "") + "<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    return _style(fig, 340, None, y_title, y_format, True)


# ------------------------------------------------------------------ tournament


def model_ranking(lb: pd.DataFrame, benchmark: float, band: tuple[float, float], highlight: str | None = None) -> go.Figure:
    """Every model's accuracy as a dot, best on top, against the benchmark and the luck zone.

    Blue = learning algorithms, gray = simple rules, orange = the highlighted model.
    """
    p = palette()
    df = lb.iloc[::-1]  # plotly draws the first row at the bottom
    colors = [p.series[1] if k == highlight else (p.muted if b else p.series[0]) for k, b in zip(df.index, df["baseline"])]
    sizes = [14 if k == highlight else 10 for k in df.index]
    lo = float(min(df["accuracy"].min(), band[0], 0.5)) - 0.01
    hi = float(max(df["accuracy"].max(), band[1])) + 0.01
    fig = go.Figure()
    fig.add_vrect(x0=band[0], x1=band[1], fillcolor=rgba(p.muted, 0.16), line_width=0, layer="below")
    fig.add_vline(x=0.5, line=dict(color=p.axis, width=1))
    fig.add_vline(x=benchmark, line=dict(color=p.ink2, width=1.5, dash="dot"))
    # Thin stems from the benchmark to each dot show "better" or "worse" at a glance.
    for y, x in zip(df["name"], df["accuracy"]):
        fig.add_shape(type="line", x0=benchmark, x1=x, y0=y, y1=y, line=dict(color=rgba(p.muted, 0.5), width=1), layer="below")
    fig.add_trace(go.Scatter(
        x=df["accuracy"], y=df["name"], mode="markers",
        marker=dict(color=colors, size=sizes, line=dict(color=p.surface, width=2)),
        customdata=np.column_stack([df["family"], df["edge"] * 100, df["auc"], df["verdict"]]),
        hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>Right %{x:.1%} of the time"
                      "<br>%{customdata[1]:+.1f} points vs. benchmark · AUC %{customdata[2]:.3f}"
                      "<br>%{customdata[3]}<extra></extra>",
        showlegend=False,
    ))
    top = len(df) - 0.5
    fig.add_annotation(x=benchmark, y=top, yref="y", text="Always guess the usual direction", showarrow=False,
                       yanchor="bottom", xanchor="left", xshift=4, font=dict(color=p.ink2, size=12))
    fig.add_annotation(x=0.5, y=top, yref="y", text="Coin flip", showarrow=False, yanchor="bottom",
                       xanchor="right", xshift=-4, font=dict(color=p.muted, size=12))
    fig = _style(fig, 26 * len(df) + 110, "Share of test days predicted correctly", None, None, False)
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=p.grid, range=[lo, hi])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=p.ink2, size=13), range=[-0.7, len(df) + 0.6])
    fig.update_layout(margin=dict(t=28))
    return fig


def family_bars(df: pd.DataFrame, benchmark: float) -> go.Figure:
    """Best and average accuracy per model family, with the benchmark as a line."""
    p = palette()
    fig = grouped_hbar(df, x_format=".1%")
    lo = float(df.min().min()) - 0.02
    fig.add_vline(x=benchmark, line=dict(color=p.ink2, width=1.5, dash="dot"),
                  annotation=dict(text="Benchmark", font=dict(color=p.ink2, size=12), yanchor="bottom"),
                  annotation_position="top")
    fig.update_xaxes(range=[lo, float(max(df.max().max(), benchmark)) + 0.01])
    return fig


# ------------------------------------------------------------------ portfolio


def frontier_chart(cloud: pd.DataFrame, frontier: pd.DataFrame, stocks: pd.DataFrame, strategies: pd.DataFrame, chosen: str) -> go.Figure:
    """Risk (x) vs. return (y). Gray = random mixes and single stocks; blue = frontier; orange = your pick."""
    p = palette()
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=cloud["vol"], y=cloud["ret"], mode="markers", name="Random mixes",
        marker=dict(color=rgba(p.muted, 0.28), size=5), hoverinfo="skip",
    ))
    if not frontier.empty:
        fig.add_trace(go.Scatter(
            x=frontier["vol"], y=frontier["ret"], mode="lines", name="Efficient frontier (best possible mixes)",
            line=dict(color=p.series[0], width=2.5),
            hovertemplate="Risk %{x:.1%} → return %{y:.1%}<extra>Efficient frontier</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=stocks["vol"], y=stocks["ret"], mode="markers+text", text=list(stocks.index), textposition="top center",
        textfont=dict(color=p.ink2, size=12), name="Single stocks",
        marker=dict(color=p.ink2, size=8, symbol="circle-open", line=dict(width=2)),
        hovertemplate="%{text}: risk %{x:.1%}, return %{y:.1%}<extra>100% in one stock</extra>",
    ))
    symbols = ("diamond", "square", "triangle-up", "x")
    for i, (label, row) in enumerate(strategies[strategies.index != chosen].iterrows()):
        fig.add_trace(go.Scatter(
            x=[row["vol"]], y=[row["ret"]], mode="markers", name=label,
            marker=dict(color=p.ink2, size=10, symbol=symbols[i % 4], line=dict(color=p.surface, width=1.5)),
            hovertemplate=f"{label}: risk %{{x:.1%}}, return %{{y:.1%}}<extra></extra>",
        ))
    pick = strategies.loc[[chosen]]
    fig.add_trace(go.Scatter(
        x=pick["vol"], y=pick["ret"], mode="markers+text", text=[f"<b>Your pick: {chosen}</b>"], textposition="middle right",
        textfont=dict(color=p.ink, size=13), name="Your pick",
        marker=dict(color=p.series[1], size=18, symbol="star", line=dict(color=p.surface, width=2)),
        hovertemplate="%{text}<br>risk %{x:.1%}, return %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(hovermode="closest")
    fig = _style(fig, 480, "Risk: typical yearly ups & downs (volatility)", "Estimated yearly return", ".0%", True)
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=p.grid)
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    p = palette()
    neg, pos = p.series[7], p.series[0]  # red <-> blue, gray midpoint
    scale = [[0.0, neg], [0.5, p.neutral], [1.0, pos]]
    show_text = len(corr) <= 10
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, colorscale=scale,
        xgap=2, ygap=2,
        text=np.round(corr.values, 2) if show_text else None,
        texttemplate="%{text}" if show_text else None,
        textfont=dict(size=12),
        hovertemplate="%{y} & %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(tickvals=[-1, 0, 1],
                      ticktext=["Opposite", "Unrelated", "Same way"], tickfont=dict(color=p.muted), thickness=12, outlinewidth=0),
    ))
    fig = _style(fig, max(320, 40 * len(corr) + 120), None, None, None, False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=p.ink2))
    fig.update_xaxes(tickfont=dict(color=p.ink2), side="bottom")
    return fig


# ------------------------------------------------------------------ risk


def price_cone(history: pd.Series, future_dates: pd.DatetimeIndex, lo1, hi1, lo2, hi2) -> go.Figure:
    """Recent prices, then the forecast range fanning out into the future."""
    p = palette()
    blue = p.series[0]
    a_out, a_in = (0.22, 0.45) if p is DARK else (0.14, 0.32)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="Price so far",
                             line=dict(color=p.ink2, width=2), hovertemplate="%{y:$,.2f}"))
    fig.add_trace(go.Scatter(x=future_dates, y=hi2, mode="lines", line=dict(width=0), showlegend=False, hovertemplate="%{y:$,.2f}", name="High end (19 in 20)"))
    fig.add_trace(go.Scatter(x=future_dates, y=lo2, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=rgba(blue, a_out),
                             name="19 in 20 chance it ends in here", hovertemplate="%{y:$,.2f}"))
    fig.add_trace(go.Scatter(x=future_dates, y=hi1, mode="lines", line=dict(width=0), showlegend=False, hovertemplate="%{y:$,.2f}", name="High end (2 in 3)"))
    fig.add_trace(go.Scatter(x=future_dates, y=lo1, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=rgba(blue, a_in),
                             name="2 in 3 chance it ends in here", hovertemplate="%{y:$,.2f}"))
    fig.update_layout(hovermode="x unified", legend=dict(traceorder="normal"))
    return _style(fig, 380, None, "Price", "$,.0f", True)
