"""
Factor Analysis Visualization Module
用法：
    from visualization import (
        plot_rolling_ic, plot_yearly_ir_heatmap,
        plot_acf, plot_significance_table
    )
    每个函数返回 plotly Figure 对象，可直接：
    - fig.show()          本地预览
    - fig.write_html()    导出 HTML
    - plotly.io.to_json() 给 FastAPI 返回前端
"""

import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── 颜色系统 ──────────────────────────────────────────────
PALETTE = {
    "bg":        "#0d1117",   # 深背景
    "surface":   "#161b22",   # 卡片背景
    "border":    "#30363d",   # 分隔线
    "text":      "#e6edf3",   # 主文字
    "muted":     "#8b949e",   # 次要文字
    "positive":  "#3fb950",   # 绿（正 IC）
    "negative":  "#f85149",   # 红（负 IC）
    "accent":    "#58a6ff",   # 蓝（高亮）
    "warn":      "#d29922",   # 黄（警告）
}

# 持有期排序
PERIOD_ORDER = ["1DaysHoldingPeriod", "5DaysHoldingPeriod", "20DaysHoldingPeriod"]

# 因子颜色映射（每个因子一个固定颜色，跨图一致）
FACTOR_COLORS = {
    "TwentyDayAvgVol":      "#58a6ff",
    "TwentyDayVolatility":  "#3fb950",
    "TwentyDayNegVolatility": "#bc8cff",
    "5DayMomentum":         "#d29922",
    "ShortTermReversal":    "#f85149",
    "VolPriCorr":           "#39d353",
    "DailyReturn":          "#8b949e",
    "ExcessReturn":         "#6e7681",
}

LAYOUT_BASE = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["surface"],
    font=dict(family="JetBrains Mono, monospace", color=PALETTE["text"], size=12),
    margin=dict(l=60, r=40, t=60, b=60),
    legend=dict(
        bgcolor=PALETTE["surface"],
        bordercolor=PALETTE["border"],
        borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
)


def _parse_col(col: str) -> tuple[str, str]:
    """'TwentyDayAvgVol_20DaysHoldingPeriod' → ('TwentyDayAvgVol', '20DaysHoldingPeriod')"""
    parts = col.rsplit("_", 1)
    if len(parts) == 2 and parts[1].endswith("HoldingPeriod"):
        return parts[0], parts[1]
    return col, ""


def _parse_acf_index(idx: str) -> tuple[str, str, int] | None:
    """'TwentyDayAvgVol_20DaysHoldingPeriod_30_ACF' → ('TwentyDayAvgVol', '20DaysHoldingPeriod', 30)"""
    m = re.match(r"(.+?)_(\d+DaysHoldingPeriod)_(\d+)_ACF", idx)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None


def _period_label(p: str) -> str:
    return p.replace("DaysHoldingPeriod", "D").replace("Days", "D")


# ── 1. Rolling IC 曲线 ────────────────────────────────────
def plot_rolling_ic(
    rolling_df: pd.DataFrame,
    holding_period: str = "20DaysHoldingPeriod",
    window_label: str = "126D",
) -> go.Figure:
    """
    画指定持有期的所有因子 Rolling IC 曲线。
    rolling_df: index=日期, columns=因子_持有期
    """
    cols = [c for c in rolling_df.columns if c.endswith(holding_period)]
    
    fig = go.Figure()

    for col in cols:
        factor, _ = _parse_col(col)
        color = FACTOR_COLORS.get(factor, PALETTE["muted"])
        series = rolling_df[col].dropna()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=factor,
            line=dict(color=color, width=1.5),
            hovertemplate=f"<b>{factor}</b><br>%{{x|%Y-%m-%d}}<br>IC = %{{y:.4f}}<extra></extra>",
        ))

    # 零线
    fig.add_hline(y=0, line=dict(color=PALETTE["muted"], dash="dash", width=1))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"Rolling IC — {_period_label(holding_period)} Holding Period ({window_label} window)",
            font=dict(size=15, color=PALETTE["text"]),
        ),
        xaxis_title="Date",
        yaxis_title="Rolling IC Mean",
        hovermode="x unified",
        height=450,
    )
    return fig


# ── 2. 分年 IR 热力图 ─────────────────────────────────────
def plot_yearly_ir_heatmap(
    yearly_df: pd.DataFrame,
    holding_period: str = "20DaysHoldingPeriod",
) -> go.Figure:
    """
    yearly_df: 多级 index (year, 因子_持有期), 列包含 IR
    """
    ir_pivot = yearly_df["IR"].unstack(level=0)  # index=因子_持有期, columns=年份
    # 只保留指定持有期
    ir_pivot = ir_pivot[ir_pivot.index.str.endswith(holding_period)]
    ir_pivot.index = [_parse_col(i)[0] for i in ir_pivot.index]
    ir_pivot = ir_pivot.sort_index()

    # 按各因子平均 IR 排序（绝对值降序）
    ir_pivot = ir_pivot.reindex(
        ir_pivot.abs().mean(axis=1).sort_values(ascending=False).index
    )

    z = ir_pivot.values
    x = [str(c) for c in ir_pivot.columns]
    y = ir_pivot.index.tolist()

    # 对称色阶
    zmax = max(abs(z.min()), abs(z.max()))

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[
            [0.0,  PALETTE["negative"]],
            [0.5,  PALETTE["surface"]],
            [1.0,  PALETTE["positive"]],
        ],
        zmid=0, zmin=-zmax, zmax=zmax,
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>Year: %{x}<br>IR = %{z:.4f}<extra></extra>",
        colorbar=dict(
            title=dict(text="IR", font=dict(color=PALETTE["text"])),
            tickfont=dict(color=PALETTE["text"]),
            bgcolor=PALETTE["surface"],
            bordercolor=PALETTE["border"],
        ),
    ))

    heatmap_layout = {k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **heatmap_layout,
        title=dict(
            text=f"Yearly IR Heatmap — {_period_label(holding_period)} Holding Period",
            font=dict(size=15, color=PALETTE["text"]),
        ),
        xaxis=dict(title="Year", side="bottom", gridcolor=PALETTE["border"]),
        yaxis=dict(title="Factor", gridcolor=PALETTE["border"]),
        height=max(300, len(y) * 45 + 120),
    )
    return fig


# ── 3. ACF 柱状图 ─────────────────────────────────────────
def plot_acf(
    acf_df: pd.DataFrame,
    holding_period: str = "20DaysHoldingPeriod",
) -> go.Figure:
    """
    acf_df: index=因子_持有期_滞后期_ACF, columns=['ACF']
    每个因子画一个子图，x 轴是滞后期，y 轴是 ACF 值
    """
    # 解析所有行
    records = []
    for idx, row in acf_df.iterrows():
        parsed = _parse_acf_index(idx)
        if parsed:
            factor, period, lag = parsed
            records.append({"factor": factor, "period": period, "lag": lag, "acf": row["ACF"]})
    df = pd.DataFrame(records)
    df = df[df["period"] == holding_period]

    factors = sorted(df["factor"].unique())
    n = len(factors)
    cols_n = 3
    rows_n = (n + cols_n - 1) // cols_n

    fig = make_subplots(
        rows=rows_n, cols=cols_n,
        subplot_titles=factors,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, factor in enumerate(factors):
        r, c = divmod(i, cols_n)
        sub = df[df["factor"] == factor].sort_values("lag")
        color = FACTOR_COLORS.get(factor, PALETTE["muted"])

        bar_colors = [
            PALETTE["positive"] if v >= 0 else PALETTE["negative"]
            for v in sub["acf"]
        ]

        fig.add_trace(
            go.Bar(
                x=sub["lag"].astype(str) + "D",
                y=sub["acf"],
                marker_color=bar_colors,
                name=factor,
                showlegend=False,
                hovertemplate=f"<b>{factor}</b><br>Lag = %{{x}}<br>ACF = %{{y:.4f}}<extra></extra>",
            ),
            row=r + 1, col=c + 1,
        )
        # 显著性阈值线 ±1.96/√n（n≈1374）
        threshold = 1.96 / (1374 ** 0.5)
        fig.add_hline(y=threshold, line=dict(color=PALETTE["warn"], dash="dot", width=1), row=r + 1, col=c + 1)
        fig.add_hline(y=-threshold, line=dict(color=PALETTE["warn"], dash="dot", width=1), row=r + 1, col=c + 1)
        fig.add_hline(y=0, line=dict(color=PALETTE["muted"], width=0.5), row=r + 1, col=c + 1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"ACF — {_period_label(holding_period)} Holding Period  |  dashed = ±1.96/√n significance band",
            font=dict(size=14, color=PALETTE["text"]),
        ),
        height=rows_n * 220 + 100,
    )
    # 统一子图背景
    for key in fig.layout:
        if key.startswith("xaxis") or key.startswith("yaxis"):
            fig.layout[key].update(
                gridcolor=PALETTE["border"],
                zerolinecolor=PALETTE["border"],
                tickfont=dict(color=PALETTE["muted"], size=9),
            )
    for ann in fig.layout.annotations:
        ann.font.color = PALETTE["text"]
        ann.font.size = 11

    return fig


# ── 4. 显著性检验表格 ─────────────────────────────────────
def plot_significance_table(sig_df: pd.DataFrame) -> go.Figure:
    """
    sig_df: index=因子_持有期, columns=[t, p_value, significant, Bonferroni_significant, Rank, BH_significant]
    """
    df = sig_df.copy().reset_index()
    df.columns = ["Factor_Period"] + list(sig_df.columns)

    # 拆分因子名和持有期
    df["Factor"] = df["Factor_Period"].apply(lambda x: _parse_col(x)[0])
    df["Period"] = df["Factor_Period"].apply(lambda x: _period_label(_parse_col(x)[1]))

    # 按 p_value 升序排列
    df = df.sort_values("p_value")

    # 颜色：BH 显著→绿，Bonferroni→蓝，普通显著→黄，不显著→默认
    def row_color(r):
        if r["BH_significant"]:
            return PALETTE["positive"]
        if r["Bonferroni_significant"]:
            return PALETTE["accent"]
        if r["significant"]:
            return PALETTE["warn"]
        return PALETTE["muted"]

    cell_colors = []
    fill_colors = [row_color(row) for _, row in df.iterrows()]

    # t 值颜色
    t_colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in df["t"]]

    def fmt_p(p):
        if p < 1e-6:
            return f"{p:.2e}"
        return f"{p:.4f}"

    fig = go.Figure(go.Table(
        columnwidth=[220, 80, 60, 100, 80, 80, 80],
        header=dict(
            values=["<b>Factor</b>", "<b>Period</b>", "<b>t</b>",
                    "<b>p-value</b>", "<b>p<0.05</b>",
                    "<b>Bonferroni</b>", "<b>BH</b>"],
            fill_color=PALETTE["border"],
            font=dict(color=PALETTE["text"], size=12),
            align="center",
            height=32,
        ),
        cells=dict(
            values=[
                df["Factor"].tolist(),
                df["Period"].tolist(),
                [f"{v:.3f}" for v in df["t"]],
                [fmt_p(v) for v in df["p_value"]],
                ["✓" if v else "✗" for v in df["significant"]],
                ["✓" if v else "✗" for v in df["Bonferroni_significant"]],
                ["✓" if v else "✗" for v in df["BH_significant"]],
            ],
            fill_color=[
                [PALETTE["surface"]] * len(df),
                [PALETTE["surface"]] * len(df),
                t_colors,
                [PALETTE["surface"]] * len(df),
                [PALETTE["positive"] if v else PALETTE["surface"] for v in df["significant"]],
                [PALETTE["accent"] if v else PALETTE["surface"] for v in df["Bonferroni_significant"]],
                [PALETTE["positive"] if v else PALETTE["surface"] for v in df["BH_significant"]],
            ],
            font=dict(color=PALETTE["text"], size=11),
            align=["left", "center", "right", "right", "center", "center", "center"],
            height=28,
        ),
    ))

    fig.update_layout(
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        title=dict(
            text="Significance Test  |  🟢 BH  🔵 Bonferroni  🟡 p<0.05",
            font=dict(size=14, color=PALETTE["text"]),
        ),
        height=len(df) * 30 + 120,
    )
    return fig


# ── 导出所有图到一个 HTML ─────────────────────────────────
def export_dashboard(
    rolling_df: pd.DataFrame,
    acf_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    output_path: str = "factor_analysis_dashboard.html",
    holding_period: str = "20DaysHoldingPeriod",
):
    """把四张图拼成一个独立 HTML 文件，可直接在浏览器打开"""
    from plotly.subplots import make_subplots
    import plotly.io as pio

    figs = [
        plot_rolling_ic(rolling_df, holding_period),
        plot_yearly_ir_heatmap(yearly_df, holding_period),
        plot_acf(acf_df, holding_period),
        plot_significance_table(sig_df),
    ]

    # 拼成单个 HTML
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Factor Analysis Dashboard</title>
<style>
  body { background: #0d1117; margin: 0; padding: 20px; font-family: 'JetBrains Mono', monospace; }
  .chart { margin-bottom: 32px; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
  h1 { color: #e6edf3; font-size: 18px; margin: 0 0 24px 0; letter-spacing: 0.05em; }
  .tag { color: #58a6ff; font-size: 12px; }
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<h1>Factor Analysis Dashboard &nbsp;<span class="tag">quant-factor-mining v0.1</span></h1>
"""]

    for i, fig in enumerate(figs):
        div_id = f"chart_{i}"
        fig_json = pio.to_json(fig)
        html_parts.append(f"""
<div class="chart">
  <div id="{div_id}"></div>
  <script>
    var fig = {fig_json};
    Plotly.newPlot('{div_id}', fig.data, fig.layout, {{responsive: true}});
  </script>
</div>
""")

    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    print(f"Dashboard exported to {output_path}")


# ── 本地测试入口 ──────────────────────────────────────────
if __name__ == "__main__":
    rolling = pd.read_excel("tmp/stationary.xlsx", sheet_name="rolling_ic", index_col=0)
    acf     = pd.read_excel("tmp/stationary.xlsx", sheet_name="acf", index_col=0)
    yearly  = pd.read_excel("tmp/stationary.xlsx", sheet_name="yearly", index_col=[0, 1])
    sig     = pd.read_csv("tmp/significant_test.csv", index_col=0)

    export_dashboard(rolling, acf, yearly, sig, output_path="tmp/factor_analysis_dashboard.html")
