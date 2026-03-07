"""
BlackRock Geopolitical Risk Indicator (BGRI) Dashboard
=======================================================
Reconstructed from BII December 2025 methodology.

Physics framing:
  - BGRI score  = signal amplitude (NLP z-score of market attention)
  - MDS score   = phase coherence  (cosine similarity of observed vs predicted shocks)
  - Risk Map    = 2D phase space of both observables

Pseudocode (methodology):
─────────────────────────────────────────────────────
# Step 1: BGRI construction (NLP pipeline)
for sentence in corpus(brokerage_reports ∪ news):
    relevance[s] = LM.classify_relevance(sentence, risk_k)   # ∈ [0,1]
    sentiment[s] = LM.classify_sentiment(sentence)            # ∈ {-1,0,+1}
BGRI_k = Σ w(source) × relevance × sentiment
BGRI_k = (BGRI_k - μ_5yr) / σ_5yr                           # z-score

# Step 2: MDS movement score
r_mds    = assumed_shock_direction[risk_k]                    # calibrated vector
r_actual = trailing_1m_returns(MDS_assets[risk_k])
similarity = cosine(r_actual, r_mds)                          # ∈ [-1, +1]
magnitude  = ‖r_actual‖ / ‖r_mds‖
MDS_score  = similarity × magnitude                           # combined

# Step 3: Risk Map
plot(MDS_score, BGRI_score, color=likelihood)
─────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0a0c10',
    'axes.facecolor':    '#111318',
    'axes.edgecolor':    '#2a2e38',
    'axes.labelcolor':   '#7a8099',
    'xtick.color':       '#454a5e',
    'ytick.color':       '#454a5e',
    'text.color':        '#e2e6f0',
    'grid.color':        '#1e2130',
    'grid.linewidth':    0.6,
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

COLOR = {'high': '#e8462a', 'medium': '#f5a623', 'low': '#4a9eff'}

# ── Risk Data (extracted from BII Dec 2025 document) ─────────────────────────
# BGRI = z-score of market attention (vertical axis, p.5 risk map)
# MDS  = market movement score, MDS similarity × magnitude (horizontal axis)
# Both faithfully encoded from the risk map chart and qualitative ratings.

risks = pd.DataFrame([
    # name                          short                      lh        bgri   mds    delta
    ("Global Technology Decoupling","Tech Decoupling",         "high",   1.45,  0.35, +0.12),
    ("Emerging Markets Pol. Crisis","EM Political Crisis",     "medium", 1.30,  0.20, +0.08),
    ("Major Cyber Attack(s)",       "Cyber Attack",            "high",   1.20,  0.15, +0.05),
    ("Middle East Regional War",    "Middle East War",         "high",   1.05,  0.52,  0.00),
    ("Global Trade Protectionism",  "Trade Protectionism",     "medium", 0.85,  0.62, -0.15),
    ("U.S.–China Competition",      "U.S.–China Competition",  "medium", 0.78,  0.28, -0.18),
    ("Major Terror Attack(s)",      "Terror Attack",           "high",   0.65,  0.08, +0.03),
    ("Russia–NATO Conflict",        "Russia–NATO",             "medium", 0.55,  0.40,  0.00),
    ("North Korea Conflict",        "North Korea",             "medium", 0.42,  0.12, +0.06),
    ("European Fragmentation",      "European Frag.",          "low",    0.22, -0.10, -0.04),
], columns=["name","short","likelihood","bgri","mds","delta"])

# ── Historical BGRI composite (aggregate, 2017–2025, from p.5 chart) ──────────
ts_dates = [
    "Jan-17","Jun-17","Jan-18","Jun-18","Jan-19","Jun-19",
    "Jan-20","Jun-20","Jan-21","Jun-21","Jan-22","Jun-22",
    "Jan-23","Jun-23","Jan-24","Jun-24","Jan-25","Jun-25","Dec-25"
]
ts_values = [
    0.18, 0.45,  # 2017 steel tariffs
    0.62, 0.38,  # 2018 Iran deal withdrawal
    0.30, 0.10,  # 2019 Phase One nearing
   -0.15, 1.82,  # 2020 COVID pandemic
    0.55, 0.32,  # 2021 election aftermath
    0.28, 1.35,  # 2022 Russia invasion
    0.95, 0.72,  # 2023 Oct. 7 attack
    0.68, 0.85,  # 2024 Trump re-elected
    1.40, 0.95,  # 2025 April 2 tariffs
    0.82          # Dec 2025 current
]
key_events = {
    1:  "U.S. steel tariffs",
    3:  "Withdrawal\nfrom Iran deal",
    7:  "COVID-19\npandemic",
    11: "Russia invades\nUkraine",
    13: "Oct. 7 attack\non Israel",
    15: "Trump\nre-elected",
    16: "April 2\n'reciprocal tariffs'",
}

# ── MDS scenario variables (p.6) ─────────────────────────────────────────────
scenario_vars = {
    "Tech Decoupling":       [("Chinese yuan","↓"),("Chinese semis","↓"),("U.S. semis","↑")],
    "EM Political Crisis":   [("LatAm cons. staples","↓"),("EM vs DM equities","↓"),("Brazil debt","↓")],
    "Cyber Attack":          [("U.S. HY utilities","↓"),("U.S. dollar","↑"),("U.S. utilities","↓")],
    "Middle East War":       [("Brent crude","↑"),("VIX","↑"),("U.S. HY credit","↓")],
    "Trade Protectionism":   [("U.S. specialty retail","↓"),("U.S. cons. durables","↓"),("U.S. 2yr Treasury","↑")],
    "U.S.–China Competition":[("Taiwanese dollar","↓"),("Taiwan equities","↓"),("China HY","↓")],
    "Terror Attack":         [("Germany 10yr bund","↑"),("Japanese yen","↑"),("Europe airlines","↓")],
    "Russia–NATO":           [("Russian equities","↓"),("Russian ruble","↓"),("Brent crude","↑")],
    "North Korea":           [("Japanese yen","↑"),("Korean won","↓"),("Korean equities","↓")],
    "European Frag.":        [("EMEA hotels","↓"),("Italy 10yr BTP","↓"),("Russian ruble","↓")],
}

# =============================================================================
# FIGURE LAYOUT
# =============================================================================
fig = plt.figure(figsize=(22, 26), facecolor='#0a0c10')
gs  = GridSpec(4, 2, figure=fig,
               hspace=0.42, wspace=0.30,
               left=0.06, right=0.97, top=0.95, bottom=0.04)

ax_map  = fig.add_subplot(gs[0, 0])   # Risk Map
ax_bar  = fig.add_subplot(gs[0, 1])   # BGRI bar
ax_ts   = fig.add_subplot(gs[1, :])   # Time series (full width)
ax_tbl  = fig.add_subplot(gs[2, :])   # Table
ax_scen = fig.add_subplot(gs[3, :])   # Scenario vars

# ─── MASTER TITLE ─────────────────────────────────────────────────────────────
fig.text(0.50, 0.975, "BLACKROCK GEOPOLITICAL RISK DASHBOARD",
         ha='center', va='top', fontsize=16, fontweight='bold',
         color='#e2e6f0', fontfamily='monospace')
fig.text(0.50, 0.962, "BGRI · Market Attention · MDS Market Movement · December 15, 2025",
         ha='center', va='top', fontsize=9, color='#7a8099', fontfamily='monospace')

# =============================================================================
# 1. RISK MAP  — 2D phase space
# =============================================================================
ax_map.set_facecolor('#111318')
ax_map.grid(True, ls='--', lw=0.5, alpha=0.4)

# Quadrant dividers
ax_map.axvline(0.35, color='#2a2e38', lw=1.2, ls='--')
ax_map.axhline(0.70, color='#2a2e38', lw=1.2, ls='--')

# Quadrant labels
quad_kw = dict(fontsize=7, color='#2e3347', fontfamily='monospace', ha='center')
ax_map.text(0.13, 1.55, "HIGH ATTENTION\nLOW PRICING", **quad_kw)
ax_map.text(0.60, 1.55, "HIGH ATTENTION\nHIGH PRICING", **quad_kw)
ax_map.text(0.13, 0.03, "LOW ATTENTION\nLOW PRICING",  **quad_kw)
ax_map.text(0.60, 0.03, "LOW ATTENTION\nHIGH PRICING", **quad_kw)

# Plot each risk
for _, r in risks.iterrows():
    c   = COLOR[r.likelihood]
    sz  = 180 + abs(r.bgri) * 120
    ax_map.scatter(r.mds, r.bgri, s=sz, color=c, alpha=0.85,
                   edgecolors='white', linewidths=0.5, zorder=5)

    # Offset labels to avoid overlap
    offsets = {
        "Tech Decoupling":        (-0.02, 0.07),
        "EM Political Crisis":    ( 0.02, 0.07),
        "Cyber Attack":           (-0.03,-0.09),
        "Middle East War":        ( 0.02, 0.07),
        "Trade Protectionism":    ( 0.02, 0.07),
        "U.S.–China Competition": (-0.03,-0.09),
        "Terror Attack":          (-0.01, 0.07),
        "Russia–NATO":            ( 0.02, 0.07),
        "North Korea":            (-0.01,-0.09),
        "European Frag.":         ( 0.01, 0.07),
    }
    dx, dy = offsets.get(r['short'], (0.01, 0.07))
    ax_map.text(r.mds + dx, r.bgri + dy, r['short'],
                fontsize=7.5, color='#c8cfe0', fontfamily='monospace',
                ha='center', va='bottom',
                path_effects=[pe.withStroke(linewidth=2, foreground='#0a0c10')])

# Legend
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR['high'],   markersize=9, label='High',   linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR['medium'], markersize=9, label='Medium', linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR['low'],    markersize=9, label='Low',    linestyle='None'),
]
ax_map.legend(handles=legend_elements, title='Likelihood', title_fontsize=7,
              fontsize=7.5, loc='lower right', framealpha=0.3,
              facecolor='#1a1d24', edgecolor='#2a2e38',
              labelcolor='#c8cfe0')

ax_map.set_xlabel("Market Pricing  →  MDS Movement Score", fontsize=8, labelpad=6)
ax_map.set_ylabel("Market Attention  ↑  BGRI z-score",     fontsize=8, labelpad=6)
ax_map.set_title("RISK MAP — Phase Space", fontsize=10, fontweight='bold',
                  color='#e2e6f0', pad=10, loc='left')
ax_map.set_xlim(-0.30, 0.85)
ax_map.set_ylim(-0.08, 1.70)

# Top accent line

# =============================================================================
# 2. BGRI BAR CHART
# =============================================================================
ax_bar.set_facecolor('#111318')
sorted_r = risks.sort_values('bgri', ascending=True)
colors   = [COLOR[lh] for lh in sorted_r.likelihood]

bars = ax_bar.barh(sorted_r['short'], sorted_r['bgri'],
                   color=colors, alpha=0.85, height=0.65,
                   edgecolor='none')

# Value labels
for bar, val, lh in zip(bars, sorted_r['bgri'], sorted_r['likelihood']):
    ax_bar.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'+{val:.2f}σ', va='center', fontsize=8,
                color=COLOR[lh], fontfamily='monospace')

ax_bar.set_xlabel("BGRI z-score vs 5yr history", fontsize=8)
ax_bar.set_xlim(0, 1.85)
ax_bar.set_title("BGRI SCORES — Market Attention", fontsize=10, fontweight='bold',
                  color='#e2e6f0', pad=10, loc='left')
ax_bar.tick_params(axis='y', labelsize=8.5)
ax_bar.grid(axis='x', ls='--', lw=0.5, alpha=0.4)

# =============================================================================
# 3. TIME SERIES — Composite BGRI
# =============================================================================
ax_ts.set_facecolor('#111318')
x = np.arange(len(ts_dates))
y = np.array(ts_values)

# Filled area
ax_ts.fill_between(x, y, 0, alpha=0.12, color='#4a9eff')
ax_ts.plot(x, y, color='#4a9eff', lw=2, zorder=4)

# Zero line
ax_ts.axhline(0, color='#2a2e38', lw=1, ls='-')

# Event markers + labels
for idx, label in key_events.items():
    ax_ts.axvline(idx, color='#f5a623', lw=0.8, ls=':', alpha=0.5, zorder=3)
    ax_ts.scatter(idx, y[idx], s=70, color='#f5a623',
                  edgecolors='white', linewidths=0.6, zorder=6)
    va = 'bottom' if y[idx] > 0 else 'top'
    dy = 0.07 if y[idx] > 0 else -0.07
    ax_ts.text(idx, y[idx] + dy, label, fontsize=6.5, color='#f5a62399',
               ha='center', va=va, fontfamily='monospace',
               path_effects=[pe.withStroke(linewidth=2, foreground='#0a0c10')])

# Current value callout
ax_ts.scatter(len(x)-1, y[-1], s=100, color='#e8462a',
              edgecolors='white', linewidths=0.8, zorder=7)
ax_ts.text(len(x)-1.1, y[-1]+0.10, f'DEC 2025\n+{y[-1]:.2f}σ',
           fontsize=8, color='#e8462a', ha='right',
           fontfamily='monospace', fontweight='bold')

ax_ts.set_xticks(x)
ax_ts.set_xticklabels(ts_dates, rotation=40, ha='right', fontsize=7.5)
ax_ts.set_ylabel("BGRI z-score", fontsize=8)
ax_ts.set_title("COMPOSITE BGRI — 2017 to December 2025", fontsize=10,
                 fontweight='bold', color='#e2e6f0', pad=10, loc='left')
ax_ts.grid(True, ls='--', lw=0.5, alpha=0.4)

# =============================================================================
# 4. SUMMARY TABLE
# =============================================================================
ax_tbl.set_facecolor('#111318')
ax_tbl.axis('off')
ax_tbl.set_title("RISK REGISTRY — Top 10 Risks, Likelihood & Scores",
                  fontsize=10, fontweight='bold', color='#e2e6f0',
                  pad=10, loc='left', x=0.0)

col_labels = ['RISK', 'LIKELIHOOD', 'BGRI (σ)', 'MDS SCORE', 'Δ vs NOV', 'DIRECTION']
table_data = []
cell_colors= []

sorted_t = risks.sort_values(['likelihood','bgri'],
                              key=lambda c: c.map({'high':0,'medium':1,'low':2}) if c.name=='likelihood' else -c,
                              ascending=True)

for _, r in sorted_t.iterrows():
    d  = r.delta
    dc = '↑' if d > 0 else ('↓' if d < 0 else '→')
    row = [
        r['short'],
        r.likelihood.upper(),
        f"+{r.bgri:.2f}",
        f"{r.mds:+.2f}",
        f"{d:+.2f}",
        dc,
    ]
    lh_c = COLOR[r.likelihood]
    bg   = '#0f1116'
    cell_colors.append([bg, bg, bg, bg, bg, bg])
    table_data.append(row)

tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
    bbox=[0.0, 0.0, 1.0, 0.93]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

# Style header
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor('#1a1d24')
    cell.set_text_props(color='#7a8099', fontfamily='monospace', fontweight='bold')
    cell.set_edgecolor('#2a2e38')

# Style rows
lh_order = list(sorted_t.likelihood)
for i, (_, r) in enumerate(sorted_t.iterrows()):
    c   = COLOR[r.likelihood]
    lhc = c + '22'
    for j in range(len(col_labels)):
        cell = tbl[i+1, j]
        cell.set_facecolor(lhc if j == 1 else '#0f1116')
        cell.set_edgecolor('#1e2130')
        tc = c if j in (1,2) else ('#4caf7d' if (j==4 and r.delta>0) else
                                    ('#e8462a' if (j==4 and r.delta<0) else '#9aa0b4'))
        cell.set_text_props(color=tc, fontfamily='monospace')

# =============================================================================
# 5. SCENARIO VARIABLES (strip chart)
# =============================================================================
ax_scen.set_facecolor('#111318')
ax_scen.axis('off')
ax_scen.set_title("MDS SCENARIO VARIABLES — Key Asset Sensitivities per Risk",
                   fontsize=10, fontweight='bold', color='#e2e6f0',
                   pad=10, loc='left')
n_risks = len(scenario_vars)
col_w   = 1.0 / n_risks
for col_i, (risk_name, assets) in enumerate(scenario_vars.items()):
    x_c = col_w * col_i + col_w / 2
    # Get likelihood color
    lh  = risks[risks.short == risk_name].likelihood.values
    c   = COLOR[lh[0]] if len(lh) else '#7a8099'

    ax_scen.text(x_c, 0.95, risk_name, ha='center', va='top',
                 fontsize=7.5, color=c, fontfamily='monospace',
                 fontweight='bold', transform=ax_scen.transAxes)
    ax_scen.axvline(x_c + col_w/2, ymin=0, ymax=0.93, color='#1e2130', lw=0.8)

    for row_i, (asset, direction) in enumerate(assets):
        y_pos = 0.76 - row_i * 0.25
        arrow_c = '#4caf7d' if direction == '↑' else '#e8462a'
        ax_scen.text(x_c - 0.01, y_pos, direction, ha='right', va='center',
                     fontsize=11, color=arrow_c, transform=ax_scen.transAxes)
        ax_scen.text(x_c + 0.01, y_pos, asset, ha='left', va='center',
                     fontsize=7, color='#7a8099', fontfamily='monospace',
                     transform=ax_scen.transAxes)

# =============================================================================
# Save
# =============================================================================
out = '/mnt/user-data/outputs/bgri_dashboard.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='#0a0c10')
print(f"Saved → {out}")
plt.close()
