"""
BGRI: BlackRock Geopolitical Risk Indicator
===========================================
Full implementation of the z-scored NLP amplitude pipeline.

PHYSICS FRAMING
───────────────
Each sentence s is a carrier wave with:
  - amplitude  A(s) = relevance score  ∈ [0, 1]      (LM classification)
  - phase      φ(s) = sentiment sign   ∈ {-1, 0, +1} (LM classification)
  - weight     w(s) = source type      (brokerage >> news)

Raw signal for risk k on day t:
  R_k(t) = Σ_s  w(s) · A(s) · φ(s)

Z-score normalization against 5-year rolling window:
  BGRI_k(t) = [R_k(t) − μ_k(t, 5yr)] / σ_k(t, 5yr)

  BGRI = 0    → average historical attention
  BGRI = +1   → 1σ above average (elevated attention)
  BGRI = +2   → 2σ above average (extreme attention, rare event)

PSEUDOCODE
──────────────────────────────────────────────────────────────────────
INPUT:  corpus of (text, source_type, date) tuples
        risk_keywords: dict[risk_name → seed_phrases]
        fine_tuned_LM: transformer model

FOR each (article, source_type, date) in corpus:
    sentences = sentence_tokenize(article)
    FOR each sentence s:
        A(s) = fine_tuned_LM.relevance(s, risk_k)   # ∈ [0, 1]
        φ(s) = fine_tuned_LM.sentiment(s)            # ∈ {-1, 0, +1}
        w     = SOURCE_WEIGHTS[source_type]           # brokerage=2x, news=1x
        signal_contribution(s) = w · A(s) · φ(s)

    R_k(date) += Σ_s signal_contribution(s)

# Normalize to z-score over rolling 5yr window
FOR each date t:
    window   = R_k[t-5yr : t]
    μ, σ     = mean(window), std(window)
    BGRI_k(t) = (R_k(t) − μ) / σ    if σ > 0 else 0
──────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0a0c10', 'axes.facecolor': '#111318',
    'axes.edgecolor': '#2a2e38',   'axes.labelcolor': '#7a8099',
    'xtick.color': '#454a5e',      'ytick.color': '#454a5e',
    'text.color': '#e2e6f0',       'grid.color': '#1e2130',
    'grid.linewidth': 0.5,         'font.family': 'monospace',
    'axes.spines.top': False,      'axes.spines.right': False,
})

# =============================================================================
# STEP 0 — SYNTHETIC CORPUS
# =============================================================================
# In production: replace with real brokerage reports + Dow Jones News feed.
# Here we generate controlled synthetic sentences so you can see every
# mechanism (relevance, sentiment, source weighting, z-scoring) fire cleanly.

RISKS = ["Middle East War", "Tech Decoupling", "Cyber Attack",
         "Trade Protectionism", "Russia-NATO"]

SOURCE_WEIGHTS = {"brokerage": 2.0, "news": 1.0}   # BlackRock: brokerage >> news

# Seed vocabulary per risk — in production this is learned by the fine-tuned LM
RISK_VOCAB = {
    "Middle East War":     ["israel", "hamas", "iran", "hezbollah", "gaza",
                            "ceasefire", "escalation", "brent", "oil supply"],
    "Tech Decoupling":     ["nvidia", "chip export", "semiconductor", "huawei",
                            "ai competition", "decoupling", "technology ban"],
    "Cyber Attack":        ["ransomware", "cyber espionage", "malware", "hack",
                            "critical infrastructure", "state-backed", "zero-day"],
    "Trade Protectionism": ["tariff", "trade war", "import duty", "protectionism",
                            "retaliatory", "section 232", "reciprocal tariff"],
    "Russia-NATO":         ["ukraine", "russia", "nato", "escalation", "drone strike",
                            "missile", "peace talks", "zelensky", "kremlin"],
}

SENTIMENT_WORDS = {
    "positive": ["ceasefire", "peace", "truce", "agreement", "stabilize", "resolve"],
    "negative": ["escalation", "attack", "war", "threat", "conflict", "ban",
                 "missile", "hack", "ransomware", "tariff", "retaliatory",
                 "espionage", "invasion", "crisis", "strike"],
}

def generate_corpus(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic corpus of (date, sentence, source_type) tuples.
    Injects real signal bursts at known event dates.

    Pseudocode:
      FOR each day t:
          n_articles ~ Poisson(λ=5)          # random daily article count
          FOR each article:
              source ~ Bernoulli(p=0.3)       # 30% brokerage, 70% news
              sentences = generate_sentences(date, source)
      RETURN DataFrame
    """
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    # Inject amplitude bursts (simulate real events)
    EVENTS = {
        pd.Timestamp("2024-04-02"): ("Trade Protectionism", 3.0, "negative"),
        pd.Timestamp("2024-06-15"): ("Middle East War",     2.5, "negative"),
        pd.Timestamp("2024-09-10"): ("Cyber Attack",        2.0, "negative"),
        pd.Timestamp("2024-11-05"): ("Tech Decoupling",     2.8, "negative"),
        pd.Timestamp("2025-01-20"): ("Russia-NATO",         1.8, "negative"),
        pd.Timestamp("2025-03-01"): ("Trade Protectionism", 2.2, "negative"),
    }

    for date in dates:
        n_articles = rng.poisson(lam=5)
        for _ in range(n_articles):
            source = rng.choice(["brokerage","news"], p=[0.3, 0.7])
            n_sent = rng.integers(3, 12)
            for _ in range(n_sent):
                # Pick a random risk topic for this sentence
                risk = rng.choice(RISKS)
                vocab = RISK_VOCAB[risk]
                word  = rng.choice(vocab)
                # Default: moderate noise signal
                rel_base = rng.beta(2, 5)    # mostly low relevance (realistic)
                sent_base = rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                rows.append({
                    "date": date, "source": source,
                    "risk_topic": risk, "keyword": word,
                    "relevance": float(rel_base),
                    "sentiment": int(sent_base),
                })
        # Inject event burst
        if date in EVENTS:
            ev_risk, ev_amp, ev_sent = EVENTS[date]
            sent_val = -1 if ev_sent == "negative" else 1
            for _ in range(30):        # burst of high-relevance sentences
                rows.append({
                    "date": date, "source": "brokerage",
                    "risk_topic": ev_risk, "keyword": "event_burst",
                    "relevance": min(1.0, float(rng.beta(8, 2) * ev_amp / 3)),
                    "sentiment": sent_val,
                })
    return pd.DataFrame(rows)


# =============================================================================
# STEP 1 — SIMULATED NLP CLASSIFIER
# =============================================================================

class SimpleLM:
    """
    Lightweight stand-in for the fine-tuned transformer LM.
    In production: replace with a BERT/RoBERTa model fine-tuned on
    geopolitical text for each risk_k.

    classify_relevance():
      Input : sentence text (here: keyword + source as proxy)
      Output: relevance score ∈ [0, 1]
        → How much does this sentence "talk about" risk_k?
        → In practice: softmax output of classification head

    classify_sentiment():
      Input : sentence text
      Output: sentiment ∈ {-1, 0, +1}
        → Negative (-1) = risk escalating / bad news
        → Neutral  ( 0) = factual/ambiguous
        → Positive (+1) = risk de-escalating / good news
        → In practice: 3-class fine-tuned head or VADER/FinBERT
    """

    def classify_relevance(self, keyword: str, target_risk: str, source_risk: str) -> float:
        """
        Proxy for transformer relevance head.
        Real version: cosine_sim(sentence_embedding, risk_k_embedding)
        """
        base = 0.75 if source_risk == target_risk else 0.05
        # Boost if keyword appears in risk vocab
        if any(kw in keyword for kw in RISK_VOCAB.get(target_risk, [])):
            base = min(1.0, base + 0.15)
        return base

    def classify_sentiment(self, keyword: str, raw_sentiment: int) -> int:
        """
        Proxy for fine-tuned sentiment head.
        Real version: FinBERT or task-specific 3-class classifier.
        """
        # Override with keyword-based rule for demonstration clarity
        for neg_word in SENTIMENT_WORDS["negative"]:
            if neg_word in keyword:
                return -1
        for pos_word in SENTIMENT_WORDS["positive"]:
            if pos_word in keyword:
                return +1
        return raw_sentiment


# =============================================================================
# STEP 2 — RAW SIGNAL AGGREGATION
# =============================================================================

def compute_raw_signal(corpus: pd.DataFrame, lm: SimpleLM) -> pd.DataFrame:
    """
    Aggregate sentence-level scores into daily raw signal R_k(t).

    R_k(t) = Σ_s  w(source_s) · A(s, risk_k) · φ(s)

    This is a weighted inner product — amplitude modulated by phase.
    Sentences with high relevance but neutral sentiment → no net signal.
    Sentences with high relevance AND negative sentiment → large negative signal.

    Pseudocode:
      FOR each (date, sentence):
          FOR each risk_k:
              A = LM.relevance(sentence, risk_k)
              φ = LM.sentiment(sentence)
              w = SOURCE_WEIGHTS[source]
              R_k(date) += w · A · φ
    """
    results = defaultdict(lambda: defaultdict(float))

    for _, row in corpus.iterrows():
        date   = row["date"]
        source = row["source"]
        w      = SOURCE_WEIGHTS[source]

        for risk_k in RISKS:
            A = lm.classify_relevance(row["keyword"], risk_k, row["risk_topic"])
            phi = lm.classify_sentiment(row["keyword"], row["sentiment"])
            results[date][risk_k] += w * A * phi

    df = pd.DataFrame(results).T.fillna(0)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    # Smooth with 5-day rolling mean (reduce day-to-day noise)
    df = df.rolling(5, min_periods=1).mean()
    return df


# =============================================================================
# STEP 3 — Z-SCORE NORMALIZATION (rolling 5-year window)
# =============================================================================

def compute_bgri(raw_signal: pd.DataFrame,
                 window_days: int = 365) -> pd.DataFrame:
    """
    Normalize raw signal to z-score using rolling window.

    BGRI_k(t) = [R_k(t) − μ_k(window)] / σ_k(window)

    Physics: this is like expressing a measurement in units of
    'how many standard deviations above the background noise level'.
    A score of 0 = average attention. Score of +2 = extreme attention.

    The rolling window means the 'normal' baseline drifts slowly,
    so a permanently elevated risk eventually becomes the new zero —
    BlackRock explicitly implements this 'eventual normalization' concept.

    Pseudocode:
      FOR each date t:
          window_data = R_k[t - window_days : t]
          μ = mean(window_data)
          σ = std(window_data)
          BGRI_k(t) = (R_k(t) - μ) / σ   if σ > 0 else 0
    """
    bgri = pd.DataFrame(index=raw_signal.index, columns=raw_signal.columns,
                        dtype=float)
    for risk_k in raw_signal.columns:
        s = raw_signal[risk_k]
        mu    = s.rolling(window_days, min_periods=30).mean()
        sigma = s.rolling(window_days, min_periods=30).std()
        bgri[risk_k] = (s - mu) / sigma.replace(0, np.nan)

    return bgri.fillna(0)


# =============================================================================
# STEP 4 — COMPOSITE BGRI (simple average across risks)
# =============================================================================

def compute_composite(bgri: pd.DataFrame) -> pd.Series:
    """
    Composite BGRI = simple average across all tracked risks.
    Captures overall 'temperature' of geopolitical risk attention.
    """
    return bgri.mean(axis=1)


# =============================================================================
# RUN PIPELINE
# =============================================================================
print("Generating synthetic corpus...")
corpus = generate_corpus(n_days=730)
print(f"  {len(corpus):,} sentence-level records across {corpus.date.nunique()} days")

print("Running NLP classification + signal aggregation...")
lm         = SimpleLM()
raw_signal = compute_raw_signal(corpus, lm)

print("Computing BGRI z-scores...")
bgri      = compute_bgri(raw_signal, window_days=180)
composite = compute_composite(bgri)

print("Done. Plotting...")

# =============================================================================
# VISUALIZATION — 4 PANELS SHOWING EACH PIPELINE STAGE
# =============================================================================
fig = plt.figure(figsize=(20, 22), facecolor='#0a0c10')
gs  = gridspec.GridSpec(4, 2, hspace=0.50, wspace=0.30,
                        left=0.07, right=0.97, top=0.94, bottom=0.04)

RISK_COLORS = {
    "Middle East War":     '#e8462a',
    "Tech Decoupling":     '#4a9eff',
    "Cyber Attack":        '#f5a623',
    "Trade Protectionism": '#a855f7',
    "Russia-NATO":         '#4caf7d',
}

EVENT_DATES = {
    "2024-04-02": ("Trade Protectionism spike",  0.85),
    "2024-06-15": ("Middle East escalation",      0.85),
    "2024-09-10": ("Cyber attack burst",          0.85),
    "2024-11-05": ("Tech Decoupling shock",       0.85),
    "2025-01-20": ("Russia-NATO escalation",      0.85),
    "2025-03-01": ("Trade tariff reimposition",   0.85),
}

def panel_top_bar(ax, color='#e8462a', width=0.12):
    """Draw thin accent bar at top-left of panel."""
    ax.axhline(ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
               xmin=0, xmax=0, color=color)  # placeholder; drawn after


def add_event_lines(ax, dates_dict, ymax):
    for d, (label, alpha) in dates_dict.items():
        ts = pd.Timestamp(d)
        if ts in ax.get_xlim() or True:
            ax.axvline(ts, color='#f5a623', lw=0.7, ls=':', alpha=0.4, zorder=1)


# ─── PANEL A: RAW SENTENCE SCORES (sample day) ───────────────────────────────
ax_sent = fig.add_subplot(gs[0, 0])
ax_sent.set_facecolor('#111318')

# Show sentence-level relevance × sentiment for one sample day
sample_day = corpus[corpus.date == pd.Timestamp("2024-04-02")].copy()
sample_day = sample_day.head(40)
sample_day["signal"] = (
    sample_day.apply(
        lambda r: SOURCE_WEIGHTS[r.source]
                  * lm.classify_relevance(r.keyword, r.risk_topic, r.risk_topic)
                  * lm.classify_sentiment(r.keyword, r.sentiment),
        axis=1
    )
)
colors_sent = [RISK_COLORS[r] for r in sample_day.risk_topic]
x_pos = np.arange(len(sample_day))
ax_sent.bar(x_pos, sample_day["signal"], color=colors_sent, alpha=0.7,
            width=0.8, edgecolor='none')
ax_sent.axhline(0, color='#2a2e38', lw=1)
ax_sent.set_title("STAGE 1 — Sentence Signal:  w · A(s) · φ(s)",
                   fontsize=9, fontweight='bold', color='#e2e6f0',
                   loc='left', pad=8)
ax_sent.set_xlabel("Sentence index (sample: 2024-04-02)", fontsize=8)
ax_sent.set_ylabel("Signal contribution", fontsize=8)
ax_sent.tick_params(labelbottom=False)
ax_sent.grid(axis='y', ls='--', lw=0.5, alpha=0.5)

# Mini legend
patches = [mpatches.Patch(color=c, label=r, alpha=0.8)
           for r, c in RISK_COLORS.items()]
ax_sent.legend(handles=patches, fontsize=6.5, loc='upper right',
               facecolor='#1a1d24', edgecolor='#2a2e38', labelcolor='#c8cfe0',
               framealpha=0.9)

# Annotation boxes
ax_sent.text(0.01, 0.97,
    "w = source weight  (brokerage=2×, news=1×)\n"
    "A = relevance ∈ [0,1]  (LM classification head)\n"
    "φ = sentiment ∈ {−1, 0, +1}  (fine-tuned classifier)",
    transform=ax_sent.transAxes, fontsize=7, va='top',
    color='#6a7299', fontfamily='monospace',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1017',
              edgecolor='#2a2e38', alpha=0.9))


# ─── PANEL B: RAW DAILY SIGNAL R_k(t) ────────────────────────────────────────
ax_raw = fig.add_subplot(gs[0, 1])
ax_raw.set_facecolor('#111318')

for risk_k in RISKS:
    ax_raw.plot(raw_signal.index, raw_signal[risk_k],
                color=RISK_COLORS[risk_k], lw=1.2, alpha=0.75, label=risk_k)

ax_raw.axhline(0, color='#2a2e38', lw=1)
add_event_lines(ax_raw, EVENT_DATES, raw_signal.values.max())
ax_raw.set_title("STAGE 2 — Raw Signal  R_k(t) = Σ w·A·φ",
                  fontsize=9, fontweight='bold', color='#e2e6f0',
                  loc='left', pad=8)
ax_raw.set_ylabel("R_k(t)  [unnormalized]", fontsize=8)
ax_raw.legend(fontsize=6.5, loc='upper left', facecolor='#1a1d24',
              edgecolor='#2a2e38', labelcolor='#c8cfe0', framealpha=0.9)
ax_raw.grid(ls='--', lw=0.5, alpha=0.5)
ax_raw.tick_params(axis='x', labelsize=7, rotation=20)


# ─── PANEL C: BGRI Z-SCORE per risk ──────────────────────────────────────────
ax_bgri = fig.add_subplot(gs[1, :])
ax_bgri.set_facecolor('#111318')

for risk_k in RISKS:
    ax_bgri.plot(bgri.index, bgri[risk_k],
                 color=RISK_COLORS[risk_k], lw=1.6, alpha=0.82, label=risk_k)

ax_bgri.axhline(0,  color='#2a2e38', lw=1.0, ls='-')
ax_bgri.axhline(1,  color='#f5a623', lw=0.8, ls='--', alpha=0.35)
ax_bgri.axhline(2,  color='#e8462a', lw=0.8, ls='--', alpha=0.35)
ax_bgri.axhline(-1, color='#4a9eff', lw=0.8, ls='--', alpha=0.25)
ax_bgri.text(bgri.index[5], 1.05,  '+1σ threshold', fontsize=7, color='#f5a62366')
ax_bgri.text(bgri.index[5], 2.05,  '+2σ extreme',   fontsize=7, color='#e8462a66')
ax_bgri.text(bgri.index[5], -1.12, '−1σ low',        fontsize=7, color='#4a9eff55')

for d, (label, _) in EVENT_DATES.items():
    ts = pd.Timestamp(d)
    ax_bgri.axvline(ts, color='#ffffff', lw=0.6, ls=':', alpha=0.15, zorder=1)
    ax_bgri.text(ts, bgri.values.max()*0.90, label,
                 fontsize=6.5, color='#f5a62399', rotation=45,
                 ha='left', va='top', fontfamily='monospace')

ax_bgri.set_title(
    "STAGE 3 — BGRI z-score:  BGRI_k(t) = [R_k(t) − μ(window)] / σ(window)",
    fontsize=9, fontweight='bold', color='#e2e6f0', loc='left', pad=8)
ax_bgri.set_ylabel("BGRI  [σ above rolling baseline]", fontsize=8)
ax_bgri.legend(fontsize=7.5, loc='upper left', facecolor='#1a1d24',
               edgecolor='#2a2e38', labelcolor='#c8cfe0', framealpha=0.9,
               ncol=5)
ax_bgri.grid(ls='--', lw=0.5, alpha=0.4)
ax_bgri.tick_params(axis='x', labelsize=7.5, rotation=20)
ax_bgri.set_xlim(bgri.index[0], bgri.index[-1])


# ─── PANEL D: COMPOSITE BGRI + rolling stats ─────────────────────────────────
ax_comp = fig.add_subplot(gs[2, :])
ax_comp.set_facecolor('#111318')

roll_mu  = composite.rolling(30).mean()
roll_std = composite.rolling(30).std()

ax_comp.fill_between(composite.index,
                     roll_mu - roll_std, roll_mu + roll_std,
                     alpha=0.12, color='#4a9eff', label='±1σ band (30d)')
ax_comp.plot(composite.index, composite,
             color='#e2e6f0', lw=0.8, alpha=0.4, label='Daily composite')
ax_comp.plot(composite.index, roll_mu,
             color='#4a9eff', lw=2.2, label='30d rolling mean')

ax_comp.axhline(0, color='#2a2e38', lw=1)
ax_comp.axhline(1, color='#f5a623', lw=0.8, ls='--', alpha=0.4)
ax_comp.axhline(2, color='#e8462a', lw=0.8, ls='--', alpha=0.4)

for d, (label, _) in EVENT_DATES.items():
    ts = pd.Timestamp(d)
    ax_comp.axvline(ts, color='#f5a623', lw=0.8, ls=':', alpha=0.4)

# Current value annotation
last_val = composite.iloc[-1]
ax_comp.scatter(composite.index[-1], last_val, s=80,
                color='#e8462a', zorder=6, edgecolors='white', lw=0.8)
ax_comp.text(composite.index[-1], last_val + 0.12,
             f' NOW: {last_val:+.2f}σ',
             fontsize=8.5, color='#e8462a', fontfamily='monospace',
             fontweight='bold')

ax_comp.set_title("STAGE 4 — Composite BGRI = mean(BGRI_k)  across all risks",
                   fontsize=9, fontweight='bold', color='#e2e6f0',
                   loc='left', pad=8)
ax_comp.set_ylabel("Composite BGRI  [σ]", fontsize=8)
ax_comp.legend(fontsize=7.5, loc='upper left', facecolor='#1a1d24',
               edgecolor='#2a2e38', labelcolor='#c8cfe0', framealpha=0.9)
ax_comp.grid(ls='--', lw=0.5, alpha=0.4)
ax_comp.tick_params(axis='x', labelsize=7.5, rotation=20)
ax_comp.set_xlim(composite.index[0], composite.index[-1])


# ─── PANEL E: Z-score mechanics diagram ──────────────────────────────────────
ax_mech = fig.add_subplot(gs[3, :])
ax_mech.set_facecolor('#111318')

# Show raw signal vs rolling μ/σ for one risk to make z-scoring intuitive
risk_demo = "Trade Protectionism"
r_s   = raw_signal[risk_demo]
mu_s  = r_s.rolling(180, min_periods=30).mean()
sig_s = r_s.rolling(180, min_periods=30).std()
z_s   = bgri[risk_demo]

ax2 = ax_mech.twinx()

# Raw signal + baseline
ax_mech.fill_between(r_s.index, mu_s - sig_s, mu_s + sig_s,
                     alpha=0.10, color='#a855f7', label='μ ± σ  (180d window)')
ax_mech.plot(r_s.index, r_s,  color='#a855f7', lw=1.0, alpha=0.55, label='Raw R_k(t)')
ax_mech.plot(r_s.index, mu_s, color='#a855f7', lw=1.8, ls='--', alpha=0.7, label='Rolling μ')

# Z-score on right axis
ax2.plot(z_s.index, z_s, color='#f5a623', lw=2.2, label='BGRI z-score (right)')
ax2.axhline(0, color='#2a2e38', lw=0.8)
ax2.axhline(2, color='#e8462a', lw=0.8, ls='--', alpha=0.4)
ax2.set_ylabel("BGRI z-score  [σ]", fontsize=8, color='#f5a623')
ax2.tick_params(axis='y', colors='#f5a623', labelsize=7.5)

ax_mech.set_title(
    f"MECHANICS — '{risk_demo}':  how raw signal R_k(t) becomes z-score BGRI_k(t)",
    fontsize=9, fontweight='bold', color='#e2e6f0', loc='left', pad=8)
ax_mech.set_ylabel("Raw signal R_k(t)  [weighted amplitude]", fontsize=8,
                    color='#a855f7')
ax_mech.tick_params(axis='y', colors='#a855f7', labelsize=7.5)
ax_mech.tick_params(axis='x', labelsize=7.5, rotation=20)
ax_mech.grid(ls='--', lw=0.5, alpha=0.3)

# Combined legend
lines1, labels1 = ax_mech.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_mech.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7.5, loc='upper left', facecolor='#1a1d24',
               edgecolor='#2a2e38', labelcolor='#c8cfe0', framealpha=0.9)

ax_mech.set_xlim(r_s.index[0], r_s.index[-1])
ax_mech.set_facecolor('#111318')

# Annotation explaining the z-score formula
ax_mech.text(0.75, 0.92,
    "BGRI_k(t) = [R_k(t) − μ(t, 180d)] / σ(t, 180d)\n\n"
    "• When R_k >> μ  →  BGRI >> 0  (elevated attention)\n"
    "• When R_k ≈ μ   →  BGRI ≈ 0  (normal attention)\n"
    "• Rolling window: 'normal' baseline drifts → prolonged\n"
    "  high attention eventually normalises to BGRI ≈ 0",
    transform=ax_mech.transAxes, fontsize=7.5, va='top',
    color='#9aa0b4', fontfamily='monospace',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1017',
              edgecolor='#2a2e38', alpha=0.95))


# ── Master title ──────────────────────────────────────────────────────────────
fig.text(0.50, 0.972,
         "BGRI PIPELINE — z-scored NLP Amplitude: Full Implementation",
         ha='center', fontsize=15, fontweight='bold', color='#e2e6f0',
         fontfamily='monospace')
fig.text(0.50, 0.958,
         "Stage 1: sentence signal  →  Stage 2: daily aggregation  →  "
         "Stage 3: z-score normalization  →  Stage 4: composite",
         ha='center', fontsize=8.5, color='#7a8099', fontfamily='monospace')

out = '/mnt/user-data/outputs/bgri_pipeline.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0c10')
print(f"Saved → {out}")
plt.close()

# =============================================================================
# SUMMARY STATS (printed to console)
# =============================================================================
print("\n── BGRI Summary Statistics (latest 30 days) ──")
latest = bgri.tail(30).mean()
for risk_k in RISKS:
    bar = '█' * max(0, int((latest[risk_k]+2)*4))
    print(f"  {risk_k:<25s}  {latest[risk_k]:+.3f}σ  {bar}")
print(f"\n  COMPOSITE                   {composite.tail(30).mean():+.3f}σ")
