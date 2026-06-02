import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from datetime import date
from scipy.stats import pearsonr, t as t_dist, shapiro, gaussian_kde, probplot as scipy_probplot

# ════════════════════════════════════════════════════════════════════════════════
# ScreenTime Duo · Bildschirmzeit-Vergleich zweier Personen
# Phase 0–2: UI/UX · Kategorien · Produktivitäts-Score · Verdrängungsanalyse
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ScreenTime Duo",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Farben & Konstanten ─────────────────────────────────────────────────────────
P1_COLOR = "#17b3a6"   # Person 1 · Teal
P2_COLOR = "#ec6ca6"   # Person 2 · Rose

INK   = "#23252f"
MUTED = "#8b8f9e"
GRID  = "#ecebe6"
LINE  = "#e3e0d8"
PAPER = "#ffffff"

# Funktionale Kategorien (Reihenfolge = Anzeigereihenfolge)
CATEGORIES = ["Produktivität", "Lernen", "Kommunikation",
              "Unterhaltung", "Soziale Medien", "Sonstiges"]

CAT_COLORS = {
    "Produktivität":  "#6366f1",
    "Lernen":         "#10b981",
    "Kommunikation":  "#3b82f6",
    "Unterhaltung":   "#f59e0b",
    "Soziale Medien": "#ef4444",
    "Sonstiges":      "#9aa3b2",
}

# Standard-Produktivitätsgewichte pro Kategorie (−1 … +1), einstellbar im UI
DEFAULT_WEIGHTS = {
    "Produktivität":   1.0,
    "Lernen":          1.0,
    "Kommunikation":   0.0,
    "Unterhaltung":   -0.5,
    "Soziale Medien": -1.0,
    "Sonstiges":       0.0,
}

# Preset: bekannte Apps → Kategorie (Schlüssel klein geschrieben). Überschreibbar im UI.
APP_PRESET = {
    # Kommunikation
    "whatsapp": "Kommunikation", "telegram": "Kommunikation", "signal": "Kommunikation",
    "messenger": "Kommunikation", "imessage": "Kommunikation", "discord": "Kommunikation",
    # Produktivität / Tools
    "slack": "Produktivität", "teams": "Produktivität", "microsoft teams": "Produktivität",
    "zoom": "Produktivität", "gmail": "Produktivität", "mail": "Produktivität",
    "outlook": "Produktivität", "notion": "Produktivität", "obsidian": "Produktivität",
    "google docs": "Produktivität", "docs": "Produktivität", "word": "Produktivität",
    "excel": "Produktivität", "onenote": "Produktivität", "maps": "Produktivität",
    "google maps": "Produktivität", "calendar": "Produktivität",
    # Soziale Medien
    "instagram": "Soziale Medien", "tiktok": "Soziale Medien", "snapchat": "Soziale Medien",
    "facebook": "Soziale Medien", "twitter": "Soziale Medien", "x": "Soziale Medien",
    "reddit": "Soziale Medien", "pinterest": "Soziale Medien", "linkedin": "Soziale Medien",
    # Unterhaltung
    "youtube": "Unterhaltung", "netflix": "Unterhaltung", "spotify": "Unterhaltung",
    "twitch": "Unterhaltung", "disney+": "Unterhaltung", "prime video": "Unterhaltung",
    "amazon prime video": "Unterhaltung",
    # Lernen
    "duolingo": "Lernen", "anki": "Lernen", "coursera": "Lernen", "udemy": "Lernen",
    "khan academy": "Lernen", "kindle": "Lernen", "books": "Lernen",
    # Browser → bewusst neutral
    "chrome": "Sonstiges", "safari": "Sonstiges", "firefox": "Sonstiges",
}

WD_MAP = {0: "Mo", 1: "Di", 2: "Mi", 3: "Do", 4: "Fr", 5: "Sa", 6: "So"}
WD_ORDER = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# Bekannte App-Namen mit korrekter Schreibweise (für Title-Case-Override)
DISPLAY_NAMES = {
    "whatsapp": "WhatsApp", "youtube": "YouTube", "linkedin": "LinkedIn",
    "tiktok": "TikTok", "instagram": "Instagram", "snapchat": "Snapchat",
    "netflix": "Netflix", "spotify": "Spotify", "discord": "Discord",
    "telegram": "Telegram", "twitter": "Twitter", "reddit": "Reddit",
    "duolingo": "Duolingo", "pinterest": "Pinterest", "facebook": "Facebook",
    "outlook": "Outlook", "onenote": "OneNote", "notion": "Notion",
}

# ── Globales Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Hanken+Grotesk:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Hanken Grotesk', sans-serif; }

  .stApp { background: #f6f5f1; color: #23252f; }

  h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600; color: #23252f; letter-spacing: -.01em; }

  /* Hero */
  .hero {
    background: linear-gradient(135deg, #ffffff 0%, #fbfaf7 100%);
    border: 1px solid #e8e6df;
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(35,37,47,.04), 0 8px 24px rgba(35,37,47,.05);
  }
  .hero::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #17b3a6 0%, #ec6ca6 100%);
  }
  .hero h1 { font-size: 2.5rem; margin: 0 0 .3rem; line-height: 1.05; }
  .hero p  { color: #6b7280; margin: 0; font-size: 1rem; font-weight: 400; }

  /* Karten */
  .metric-card {
    background: #ffffff;
    border: 1px solid #e8e6df;
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    text-align: left;
    transition: transform .15s ease, box-shadow .15s ease;
    box-shadow: 0 1px 2px rgba(35,37,47,.04);
    height: 100%;
  }
  .metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(35,37,47,.08); }
  .metric-label { color: #9ca3af; font-size: .72rem; text-transform: uppercase;
    letter-spacing: .1em; font-weight: 600; }
  .metric-value { color: #23252f; font-family: 'Fraunces', serif;
    font-size: 1.9rem; font-weight: 600; margin: .25rem 0 0; line-height: 1; }
  .metric-sub   { color: #b6bac4; font-size: .74rem; margin-top: .35rem; }

  /* Personen-Tags */
  .tag-p1 { background: #d7f5f1; color: #0c8c81; padding: 4px 13px;
    border-radius: 20px; font-size: .8rem; font-weight: 600; }
  .tag-p2 { background: #fbe1ee; color: #c93d80; padding: 4px 13px;
    border-radius: 20px; font-size: .8rem; font-weight: 600; }

  /* Kategorie-Chips */
  .cat-chip { display:inline-block; padding:3px 11px; border-radius:20px;
    font-size:.76rem; font-weight:600; margin:2px 4px 2px 0; }

  /* Sidebar */
  div[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8e6df; }
  div[data-testid="stSidebar"] * { color: #374151; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    color: #9ca3af; font-family: 'Hanken Grotesk', sans-serif;
    font-size: .9rem; font-weight: 600; padding: .4rem .2rem;
  }
  .stTabs [aria-selected="true"] { color: #17b3a6 !important; border-bottom: 2px solid #17b3a6; }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

  /* Section-Titel */
  .section-title {
    font-family: 'Hanken Grotesk', sans-serif; color: #9ca3af; font-size: .73rem;
    text-transform: uppercase; letter-spacing: .12em; font-weight: 700;
    margin: 1.8rem 0 .9rem; border-bottom: 1px solid #e8e6df; padding-bottom: .45rem;
  }
  .lead { color: #6b7280; font-size: .9rem; line-height: 1.55; margin: 0 0 1rem; }
  div[data-testid="stMarkdownContainer"] p { color: #374151; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Plotly-Basistemplate ─────────────────────────────────────────────────────────
# Achsen-Styling (Grid/Linien) kommt aus dem Default-Template "duo".
# CS enthält bewusst KEIN xaxis/yaxis, damit update_layout(**CS, xaxis=…, yaxis=…)
# kein doppeltes Keyword-Argument auslöst (sonst TypeError beim App-Start).
pio.templates["duo"] = go.layout.Template(layout=dict(
    plot_bgcolor=PAPER, paper_bgcolor=PAPER,
    font=dict(color=INK, family="Hanken Grotesk"),
    xaxis=dict(gridcolor=GRID, linecolor=LINE, zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor=LINE, zeroline=False),
    colorway=[P1_COLOR, P2_COLOR, "#6366f1", "#f59e0b", "#10b981", "#ef4444"],
))
pio.templates.default = "duo"

CS = dict(
    plot_bgcolor=PAPER, paper_bgcolor=PAPER,
    font_color=INK, font_family="Hanken Grotesk",
)

def mins_to_hm(minutes):
    if pd.isna(minutes):
        return "—"
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m"

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def add_weekday(df: pd.DataFrame):
    """
    Ergänzt 'wochentag' (Mo/Di/Mi …) und 'datum_dt' aus 'datum' (Format TT.MM.JJJJ).
    Gibt (df, n_failed) zurück — n_failed = Anzahl nicht parsebarer Datumswerte,
    damit fehlerhafte Datumsformate nicht still zu NaN werden.
    """
    if df.empty:
        df = df.copy()
        df["wochentag"] = pd.Series(dtype="object")
        df["datum_dt"]  = pd.Series(dtype="datetime64[ns]")
        return df, 0
    dt = pd.to_datetime(df["datum"], format="%d.%m.%Y", errors="coerce")
    df = df.copy()
    df["datum_dt"]  = dt
    df["wochentag"] = dt.dt.weekday.map(WD_MAP)
    n_failed = int(dt.isna().sum())
    return df, n_failed

# ── Produktivitäts-Score ────────────────────────────────────────────────────────
def score_color(s: float) -> str:
    """Farbe passend zum Score-Wert (0–100)."""
    if s >= 65: return "#10b981"   # grün
    if s >= 40: return "#f59e0b"   # amber
    return "#ef4444"               # rot

def score_label(s: float) -> str:
    if s >= 65: return "produktiv"
    if s >= 40: return "ausgewogen"
    return "unproduktiv"

def compute_score(apps_df: pd.DataFrame, tag_df: pd.DataFrame) -> dict | None:
    """
    Berechnet den Produktivitäts-Score für einen Apps-DataFrame.

    Formel:
        raw   = Σ(app_dauer × gewicht) / Σ(app_dauer)   →  [-1, +1]
        score = (raw + 1) / 2 × 100                      →  [0, 100]

    Gewichte (einstellbar im Kategorien-Tab):
        +1 = voll produktiv  ·  0 = neutral  ·  −1 = voll unproduktiv

    Hinweis: Der Score ist subjektiv – er hängt direkt von den gewählten
    Kategorie-Gewichten ab. Gleiche Nutzung, andere Gewichte → anderer Score.
    """
    if apps_df.empty or "gewicht" not in apps_df.columns:
        return None
    app_total = float(apps_df["dauer_minuten"].sum())
    if app_total == 0:
        return None

    weighted = float((apps_df["dauer_minuten"] * apps_df["gewicht"]).sum())
    raw   = weighted / app_total               # in [-1, +1]
    score = (raw + 1) / 2 * 100               # in [0, 100]

    prod   = float(apps_df[apps_df["gewicht"] > 0 ]["dauer_minuten"].sum())
    neut   = float(apps_df[apps_df["gewicht"] == 0]["dauer_minuten"].sum())
    unprod = float(apps_df[apps_df["gewicht"] < 0 ]["dauer_minuten"].sum())

    screen_total = float(tag_df["dauer_minuten"].sum()) if not tag_df.empty else 0.0
    coverage = (app_total / screen_total * 100) if screen_total > 0 else 0.0

    return {
        "score":            round(score, 1),
        "productive_pct":   round(prod   / app_total * 100, 1),
        "neutral_pct":      round(neut   / app_total * 100, 1),
        "unproductive_pct": round(unprod / app_total * 100, 1),
        "coverage_pct":     round(coverage, 1),
    }

def compute_weekly_scores(apps_df: pd.DataFrame, tag_df: pd.DataFrame,
                          weeks_ordered: list) -> pd.DataFrame:
    """Score + Aufschlüsselung pro Woche, in der gegebenen Reihenfolge."""
    rows = []
    for w in weeks_ordered:
        res = compute_score(apps_df[apps_df["woche"] == w],
                            tag_df [tag_df ["woche"] == w])
        if res:
            rows.append({"woche": w, **res})
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def score_card_html(name: str, res: dict, color: str) -> str:
    """Erzeugt eine HTML-Score-Karte mit Gauge-Balken und Aufschlüsselung."""
    s = res["score"]
    prod, neut, unprod, cov = (
        res["productive_pct"], res["neutral_pct"],
        res["unproductive_pct"], res["coverage_pct"],
    )
    label = score_label(s)
    # Dreifarbiger Balken: produktiv | neutral | unproduktiv
    bar = (
        f'<div style="display:flex;height:10px;border-radius:6px;overflow:hidden;margin:14px 0 10px">'
        f'<div style="width:{prod:.1f}%;background:#10b981"></div>'
        f'<div style="width:{neut:.1f}%;background:#cbd5e1"></div>'
        f'<div style="width:{unprod:.1f}%;background:#ef4444"></div>'
        f'</div>'
    )
    return f"""
<div style="background:#fff;border:1px solid #e8e6df;border-top:4px solid {color};
            border-radius:16px;padding:1.6rem 1.8rem;
            box-shadow:0 1px 2px rgba(35,37,47,.04)">
  <div style="color:#9ca3af;font-size:.72rem;text-transform:uppercase;
              letter-spacing:.1em;font-weight:600">{name}</div>
  <div style="display:flex;align-items:baseline;gap:.5rem;margin-top:.3rem">
    <span style="font-family:'Fraunces',serif;font-size:3.4rem;font-weight:600;
                 line-height:1;color:{color}">{s:.0f}</span>
    <span style="color:#9ca3af;font-size:1rem">/ 100</span>
    <span style="margin-left:.4rem;background:{color}18;color:{color};
                 padding:3px 10px;border-radius:20px;font-size:.78rem;
                 font-weight:600">{label}</span>
  </div>
  {bar}
  <div style="display:flex;gap:1.2rem;font-size:.78rem;flex-wrap:wrap">
    <span style="color:#10b981;font-weight:600">▮ {prod:.0f}% produktiv</span>
    <span style="color:#94a3b8;font-weight:600">▮ {neut:.0f}% neutral</span>
    <span style="color:#ef4444;font-weight:600">▮ {unprod:.0f}% unproduktiv</span>
  </div>
  <div style="color:#b6bac4;font-size:.72rem;margin-top:.7rem">
    Basiert auf {cov:.0f}% der erfassten Bildschirmzeit (Top-Apps)
  </div>
</div>"""

# ── Verdrängungsanalyse (Phase 2) ───────────────────────────────────────────────
def critical_r(n: int, alpha: float = 0.05) -> float:
    """Kritischer |r|-Wert für p < alpha (zweiseitig) bei gegebenem n."""
    if n <= 2:
        return np.nan
    df = n - 2
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    return float(t_crit / np.sqrt(t_crit**2 + df))

def sig_star(p) -> str:
    """Signifikanzstern für einen p-Wert."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def compute_corr_matrix(apps_df: pd.DataFrame, min_presence: int = 3):
    """
    Pearson-Korrelationsmatrix der wöchentlichen App-Zeiten.

    Nur Apps, die in mindestens `min_presence` Wochen aktiv (> 0 Minuten) waren.
    n = Anzahl Wochen (für alle Paare gleich — macht p-Werte vergleichbar).
    Gibt (corr_df, pval_df, n) zurück oder (None, None, n) falls zu wenig Daten.
    """
    weeks_all = sorted(apps_df["woche"].unique(),
                       key=lambda x: int(x) if str(x).isdigit() else x)
    n = len(weeks_all)

    # Wöchentliche Summe; fehlende Wochen = 0
    pivot = (apps_df.groupby(["woche", "name"])["dauer_minuten"]
             .sum().unstack(fill_value=0)
             .reindex(index=weeks_all, fill_value=0)
             .astype(float))

    # Nur Apps mit ausreichend Präsenz
    active_weeks = (pivot > 0).sum()
    keep = active_weeks[active_weeks >= min_presence].index.tolist()
    if len(keep) < 2:
        return None, None, n

    pivot = pivot[keep]
    k = len(keep)
    corr_mat = np.full((k, k), np.nan)
    pval_mat = np.full((k, k), np.nan)

    for i in range(k):
        corr_mat[i, i] = 1.0         # Diagonale
        for j in range(i + 1, k):
            xi, xj = pivot.iloc[:, i], pivot.iloc[:, j]
            if xi.std() > 0 and xj.std() > 0:
                r, p = pearsonr(xi, xj)
                corr_mat[i, j] = corr_mat[j, i] = r
                pval_mat[i, j] = pval_mat[j, i] = p

    corr_df = pd.DataFrame(corr_mat, index=keep, columns=keep)
    pval_df  = pd.DataFrame(pval_mat, index=keep, columns=keep)
    return corr_df, pval_df, n

def build_pairs_df(corr_df: pd.DataFrame, pval_df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    Extrahiert Paare aus der oberen Dreiecksmatrix, sortiert nach r.
    Gibt ein DataFrame mit Spalten [label, a, b, r, p, sig, typ] zurück.
    """
    items = corr_df.index.tolist()
    rows = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            r = corr_df.loc[a, b]
            p = pval_df.loc[a, b]
            if pd.isna(r):
                continue
            typ = ("Substitut" if r < -0.3
                   else "Komplement" if r > 0.3
                   else "Kein Muster")
            rows.append({
                "label": f"{a}  ↔  {b}",
                "a": a, "b": b,
                "r": round(float(r), 3),
                "p": float(p) if not pd.isna(p) else np.nan,
                "sig": sig_star(p if not pd.isna(p) else None),
                "typ": typ,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("r")
    # Top N Negative (Substitute) + Top N Positive (Komplemente)
    neg = df[df["r"] < 0].head(top_n)
    pos = df[df["r"] > 0].tail(top_n).iloc[::-1]
    return pd.concat([pos, neg], ignore_index=True)

def corr_heatmap(corr_df: pd.DataFrame, pval_df: pd.DataFrame,
                 title: str = "", height: int = 480) -> go.Figure:
    """
    Plotly-Heatmap der Korrelationsmatrix.
    Zelltext = r-Wert + Signifikanzstern; Hover zeigt r, p, Interpretation.
    """
    labels = corr_df.index.tolist()
    k = len(labels)
    z = corr_df.values
    p = pval_df.values

    # Zelltexte: r-Wert + Stern (kein Stern bei Diagonale)
    cell_text  = [[f"{z[i,j]:.2f}{'' if i==j else sig_star(p[i,j]) if not np.isnan(p[i,j]) else ''}"
                   for j in range(k)] for i in range(k)]
    # Hover-Text
    hover_text = [[
        (f"<b>{labels[i]} ↔ {labels[j]}</b><br>"
         f"r = {z[i,j]:.3f}<br>"
         f"p = {p[i,j]:.4f}  {sig_star(p[i,j])}<br>"
         f"Typ: {'Substitut' if z[i,j]<-0.3 else 'Komplement' if z[i,j]>0.3 else 'Kein Muster'}")
        if i != j else f"<b>{labels[i]}</b><br>Autokorrelation = 1.0"
        for j in range(k)] for i in range(k)]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=cell_text,
        customdata=hover_text,
        hovertemplate="%{customdata}<extra></extra>",
        texttemplate="%{text}",
        textfont=dict(size=10, family="Hanken Grotesk"),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(
            title="r", tickvals=[-1, -0.5, 0, 0.5, 1],
            tickfont=dict(size=10), len=0.8,
        ),
    ))
    fig.update_layout(
        paper_bgcolor=PAPER, font_color=INK, font_family="Hanken Grotesk",
        height=height, margin=dict(t=40 if title else 10, b=10, l=10, r=10),
        xaxis=dict(side="bottom", tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        title=dict(text=title, font=dict(size=13, family="Fraunces")) if title else {},
    )
    return fig

# ── Bootstrap-Konfidenzintervalle (Phase 3) ────────────────────────────────────
def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """
    Vectorisiertes Bootstrap-95%-CI des Mittelwerts. Keine Verteilungsannahme.

    Formel: Resample B-mal mit Zurücklegen → B Mittelwerte → Percentile α/2 & 1−α/2
    Gibt (ci_lo, mean, ci_hi) zurück; bei n < 2 → (nan, mean, nan).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return np.nan, float(values[0]), np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))      # n_boot × n Resample-Indizes
    boot_means = values[idx].mean(axis=1)            # B Mittelwerte in einem Schritt
    return (float(np.percentile(boot_means, 100 * alpha / 2)),
            float(np.mean(values)),
            float(np.percentile(boot_means, 100 * (1 - alpha / 2))))

def weekday_ci(tag_df: pd.DataFrame,
               wd_order: list, n_boot: int = 2000) -> pd.DataFrame:
    """Berechnet Bootstrap-CI des Tagesmittelwerts + Standardabweichung je Wochentag."""
    rows = []
    for i, wd in enumerate(wd_order):
        vals = tag_df[tag_df["wochentag"] == wd]["dauer_minuten"].dropna().values
        if len(vals) == 0:
            continue
        lo, mean, hi = bootstrap_ci(vals, n_boot=n_boot, seed=i)
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append({"wochentag": wd, "mean": round(mean, 1),
                     "std": round(std, 1),
                     "ci_lo": round(lo, 1), "ci_hi": round(hi, 1),
                     "n": len(vals),
                     "ci_width": round(hi - lo, 1) if not np.isnan(hi) else np.nan})
    return pd.DataFrame(rows)

# ── Verteilungsanalyse (Phase 4) ─────────────────────────────────────────────
def detect_outliers(tag_df: pd.DataFrame, person_name: str) -> pd.DataFrame:
    """
    Tukey-IQR-Ausreißertest: Werte außerhalb [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
    Gibt betroffene Tage mit Datum, Wochentag, Wert und IQR-Abstand zurück.
    """
    vals = tag_df["dauer_minuten"].dropna()
    if len(vals) < 4:
        return pd.DataFrame()
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return pd.DataFrame()
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = tag_df[(tag_df["dauer_minuten"] < lower) |
                 (tag_df["dauer_minuten"] > upper)].copy()
    if out.empty:
        return pd.DataFrame()
    out["person"]      = person_name
    out["richtung"]    = out["dauer_minuten"].apply(
        lambda x: "↑ sehr hoch" if x > upper else "↓ sehr niedrig")
    out["IQR-Abstand"] = out["dauer_minuten"].apply(
        lambda x: f"+{(x - upper) / iqr:.1f}×" if x > upper
                  else f"−{(lower - x) / iqr:.1f}×")
    keep = [c for c in ["person","datum","wochentag","woche",
                         "dauer_minuten","richtung","IQR-Abstand"]
            if c in out.columns]
    return out[keep].sort_values("dauer_minuten", ascending=False).reset_index(drop=True)

# ── Trendwende-Erkennung (Phase 5) ────────────────────────────────────────────
def detect_turning_points(weekly_series: pd.Series,
                          smooth_window: int = 3) -> pd.DataFrame:
    """
    Erkennt lokale Hoch- und Tiefpunkte einer Wochen-Zeitreihe.

    Methode (bewusst einfach, n < 20):
      1. Rolling Mean (Fenster = smooth_window, zentriert) — reduziert Rauschen
      2. Erste Differenz der geglätteten Reihe  (= lokale Steigung Δ)
      3. Vorzeichenwechsel von Δ[i] → Δ[i+1] = lokales Extremum

    Warum kein PELT / BINSEG?
      Bei n < 20 Datenpunkten sind formale Change-Point-Verfahren statistisch
      instabil; ihre Konfidenzmaße täuschen eine Präzision vor, die die Daten
      nicht hergeben. Das Rolling-Differenzen-Verfahren ist explizit und ehrlich.
    """
    n = len(weekly_series)
    if n < 4:
        return pd.DataFrame()
    smoothed = weekly_series.rolling(window=smooth_window,
                                     center=True, min_periods=1).mean()
    diff1 = smoothed.diff()
    rows = []
    for i in range(1, n - 1):
        d_before = diff1.iloc[i]       # Steigung zum Punkt hin
        d_after  = diff1.iloc[i + 1]  # Steigung vom Punkt weg
        if pd.isna(d_before) or pd.isna(d_after):
            continue
        if np.sign(d_before) == np.sign(d_after) or d_before == 0 or d_after == 0:
            continue
        typ = "Hochpunkt" if d_before > 0 else "Tiefpunkt"
        rows.append({
            "woche":       str(weekly_series.index[i]),
            "wert":        round(float(weekly_series.iloc[i])),
            "wert_smooth": round(float(smoothed.iloc[i])),
            "typ":         typ,
            "symbol":      "▼" if typ == "Hochpunkt" else "▲",
        })
    return pd.DataFrame(rows)

# ── CSV-Parser (Format bleibt fix, unterstützt bis zu 25 Dateien) ───────────────
def _read_one_csv(file) -> pd.DataFrame:
    """Liest eine einzelne CSV-Datei, bereinigt Spalten und Typen."""
    df = pd.read_csv(file, skip_blank_lines=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Leerzeilen (entstehen als NaN-Zeilen trotz skip_blank_lines in manchen Pandas-Versionen)
    df = df.dropna(how="all")
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()
    df["dauer_minuten"] = pd.to_numeric(df["dauer_minuten"], errors="coerce").fillna(0)
    return df

def parse_csv(files, person_name: str) -> dict:
    """
    Liest eine oder mehrere CSV-Dateien einer Person und gibt ein einheitliches
    Daten-Dict zurück. `files` kann ein einzelnes UploadedFile oder eine Liste sein.
    """
    if not isinstance(files, (list, tuple)):
        files = [files]
    frames = [_read_one_csv(f) for f in files if f is not None]
    if not frames:
        raise ValueError("Keine gültigen Dateien übergeben.")
    df = pd.concat(frames, ignore_index=True)
    df["person"] = person_name
    df["woche"]  = df["woche"].astype(str)
    # Duplikate entfernen (falls eine Woche mehrfach hochgeladen)
    df = df.drop_duplicates(subset=["woche", "datum", "daten_kategorie", "name"])
    return {
        "tag":   df[df["daten_kategorie"] == "tag_gesamt"].copy(),
        "woche": df[df["daten_kategorie"] == "woche_gesamt"].copy(),
        "apps":  df[df["daten_kategorie"] == "top_app"].copy(),
        "raw":   df,
    }

# ── Demo-Daten · 3 Tage/Woche = Montag, Dienstag, Mittwoch ───────────────────────
@st.cache_data
def make_demo(person: str, seed: int, start_kw: int = 6, n_weeks: int = 13,
              iso_year: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    app_pool = ["WhatsApp", "Instagram", "YouTube", "TikTok", "Spotify",
                "Chrome", "Maps", "Snapchat", "Netflix", "LinkedIn", "Slack",
                "Notion", "Duolingo", "Reddit"]
    rows_tag, rows_woche, rows_apps = [], [], []
    for i in range(n_weeks):
        kw = start_kw + i
        mon = date.fromisocalendar(iso_year, kw, 1)   # Montag
        days = [mon, date.fromisocalendar(iso_year, kw, 2),  # Dienstag
                date.fromisocalendar(iso_year, kw, 3)]       # Mittwoch
        dstr = [f"{d.day:02d}.{d.month:02d}.{d.year}" for d in days]
        week_total = int(rng.normal(1600, 280))
        rows_woche.append({
            "woche": str(kw), "datum": f"{mon.day:02d}.{mon.month:02d}.-{days[-1].day:02d}.{days[-1].month:02d}.",
            "daten_kategorie": "woche_gesamt", "name": "gesamt",
            "dauer_minuten": week_total, "person": person,
        })
        for ds in dstr:
            daily = max(60, int(rng.normal(week_total / 3, 45)))
            rows_tag.append({
                "woche": str(kw), "datum": ds, "daten_kategorie": "tag_gesamt",
                "name": "gesamt", "dauer_minuten": daily, "person": person,
            })
            top = rng.choice(app_pool, 5, replace=False)
            portions = rng.dirichlet(np.ones(5)) * daily * 0.7
            for app, t in zip(top, portions):
                rows_apps.append({
                    "woche": str(kw), "datum": ds, "daten_kategorie": "top_app",
                    "name": app, "dauer_minuten": int(t), "person": person,
                })
    all_rows = rows_tag + rows_woche + rows_apps
    return {
        "tag":   pd.DataFrame(rows_tag),
        "woche": pd.DataFrame(rows_woche),
        "apps":  pd.DataFrame(rows_apps),
        "raw":   pd.DataFrame(all_rows),
    }

# ── Kategorie-State ──────────────────────────────────────────────────────────────
def init_category_state():
    st.session_state.setdefault("app_categories", {})   # {app_lower: kategorie}
    for cat in CATEGORIES:
        st.session_state.setdefault(f"w_{cat}", DEFAULT_WEIGHTS[cat])

def sync_categories(real_apps_lower):
    """Für jede (noch unbekannte) App eine Default-Kategorie aus dem Preset setzen."""
    cats = st.session_state["app_categories"]
    for a in real_apps_lower:
        if a not in cats:
            cats[a] = APP_PRESET.get(a, "Sonstiges")

def current_weights():
    return {cat: float(st.session_state[f"w_{cat}"]) for cat in CATEGORIES}

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📥 Daten")
    st.markdown('<span class="tag-p1">Person 1</span>', unsafe_allow_html=True)
    name1  = st.text_input("Name", value="Sarah", key="n1")
    files1 = st.file_uploader("CSV hochladen (alle Wochen auf einmal)",
                               type="csv", key="f1", accept_multiple_files=True)
    if files1:
        st.caption(f"✓ {len(files1)} Datei{'en' if len(files1)>1 else ''} geladen")
    st.markdown("---")
    st.markdown('<span class="tag-p2">Person 2</span>', unsafe_allow_html=True)
    name2  = st.text_input("Name", value="Bella", key="n2")
    files2 = st.file_uploader("CSV hochladen (alle Wochen auf einmal)",
                               type="csv", key="f2", accept_multiple_files=True)
    if files2:
        st.caption(f"✓ {len(files2)} Datei{'en' if len(files2)>1 else ''} geladen")
    st.markdown("---")
    st.markdown("### ⚙️ Optionen")
    pseudonym = st.toggle("App-Namen pseudonymisieren", False)
    show_raw  = st.toggle("Rohdaten anzeigen", False)
    st.markdown("---")
    st.caption("ScreenTime Duo · Semesterprojekt")

# ── Daten laden ──────────────────────────────────────────────────────────────────
def load(files, name, seed) -> dict:
    """
    Lädt Daten aus CSV-Dateien; fällt auf Demo zurück, wenn keine Datei da ist
    oder das Parsen scheitert. Hängt Metadaten an (_source, _error, _date_failures),
    damit der Aufrufer Probleme sichtbar melden kann — keine stillen Fehler.
    """
    source, error_msg = "upload", None
    try:
        if files:
            d = parse_csv(files, name)
        else:
            d = make_demo(name, seed)
            source = "demo"
    except Exception as e:
        error_msg = str(e)
        d = make_demo(name, seed)
        source = "demo_fallback"
    d["tag"],  nf_tag  = add_weekday(d["tag"])
    d["apps"], nf_apps = add_weekday(d["apps"])
    # App-Namen für Anzeige normalisieren (bekannte Apps korrekt, Rest Title-Case)
    d["apps"]["name"] = d["apps"]["name"].apply(
        lambda n: DISPLAY_NAMES.get(n.lower(), n if any(c.isupper() for c in n) else n.title())
    )
    d["_source"]         = source
    d["_error"]          = error_msg
    d["_date_failures"]  = nf_tag + nf_apps
    d["_dated_rows"]     = int(len(d["tag"]) + len(d["apps"]))
    return d

init_category_state()

d1 = load(files1, name1, 42)
d2 = load(files2, name2, 99)

# Probleme beim Laden sichtbar melden (kein stiller Demo-Fallback, keine stillen NaN)
for d, nm in [(d1, name1), (d2, name2)]:
    if d["_source"] == "demo_fallback":
        st.error(
            f"⚠️ **{nm}:** Die hochgeladene CSV konnte nicht gelesen werden "
            f"(`{d['_error']}`). Es werden ersatzweise **Demo-Daten** angezeigt — "
            f"prüfe Spalten (woche, datum, daten_kategorie, name, dauer_minuten) und Format."
        )
    elif d["_source"] == "upload" and d["_date_failures"] > 0:
        st.warning(
            f"⚠️ **{nm}:** {d['_date_failures']} von {d['_dated_rows']} Datumswerten "
            f"konnten nicht als **TT.MM.JJJJ** gelesen werden (z. B. 16.03.2026). "
            f"Für diese Zeilen bleibt die Wochentags-Analyse (Statistik-Tab) leer. "
            f"Bitte Datumsformat in der CSV prüfen."
        )

# Kategorien synchronisieren (auf Basis der ECHTEN App-Namen beider Personen)
real_apps_lower = sorted(set(d1["apps"]["name"].str.lower()) | set(d2["apps"]["name"].str.lower()))
sync_categories(real_apps_lower)
weights = current_weights()

# Apps mit Kategorie & Gewicht annotieren (vor evtl. Pseudonymisierung)
for d in (d1, d2):
    lname = d["apps"]["name"].str.lower()
    d["apps"]["kategorie"] = lname.map(st.session_state["app_categories"]).fillna("Sonstiges")
    d["apps"]["gewicht"]   = d["apps"]["kategorie"].map(weights).astype(float)

# Anzeige-Namen pro echter App (für Editor & ggf. Pseudonymisierung)
display_name = {}
for d in (d1, d2):
    for n in d["apps"]["name"].unique():
        display_name.setdefault(n.lower(), n)

if pseudonym:
    pseudo = {a: f"App {i+1}" for i, a in enumerate(real_apps_lower)}
    for d in (d1, d2):
        d["apps"]["name"] = d["apps"]["name"].str.lower().map(pseudo)

tag_all   = pd.concat([d1["tag"],   d2["tag"]],   ignore_index=True)
woche_all = pd.concat([d1["woche"], d2["woche"]], ignore_index=True)
apps_all  = pd.concat([d1["apps"],  d2["apps"]],  ignore_index=True)
weeks     = sorted(tag_all["woche"].unique(), key=lambda x: int(x) if str(x).isdigit() else x)

# ── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>ScreenTime Duo</h1>
  <p>Bildschirmzeit als Geschichte erzählt &middot; <b>{name1}</b> &amp; <b>{name2}</b></p>
</div>
""", unsafe_allow_html=True)

if not files1 and not files2:
    st.info("Demo-Daten aktiv (13 Wochen, je Mo/Di/Mi). Lade links deine CSV-Dateien hoch — "
            "alle Wochen auf einmal auswählbar.", icon="🎲")

# ── KPIs ─────────────────────────────────────────────────────────────────────────
tot1   = d1["woche"]["dauer_minuten"].sum()
tot2   = d2["woche"]["dauer_minuten"].sum()
avg1   = d1["tag"]["dauer_minuten"].mean()
avg2   = d2["tag"]["dauer_minuten"].mean()
shared = set(d1["apps"]["name"].str.lower()) & set(d2["apps"]["name"].str.lower())

c1, c2, c3, c4 = st.columns(4)
for col, label, val, sub in [
    (c1, f"Gesamt {name1}", mins_to_hm(tot1), f"{len(d1['woche'])} Wochen"),
    (c2, f"Gesamt {name2}", mins_to_hm(tot2), f"{len(d2['woche'])} Wochen"),
    (c3, "Tagesschnitt",    f"{mins_to_hm(avg1)} / {mins_to_hm(avg2)}", f"{name1} vs. {name2}"),
    (c4, "Gemeinsame Apps", str(len(shared)), "beide genutzt"),
]:
    col.markdown(f"""<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{val}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab_prod, tab_verd, tab_dist, tab4, tab_cat, tab_life, tab5 = st.tabs([
    "📈 Zeitverlauf", "📱 App-Analyse", "⚔️ Vergleich",
    "🎯 Produktivität", "🔗 Verdrängung", "📉 Verteilung",
    "📊 Statistik", "🏷️ Kategorien", "⏳ Lebenszeit", "🗂️ Daten",
])

# ════════ TAB 1 · Zeitverlauf ════════════════════════════════════════════════════
with tab1:
    section("Wöchentliche Gesamtbildschirmzeit")
    if len(weeks) >= 2:
        wr = st.select_slider("Wochenbereich (KW)", options=weeks,
                              value=(weeks[0], weeks[-1]), key="wr1")
        fw = [w for w in weeks if weeks.index(wr[0]) <= weeks.index(w) <= weeks.index(wr[1])]
    else:
        fw = weeks

    wf = woche_all[woche_all["woche"].isin(fw)]
    fig = px.line(wf, x="woche", y="dauer_minuten", color="person",
                  color_discrete_sequence=[P1_COLOR, P2_COLOR],
                  markers=True, labels={"dauer_minuten": "Minuten / Woche", "woche": "KW"})
    fig.update_traces(line_width=2.6, marker_size=8)
    fig.update_layout(**CS, height=360, hovermode="x unified", legend_title_text="",
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    section("Tagesdetail (Mo / Di / Mi)")
    sel_w = st.selectbox("Woche (KW)", weeks, key="tw")
    daily_sel = tag_all[tag_all["woche"] == sel_w].copy()
    daily_sel["wochentag"] = pd.Categorical(daily_sel["wochentag"], categories=WD_ORDER, ordered=True)
    daily_sel = daily_sel.sort_values("wochentag")
    fig2 = px.bar(daily_sel, x="wochentag", y="dauer_minuten", color="person", barmode="group",
                  color_discrete_sequence=[P1_COLOR, P2_COLOR],
                  labels={"dauer_minuten": "Minuten", "wochentag": "Wochentag"})
    fig2.update_layout(**CS, height=320, margin=dict(t=20, b=20), legend_title_text="")
    st.plotly_chart(fig2, width="stretch")

    # ── Trendwende-Analyse ──────────────────────────────────────────────────────
    section("Trendwende-Analyse (lokale Hoch- & Tiefpunkte)")

    # Methodik-Info
    smooth_w = st.slider("Glättungsfenster (Wochen)", 2, 4, 3, key="tp_window",
                         help="Größeres Fenster → weniger, robustere Trendwenden")
    st.markdown(
        f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:'
        f'4px solid #10b981;border-radius:12px;padding:.85rem 1.2rem;'
        f'margin-bottom:1rem;font-size:.84rem;color:#065f46">'
        f'<b>🔍 Methodik (Rolling-Differenzen, Fenster = {smooth_w} Wochen)</b>'
        f'&ensp;·&ensp;Wochenwerte glätten → <b>erste</b> Differenz → Vorzeichenwechsel = '
        f'lokales Maximum/Minimum (Trendwende).<br>'
        f'<span style="color:#047857">Begriffs-Hinweis:</span> Das sind '
        f'<b>Trendwenden</b> (die Reihe kehrt ihre Richtung um), nicht „Wendepunkte" '
        f'im mathematischen Sinn (Krümmungswechsel = Vorzeichenwechsel der <em>zweiten</em> '
        f'Ableitung). Bei nur ~13 Wochen sind echte Wendepunkte instabil und schwerer '
        f'interpretierbar als Trendwenden. Kein PELT/BINSEG, weil formale Change-Point-'
        f'Verfahren bei n &lt; 20 unzuverlässig sind.</div>',
        unsafe_allow_html=True,
    )

    # Wöchentliche Gesamtzeiten als geordnete Series (eine pro Person)
    def make_weekly_series(d: dict) -> pd.Series:
        s = d["woche"].groupby("woche")["dauer_minuten"].sum()
        return s.reindex(weeks).fillna(np.nan)

    ws1 = make_weekly_series(d1)
    ws2 = make_weekly_series(d2)

    tp1 = detect_turning_points(ws1.dropna(), smooth_window=smooth_w)
    tp2 = detect_turning_points(ws2.dropna(), smooth_window=smooth_w)

    # ── Chart: Rohdaten + geglättet + Wendepunkt-Marker ─────────────────────────
    fig_tp = go.Figure()
    configs = [
        (ws1, tp1, name1, P1_COLOR, "rgba(23,179,166,0.18)",  -55),
        (ws2, tp2, name2, P2_COLOR, "rgba(236,108,166,0.18)", +55),
    ]
    for ws, tp, nm, color, fill_rgba, ay_base in configs:
        ws_clean = ws.dropna()
        if ws_clean.empty:
            continue
        weeks_nm = ws_clean.index.tolist()
        smoothed  = ws_clean.rolling(window=smooth_w, center=True, min_periods=1).mean()

        # Rohdaten-Linie
        fig_tp.add_trace(go.Scatter(
            x=weeks_nm, y=ws_clean.values,
            name=nm, mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color,
                        line=dict(color="white", width=1.2)),
            hovertemplate=f"<b>{nm}</b> KW%{{x}}<br>%{{y:.0f}} min<extra></extra>",
        ))
        # Geglättete Linie
        fig_tp.add_trace(go.Scatter(
            x=weeks_nm, y=smoothed.values,
            name=f"{nm} (geglättet)", mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            opacity=0.55, showlegend=False,
            hovertemplate=f"<b>{nm} glatt</b> KW%{{x}}<br>%{{y:.0f}} min<extra></extra>",
        ))
        if tp.empty:
            continue
        # Stern-Marker an den Trendwenden (auf dem Originalwert)
        fig_tp.add_trace(go.Scatter(
            x=tp["woche"], y=tp["wert"],
            mode="markers",
            name=f"{nm} Trendwenden",
            marker=dict(symbol="star", size=16, color=color,
                        line=dict(color="white", width=2)),
            customdata=tp["typ"].values,
            hovertemplate=(f"<b>{nm} — %{{customdata}}</b><br>"
                           "KW %{x}: %{y:.0f} min<extra></extra>"),
            showlegend=True,
        ))
        # Annotationen mit Pfeil
        for _, row in tp.iterrows():
            ay = ay_base   # P1 oben, P2 unten — vermeidet Überlappung
            fig_tp.add_annotation(
                x=str(row["woche"]), y=row["wert"],
                text=(f"<b>{row['symbol']} KW {row['woche']}</b>"
                      f"<br>{row['typ']}"),
                showarrow=True, arrowhead=2, arrowsize=1.1,
                arrowwidth=2, arrowcolor=color,
                ax=0, ay=ay,
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor=color, borderwidth=1.5, borderpad=5,
                font=dict(size=10, family="Hanken Grotesk", color=INK),
            )

    fig_tp.update_layout(
        **CS, height=460, hovermode="x unified",
        xaxis=dict(title="Kalenderwoche", categoryorder="array",
                   categoryarray=weeks, gridcolor=GRID, linecolor=LINE),
        yaxis=dict(title="Minuten / Woche", gridcolor=GRID,
                   linecolor=LINE, zeroline=False),
        legend_title_text="",
        margin=dict(t=20, b=20, l=10, r=10),
    )
    st.plotly_chart(fig_tp, width="stretch")
    st.caption(
        "Gestrichelte Linie = geglättet (Rolling Mean). "
        "Sterne ★ markieren lokale Extrema (▼ Hochpunkt, ▲ Tiefpunkt). "
        "P1-Annotationen erscheinen oberhalb, P2-Annotationen unterhalb des Datenpunkts."
    )

    # ── Trendwende-Tabelle ──────────────────────────────────────────────────────
    if not tp1.empty or not tp2.empty:
        section("Trendwenden im Detail")
        tbl_rows = []
        for tp, nm in [(tp1, name1), (tp2, name2)]:
            for _, row in tp.iterrows():
                tbl_rows.append({
                    "Person":   nm,
                    "KW":       row["woche"],
                    "Typ":      f"{row['symbol']} {row['typ']}",
                    "Originalwert": mins_to_hm(row["wert"]),
                    "Geglättet":    mins_to_hm(row["wert_smooth"]),
                })
        tbl_tp = pd.DataFrame(tbl_rows)
        st.dataframe(tbl_tp, width="stretch", hide_index=True)

        # Kurzinterpretation
        n1, n2 = len(tp1), len(tp2)
        more = name1 if n1 > n2 else (name2 if n2 > n1 else None)
        if more:
            st.caption(
                f"{more} zeigt mehr Trendwenden ({max(n1,n2)} vs {min(n1,n2)}) — "
                f"das kann auf einen weniger stabilen Nutzungs-Rhythmus hindeuten, "
                f"ist aber bei kleinem n mit Vorsicht zu interpretieren."
            )
    else:
        st.info(
            "Keine Trendwenden gefunden. Mögliche Ursachen: zu wenige Wochen, "
            "monotoner Verlauf, oder das Glättungsfenster ist zu groß."
        )

# ════════ TAB 2 · App-Analyse ════════════════════════════════════════════════════
with tab2:
    section("Top-Apps gesamt pro Person")
    col_a, col_b = st.columns(2)
    for col, d, name in [(col_a, d1, name1), (col_b, d2, name2)]:
        with col:
            top = (d["apps"].groupby(["name", "kategorie"])["dauer_minuten"].sum()
                   .reset_index().nlargest(6, "dauer_minuten"))
            top["hm"] = top["dauer_minuten"].apply(mins_to_hm)
            fig = px.bar(top, x="dauer_minuten", y="name", orientation="h",
                         color="kategorie", color_discrete_map=CAT_COLORS,
                         category_orders={"kategorie": CATEGORIES}, text="hm",
                         labels={"dauer_minuten": "Minuten gesamt", "name": "", "kategorie": "Kategorie"})
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_layout(**CS, height=330, margin=dict(t=34, b=10),
                              legend_title_text="", showlegend=False,
                              yaxis=dict(categoryorder="total ascending", gridcolor=GRID, linecolor=LINE),
                              title=dict(text=f"Top-Apps — {name}",
                                         font=dict(color=INK, size=14, family="Fraunces")))
            st.plotly_chart(fig, width="stretch")

    section("App-Zeitverlauf")
    sel_p = st.radio("Person", [name1, name2], horizontal=True, key="ap")
    d_sel = d1 if sel_p == name1 else d2
    top5  = d_sel["apps"].groupby("name")["dauer_minuten"].sum().nlargest(5).index.tolist()
    s_apps = st.multiselect("Apps auswählen", top5, default=top5)

    if len(weeks) >= 2:
        wr2 = st.select_slider("Wochenbereich", options=weeks,
                               value=(weeks[0], weeks[-1]), key="wr2")
        fw2 = [w for w in weeks if weeks.index(wr2[0]) <= weeks.index(w) <= weeks.index(wr2[1])]
    else:
        fw2 = weeks

    aw = (d_sel["apps"][d_sel["apps"]["woche"].isin(fw2) & d_sel["apps"]["name"].isin(s_apps)]
          .groupby(["woche", "name"])["dauer_minuten"].sum().reset_index())
    fig3 = px.line(aw, x="woche", y="dauer_minuten", color="name",
                   markers=True, labels={"dauer_minuten": "Minuten", "woche": "KW"})
    fig3.update_layout(**CS, height=360, margin=dict(t=20), legend_title_text="")
    st.plotly_chart(fig3, width="stretch")

# ════════ TAB 3 · Vergleich ══════════════════════════════════════════════════════
with tab3:
    section("Gemeinsame Apps — Radardiagramm")
    a1 = d1["apps"].copy(); a1["name_l"] = a1["name"].str.lower()
    a2 = d2["apps"].copy(); a2["name_l"] = a2["name"].str.lower()
    shared_l = sorted(set(a1["name_l"]) & set(a2["name_l"]))[:7]

    if len(shared_l) >= 3:
        r1 = a1.groupby("name_l")["dauer_minuten"].sum()
        r2 = a2.groupby("name_l")["dauer_minuten"].sum()
        labels = [s.capitalize() for s in shared_l]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=[r1.get(a, 0) for a in shared_l], theta=labels,
            fill='toself', name=name1, line_color=P1_COLOR, fillcolor="rgba(23,179,166,0.18)"))
        fig_r.add_trace(go.Scatterpolar(r=[r2.get(a, 0) for a in shared_l], theta=labels,
            fill='toself', name=name2, line_color=P2_COLOR, fillcolor="rgba(236,108,166,0.18)"))
        fig_r.update_layout(
            polar=dict(bgcolor="#fbfaf7",
                       radialaxis=dict(visible=True, gridcolor=GRID, color=MUTED),
                       angularaxis=dict(gridcolor=GRID)),
            paper_bgcolor=PAPER, font_color=INK, font_family="Hanken Grotesk",
            legend_orientation="h", height=420, margin=dict(t=30))
        st.plotly_chart(fig_r, width="stretch")
    else:
        st.info("Zu wenige gemeinsame Apps — weitere Wochen hinzufügen.")

    section(f"Wöchentliche Differenz ({name1} minus {name2})")
    w1s = d1["woche"].groupby("woche")["dauer_minuten"].sum()
    w2s = d2["woche"].groupby("woche")["dauer_minuten"].sum()
    cw  = sorted(set(w1s.index) & set(w2s.index), key=lambda x: int(x) if str(x).isdigit() else x)
    if cw:
        diff = pd.DataFrame({"woche": cw, "diff": [w1s[w] - w2s[w] for w in cw]})
        diff["farbe"] = diff["diff"].apply(lambda x: P1_COLOR if x >= 0 else P2_COLOR)
        diff["label"] = diff.apply(
            lambda r: f"{name1} +{abs(int(r['diff']))}" if r["diff"] >= 0
                      else f"{name2} +{abs(int(r['diff']))}", axis=1)
        fig4 = px.bar(diff, x="woche", y="diff", color="farbe",
                      color_discrete_map="identity", text="label",
                      labels={"diff": "Differenz in Minuten", "woche": "KW"})
        fig4.add_hline(y=0, line_color="#d1d5db", line_dash="dot")
        fig4.update_traces(textposition="outside", textfont_size=10, marker_line_width=0)
        fig4.update_layout(**CS, showlegend=False, height=340, margin=dict(t=20))
        st.plotly_chart(fig4, width="stretch")

    section("Korrelation der Wochengesamtzeiten")
    if len(cw) > 1:
        mc = pd.DataFrame({name1: [w1s[w] for w in cw],
                           name2: [w2s[w] for w in cw], "woche": cw})
        corr = mc[name1].corr(mc[name2])
        # Trendlinie nur, wenn statsmodels verfügbar ist (sonst Scatter ohne OLS)
        try:
            import statsmodels.api as _sm  # noqa: F401
            trend = "ols"
        except ImportError:
            trend = None
        fig5 = px.scatter(mc, x=name1, y=name2, text="woche", trendline=trend,
                          color_discrete_sequence=[P1_COLOR],
                          labels={name1: f"Min/Woche {name1}", name2: f"Min/Woche {name2}"})
        fig5.update_traces(marker_size=11, textposition="top center", textfont_size=9,
                           selector=dict(mode="markers+text"))
        fig5.update_layout(**CS, height=340, margin=dict(t=40),
                           title=dict(text=f"Korrelationskoeffizient r = {corr:.2f}",
                                      font=dict(color=INK, size=14, family="Fraunces")))
        st.plotly_chart(fig5, width="stretch")
        cap = ("Hinweis: Bei nur wenigen Wochen ist r mit großer Unsicherheit behaftet — "
               "Korrelation bedeutet außerdem nicht Kausalität.")
        if trend is None:
            cap += " (Trendlinie ausgeblendet — statsmodels nicht installiert.)"
        st.caption(cap)
    else:
        st.info("Mindestens 2 gemeinsame Wochen erforderlich.")

# ════════ TAB · Produktivität ════════════════════════════════════════════════════
with tab_prod:

    # ── Disclaimer ───────────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;
            border-radius:12px;padding:1rem 1.3rem;margin-bottom:1.2rem">
  <div style="font-weight:700;color:#92400e;margin-bottom:.25rem">⚖️ Dieser Score ist subjektiv</div>
  <div style="color:#78350f;font-size:.88rem;line-height:1.55">
    Der Produktivitäts-Score berechnet sich aus den Gewichten, die du im
    <b>Kategorien-Tab</b> eingestellt hast. Gleiche Nutzung mit anderen Gewichten
    ergibt einen anderen Score. Er misst also nicht <em>objektive Produktivität</em>,
    sondern wie gut die Bildschirmzeit mit deinen eigenen Prioritäten übereinstimmt.
    Die Formel: <code>Score = (Σ App-Zeit × Gewicht / Σ App-Zeit + 1) / 2 × 100</code>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Score-Berechnung ─────────────────────────────────────────────────────────
    res1 = compute_score(d1["apps"], d1["tag"])
    res2 = compute_score(d2["apps"], d2["tag"])

    # ── Score-Karten ─────────────────────────────────────────────────────────────
    section("Gesamt-Score")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if res1:
            st.markdown(score_card_html(name1, res1, score_color(res1["score"])),
                        unsafe_allow_html=True)
        else:
            st.info("Keine App-Daten für Score-Berechnung.")
    with col_s2:
        if res2:
            st.markdown(score_card_html(name2, res2, score_color(res2["score"])),
                        unsafe_allow_html=True)
        else:
            st.info("Keine App-Daten für Score-Berechnung.")

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # ── Wöchentlicher Trend ───────────────────────────────────────────────────────
    section("Wöchentlicher Score-Trend")
    ws1 = compute_weekly_scores(d1["apps"], d1["tag"], weeks)
    ws2 = compute_weekly_scores(d2["apps"], d2["tag"], weeks)

    if not ws1.empty or not ws2.empty:
        trend_frames = []
        if not ws1.empty:
            ws1["person"] = name1
            trend_frames.append(ws1)
        if not ws2.empty:
            ws2["person"] = name2
            trend_frames.append(ws2)
        trend = pd.concat(trend_frames, ignore_index=True)

        fig_trend = px.line(
            trend, x="woche", y="score", color="person",
            color_discrete_map={name1: P1_COLOR, name2: P2_COLOR},
            markers=True,
            labels={"score": "Score (0 – 100)", "woche": "KW", "person": ""},
        )
        fig_trend.update_traces(line_width=2.6, marker_size=9)
        # Referenzlinien für die drei Zonen
        fig_trend.add_hrect(y0=65, y1=100, fillcolor="#10b981", opacity=.05,
                            line_width=0, annotation_text="produktiv ≥ 65",
                            annotation_position="top right",
                            annotation_font=dict(size=10, color="#10b981"))
        fig_trend.add_hrect(y0=40, y1=65, fillcolor="#f59e0b", opacity=.05,
                            line_width=0, annotation_text="ausgewogen 40–64",
                            annotation_position="top right",
                            annotation_font=dict(size=10, color="#b45309"))
        fig_trend.add_hrect(y0=0, y1=40, fillcolor="#ef4444", opacity=.04,
                            line_width=0, annotation_text="unproduktiv < 40",
                            annotation_position="bottom right",
                            annotation_font=dict(size=10, color="#ef4444"))
        fig_trend.add_hline(y=50, line_dash="dot", line_color="#d1d5db",
                            annotation_text="Neutral (50)", annotation_font_size=10)
        fig_trend.update_layout(
            **CS, height=400, hovermode="x unified",
            yaxis=dict(range=[0, 105], gridcolor=GRID, linecolor=LINE, zeroline=False),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_trend, width="stretch")

        # Score-Delta als Annotation unter dem Chart
        cw_both = sorted(set(ws1["woche"].tolist()) & set(ws2["woche"].tolist()),
                         key=lambda x: int(x) if str(x).isdigit() else x)
        if cw_both and not ws1.empty and not ws2.empty:
            s1_map = ws1.set_index("woche")["score"]
            s2_map = ws2.set_index("woche")["score"]
            deltas  = [s1_map[w] - s2_map[w] for w in cw_both]
            avg_delta = sum(deltas) / len(deltas)
            winner = name1 if avg_delta > 0 else name2
            st.caption(
                f"Ø Score-Differenz über alle Wochen: {abs(avg_delta):.1f} Punkte "
                f"zugunsten von **{winner}**."
            )
    else:
        st.info("Noch keine Daten für den Score-Trend vorhanden.")

    # ── Kategorie-Aufschlüsselung nach Woche ─────────────────────────────────────
    section("Kategorien-Aufschlüsselung nach Woche")
    sel_prod_p = st.radio("Person", [name1, name2], horizontal=True, key="prod_p")
    d_prod = d1 if sel_prod_p == name1 else d2

    cat_woche = (d_prod["apps"]
                 .groupby(["woche", "kategorie"])["dauer_minuten"]
                 .sum().reset_index())
    cat_woche["woche"] = pd.Categorical(
        cat_woche["woche"], categories=weeks, ordered=True)
    cat_woche = cat_woche.sort_values("woche")

    if not cat_woche.empty:
        fig_cw = px.bar(
            cat_woche, x="woche", y="dauer_minuten", color="kategorie",
            barmode="stack", color_discrete_map=CAT_COLORS,
            category_orders={"kategorie": CATEGORIES},
            labels={"dauer_minuten": "Minuten", "woche": "KW", "kategorie": "Kategorie"},
        )
        fig_cw.update_layout(**CS, height=360, legend_title_text="",
                             margin=dict(t=10, b=10))
        st.plotly_chart(fig_cw, width="stretch")

    # ── Produktivste vs. unproduktivste Woche ────────────────────────────────────
    section("Beste & schlechteste Woche")
    score_pair = [(ws1, name1, P1_COLOR), (ws2, name2, P2_COLOR)]
    bp_cols = st.columns(2)
    for col, (ws, nm, clr) in zip(bp_cols, score_pair):
        with col:
            if ws.empty:
                col.info(f"Keine Daten für {nm}.")
                continue
            best  = ws.loc[ws["score"].idxmax()]
            worst = ws.loc[ws["score"].idxmin()]
            col.markdown(f"""
<div style="background:#fff;border:1px solid #e8e6df;border-radius:14px;
            padding:1.2rem 1.5rem;box-shadow:0 1px 2px rgba(35,37,47,.04)">
  <div style="color:#9ca3af;font-size:.72rem;font-weight:600;
              text-transform:uppercase;letter-spacing:.1em">{nm}</div>
  <div style="margin-top:.8rem">
    <span style="font-size:.78rem;color:#10b981;font-weight:700">▲ BESTE WOCHE</span>
    <div style="font-family:'Fraunces',serif;font-size:1.5rem;
                color:#10b981;font-weight:600">KW {best['woche']} &nbsp;
      <span style="font-size:1rem">{best['score']:.0f}/100</span></div>
  </div>
  <div style="margin-top:.8rem">
    <span style="font-size:.78rem;color:#ef4444;font-weight:700">▼ SCHLECHTESTE WOCHE</span>
    <div style="font-family:'Fraunces',serif;font-size:1.5rem;
                color:#ef4444;font-weight:600">KW {worst['woche']} &nbsp;
      <span style="font-size:1rem">{worst['score']:.0f}/100</span></div>
  </div>
</div>""", unsafe_allow_html=True)

# ════════ TAB · Verdrängung ══════════════════════════════════════════════════════
with tab_verd:

    # ── Statistik-Disclaimer ──────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#fff7ed;border:1px solid #fed7aa;border-left:4px solid #f97316;
            border-radius:12px;padding:1rem 1.3rem;margin-bottom:1.2rem">
  <div style="font-weight:700;color:#9a3412;margin-bottom:.3rem">
    📐 Statistische Einschränkungen — bitte lesen
  </div>
  <div style="color:#7c2d12;font-size:.875rem;line-height:1.6">
    <b>Korrelation ≠ Kausalität.</b> Selbst eine starke negative Korrelation zwischen
    App A und App B bedeutet nicht, dass A die Nutzung von B verdrängt — beide könnten
    von einem dritten Faktor abhängen (z. B. Wochentag, Prüfungsphase, Wetter).<br><br>
    <b>Kleines n = breite Konfidenzintervalle.</b> Mit 10–15 Wochen als Datenpunkte sind
    fast alle Korrelationen statistisch nicht signifikant (p ≥ 0.05). Das ist keine
    Fehlfunktion, sondern ehrliche Statistik. Signifikante Werte (mit ✶) sind trotzdem
    mit Vorsicht zu interpretieren. Für belastbare Ergebnisse wären ≥ 30 Wochen nötig.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Person & Einstellungen ────────────────────────────────────────────────────
    col_ps, col_mp = st.columns([3, 1])
    with col_ps:
        sel_verd = st.radio("Person", [name1, name2], horizontal=True, key="verd_p")
    with col_mp:
        min_pres = st.number_input("Mindest-Wochen für App", 2, 10, 3, 1,
                                   key="verd_min",
                                   help="App muss in mind. N Wochen aktiv gewesen sein.")

    dv = d1 if sel_verd == name1 else d2
    clr_verd = P1_COLOR if sel_verd == name1 else P2_COLOR

    # ══════════════════════════════════════════════════════════════════════════════
    # APP-KORRELATIONEN
    # ══════════════════════════════════════════════════════════════════════════════
    section("App-zu-App-Korrelationen")
    corr_df, pval_df, n_weeks = compute_corr_matrix(dv["apps"], min_presence=min_pres)

    if corr_df is None:
        st.info(f"Zu wenige Apps oder Wochen für die Analyse "
                f"(Mindest-Wochen: {min_pres}). Passe den Regler oben an.")
    else:
        k = len(corr_df)
        r_crit = critical_r(n_weeks)

        # Statistischer Kontext-Banner
        st.markdown(
            f'<div style="background:#f6f5f1;border-radius:10px;padding:.75rem 1.1rem;'
            f'margin-bottom:1rem;font-size:.84rem;color:#4b5563">'
            f'<b>n = {n_weeks} Wochen</b> &nbsp;·&nbsp; '
            f'{k} Apps (mind. {min_pres}×  aktiv) &nbsp;·&nbsp; '
            f'Signifikanzschwelle: |r| &gt; <b>{r_crit:.2f}</b> für p &lt; 0.05 '
            f'&nbsp;·&nbsp; <span style="color:#6b7280">Sterne: * p&lt;0.05 &nbsp;'
            f'** p&lt;0.01 &nbsp; *** p&lt;0.001 &nbsp; ns = nicht signifikant</span>'
            f'</div>', unsafe_allow_html=True)

        # Heatmap
        fig_hm = corr_heatmap(corr_df, pval_df, height=max(380, k * 46))
        st.plotly_chart(fig_hm, width="stretch")

        # Stärkste Paare
        pairs_df = build_pairs_df(corr_df, pval_df, top_n=6)

        if not pairs_df.empty:
            section("Stärkste Substitute & Komplemente")

            # Farbiger Bar-Chart der Paare
            pairs_df["farbe"] = pairs_df["r"].apply(
                lambda r: "#10b981" if r > 0 else "#ef4444")
            pairs_df["hover"] = pairs_df.apply(
                lambda row: f"r = {row['r']:.3f}  {row['sig']}<br>"
                            f"p = {row['p']:.4f}<br>{row['typ']}", axis=1)

            fig_pairs = go.Figure()
            for _, row in pairs_df.iterrows():
                fig_pairs.add_trace(go.Bar(
                    x=[row["r"]], y=[row["label"]],
                    orientation="h",
                    marker_color=row["farbe"],
                    hovertemplate=f"<b>{row['label']}</b><br>{row['hover']}<extra></extra>",
                    showlegend=False,
                    text=[f"  {row['r']:+.2f} {row['sig']}"],
                    textposition="outside" if abs(row["r"]) < 0.85 else "inside",
                    textfont=dict(size=11),
                ))
            fig_pairs.add_vline(x=0,  line_color="#d1d5db", line_width=1.5)
            fig_pairs.add_vline(x= r_crit, line_color="#10b981", line_dash="dot",
                                line_width=1,
                                annotation_text=f"p<0.05 ({r_crit:.2f})",
                                annotation_font_size=9, annotation_font_color="#10b981",
                                annotation_position="top right")
            fig_pairs.add_vline(x=-r_crit, line_color="#ef4444", line_dash="dot",
                                line_width=1,
                                annotation_text=f"p<0.05 (−{r_crit:.2f})",
                                annotation_font_size=9, annotation_font_color="#ef4444",
                                annotation_position="top left")
            fig_pairs.update_layout(
                **CS, height=max(280, len(pairs_df) * 46 + 60),
                xaxis=dict(range=[-1.15, 1.15], title="Pearson r",
                           gridcolor=GRID, linecolor=LINE, zeroline=False),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=LINE,
                           categoryorder="array",
                           categoryarray=pairs_df["label"].tolist()),
                margin=dict(t=20, b=40, l=10, r=80),
                bargap=0.35,
            )
            st.plotly_chart(fig_pairs, width="stretch")

            # Detail-Tabelle im Expander
            with st.expander("📋 Vollständige Tabelle aller Paare"):
                all_pairs = build_pairs_df(corr_df, pval_df, top_n=99)
                if not all_pairs.empty:
                    display_tbl = all_pairs[["label", "r", "p", "sig", "typ"]].copy()
                    display_tbl.columns = ["App-Paar", "r", "p-Wert", "Sig.", "Typ"]
                    display_tbl["p-Wert"] = display_tbl["p-Wert"].apply(
                        lambda x: f"{x:.4f}" if not pd.isna(x) else "—")
                    display_tbl["r"] = display_tbl["r"].apply(lambda x: f"{x:+.3f}")
                    st.dataframe(display_tbl, width="stretch", hide_index=True)
                st.caption(
                    "Nur Paare mit |r| ≥ 0.3 als Substitut / Komplement klassifiziert. "
                    "Alle Werte sind Pearson-Koeffizienten auf Wochenbasis.")
        else:
            st.info("Keine Paare mit ausreichend Daten gefunden.")

    # ══════════════════════════════════════════════════════════════════════════════
    # KATEGORIE-KORRELATIONEN
    # ══════════════════════════════════════════════════════════════════════════════
    section("Kategorie-zu-Kategorie-Korrelationen")
    st.markdown('<p class="lead">Verdrängt Unterhaltung das Lernen? '
                'Gehen Kommunikation und Soziale Medien Hand in Hand? '
                'Diese Matrix aggregiert die App-Zeiten nach Kategorie.</p>',
                unsafe_allow_html=True)

    # Kategorie-Zeit pro Woche berechnen
    cat_apps = dv["apps"].copy()
    cat_apps["name"] = cat_apps["kategorie"]   # Kategorie als "App-Name" nutzen
    corr_cat, pval_cat, n_cat = compute_corr_matrix(cat_apps, min_presence=1)

    if corr_cat is None:
        st.info("Zu wenige Kategorien oder Wochen für die Kategorie-Analyse.")
    else:
        k_cat = len(corr_cat)
        # Kategorie-Heatmap immer mit Texten (nur 6×6)
        fig_cat_hm = corr_heatmap(corr_cat, pval_cat,
                                  height=max(340, k_cat * 52))
        st.plotly_chart(fig_cat_hm, width="stretch")

        # Stärkste Kategorie-Paare
        cat_pairs = build_pairs_df(corr_cat, pval_cat, top_n=4)
        if not cat_pairs.empty:
            col_pos, col_neg = st.columns(2)
            pos_pairs = cat_pairs[cat_pairs["r"] > 0]
            neg_pairs = cat_pairs[cat_pairs["r"] < 0]

            with col_pos:
                st.markdown(
                    '<div style="font-weight:700;color:#10b981;margin-bottom:.4rem">'
                    '🟢 Komplementäre Kategorien</div>', unsafe_allow_html=True)
                for _, row in pos_pairs.iterrows():
                    st.markdown(
                        f'<div style="background:#f0fdf4;border-radius:8px;'
                        f'padding:.6rem .9rem;margin-bottom:.4rem;font-size:.85rem">'
                        f'<b>{row["a"]}</b> & <b>{row["b"]}</b><br>'
                        f'<span style="color:#10b981">r = {row["r"]:+.2f}</span> &nbsp; '
                        f'<span style="color:#9ca3af">{row["sig"]}</span></div>',
                        unsafe_allow_html=True)

            with col_neg:
                st.markdown(
                    '<div style="font-weight:700;color:#ef4444;margin-bottom:.4rem">'
                    '🔴 Substituierende Kategorien</div>', unsafe_allow_html=True)
                for _, row in neg_pairs.iterrows():
                    st.markdown(
                        f'<div style="background:#fef2f2;border-radius:8px;'
                        f'padding:.6rem .9rem;margin-bottom:.4rem;font-size:.85rem">'
                        f'<b>{row["a"]}</b> vs. <b>{row["b"]}</b><br>'
                        f'<span style="color:#ef4444">r = {row["r"]:+.2f}</span> &nbsp; '
                        f'<span style="color:#9ca3af">{row["sig"]}</span></div>',
                        unsafe_allow_html=True)

    # ── Abschluss-Note ────────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
            padding:.9rem 1.1rem;margin-top:1.2rem;font-size:.82rem;color:#64748b">
  <b>Methodik:</b> Pearson-Korrelation der wöchentlichen App-Zeiten (Σ Minuten/Woche).
  Wochen ohne Nutzung einer App zählen als 0. Apps mit konstantem Wert über alle Wochen
  (Std. = 0) werden ausgeschlossen. Für belastbare Inferenz werden ≥ 30 unabhängige
  Beobachtungen empfohlen; diese Analyse ist explorativer Natur.
</div>""", unsafe_allow_html=True)

# ════════ TAB · Verteilung ═══════════════════════════════════════════════════════
with tab_dist:

    P1_FILL = "rgba(23,179,166,0.22)"
    P2_FILL = "rgba(236,108,166,0.22)"

    # Tatsächliche Stichprobengröße pro Person (Anzahl Tagesmesswerte)
    n_dist1 = int(d1["tag"]["dauer_minuten"].dropna().shape[0])
    n_dist2 = int(d2["tag"]["dauer_minuten"].dropna().shape[0])
    n_dist_str = f"{n_dist1}" if n_dist1 == n_dist2 else f"{n_dist1} bzw. {n_dist2}"

    # ── Disclaimer ───────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #3b82f6;
            border-radius:12px;padding:1rem 1.3rem;margin-bottom:1.1rem">
  <div style="font-weight:700;color:#1e3a8a;margin-bottom:.3rem">📐 Statistische Einordnung</div>
  <div style="color:#1e40af;font-size:.875rem;line-height:1.6">
    <b>Shapiro-Wilk bei kleinem n:</b> Mit {n_dist_str} Datenpunkten ist die Teststärke gering.
    p &gt; 0.05 bedeutet <em>nicht</em>, dass die Daten normalverteilt sind — nur, dass dieser Test
    es nicht widerlegen kann. p &lt; 0.05 bedeutet <em>nicht</em>, dass die Verteilung „falsch" ist.
    Der <b>Q-Q-Plot</b> ist oft informativer als der Testwert allein.
    <b>IQR-Ausreißer</b> sind statistische Extremwerte, keine Datenfehler — sie können
    echte Ereignisse (Prüfungsphase, Urlaub, Erkrankung) widerspiegeln.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Violin + Ausreißer-Marker ────────────────────────────────────────────────
    section("Tageszeit-Verteilung (Violin-Plot)")
    fig_viol = go.Figure()
    outlier_store = {}

    for d_p, nm, color, fill in [
        (d1, name1, P1_COLOR, P1_FILL),
        (d2, name2, P2_COLOR, P2_FILL),
    ]:
        vals = d_p["tag"]["dauer_minuten"].dropna().values
        if len(vals) == 0:
            continue
        fig_viol.add_trace(go.Violin(
            x=[nm] * len(vals), y=vals, name=nm,
            line_color=color, fillcolor=fill,
            box_visible=True, meanline_visible=True,
            points="all", jitter=0.38, pointpos=0,
            marker=dict(size=5, color=color, opacity=0.45),
            hoveron="points+kde",
            hovertemplate=f"<b>{nm}</b><br>%{{y:.0f}} min<extra></extra>",
        ))
        out_df = detect_outliers(d_p["tag"], nm)
        outlier_store[nm] = out_df
        if not out_df.empty:
            fig_viol.add_trace(go.Scatter(
                x=[nm] * len(out_df), y=out_df["dauer_minuten"],
                mode="markers", showlegend=False,
                marker=dict(symbol="x-thin-open", size=16, color="#ef4444",
                            line=dict(width=3, color="#ef4444")),
                text=out_df["datum"] if "datum" in out_df.columns else None,
                hovertemplate="<b>Ausreißer</b><br>%{text}<br>%{y:.0f} min<extra></extra>",
            ))

    fig_viol.update_layout(
        **CS, violinmode="group", height=440,
        yaxis=dict(title="Minuten / Tag", gridcolor=GRID, linecolor=LINE, zeroline=False),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=LINE),
        legend_title_text="", margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_viol, width="stretch")
    st.caption("Rote ✕ = Ausreißer nach Tukey-IQR (1.5 × IQR-Regel). "
               "Hover auf Punkte für Datum-Details.")

    # ── Histogramm + KDE (Expander) ──────────────────────────────────────────────
    with st.expander("📊 Histogramm + KDE-Kurve"):
        fig_hist = go.Figure()
        for d_p, nm, color, fill in [
            (d1, name1, P1_COLOR, P1_FILL),
            (d2, name2, P2_COLOR, P2_FILL),
        ]:
            vals = d_p["tag"]["dauer_minuten"].dropna().values
            if len(vals) < 3:
                continue
            # Histogramm (normiert auf Dichte)
            fig_hist.add_trace(go.Histogram(
                x=vals, name=f"{nm}", histnorm="probability density",
                marker_color=fill, nbinsx=12, showlegend=True,
                hovertemplate="[%{x}] Dichte: %{y:.4f}<extra></extra>",
            ))
            # KDE-Kurve (nur bei vorhandener Streuung — sonst singuläre Matrix)
            if np.std(vals) > 0:
                kde_fn = gaussian_kde(vals)
                x_grid = np.linspace(vals.min() * 0.85, vals.max() * 1.1, 300)
                fig_hist.add_trace(go.Scatter(
                    x=x_grid, y=kde_fn(x_grid),
                    name=f"{nm} KDE", mode="lines",
                    line=dict(color=color, width=2.5),
                    hovertemplate="~%{x:.0f} min<br>Dichte: %{y:.5f}<extra></extra>",
                ))
        fig_hist.update_traces(selector=dict(type="histogram"), opacity=0.55)
        fig_hist.update_layout(
            **CS, barmode="overlay", height=350,
            xaxis=dict(title="Minuten / Tag", gridcolor=GRID, linecolor=LINE),
            yaxis=dict(title="Dichte", gridcolor=GRID, linecolor=LINE, zeroline=False),
            legend_title_text="", margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_hist, width="stretch")

    # ── Q-Q-Plots + Shapiro-Wilk ─────────────────────────────────────────────────
    section("Normalitätsprüfung: Q-Q-Plot & Shapiro-Wilk")
    col_qq1, col_qq2 = st.columns(2)

    for col, d_p, nm, color in [
        (col_qq1, d1, name1, P1_COLOR),
        (col_qq2, d2, name2, P2_COLOR),
    ]:
        vals = d_p["tag"]["dauer_minuten"].dropna().values
        with col:
            if len(vals) < 4:
                st.info(f"Zu wenige Daten für {nm} (mind. 4 Tageswerte nötig).")
                continue
            if np.std(vals) == 0:
                st.info(f"{nm}: alle Tageswerte identisch — keine Verteilungsanalyse möglich.")
                continue

            # Q-Q Plot
            (osm, osr), (slope, intercept, r_qq) = scipy_probplot(vals, dist="norm", fit=True)
            x_ref = np.array([float(osm.min()), float(osm.max())])

            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=x_ref, y=slope * x_ref + intercept,
                mode="lines", name="Normalverteilung",
                line=dict(color="#cbd5e1", dash="dash", width=2),
                showlegend=True,
            ))
            fig_qq.add_trace(go.Scatter(
                x=osm, y=osr, mode="markers",
                name=nm,
                marker=dict(color=color, size=7, opacity=0.85,
                            line=dict(color="#fff", width=1)),
                hovertemplate="Theor. Q: %{x:.2f}<br>Emp. Q: %{y:.0f} min<extra></extra>",
            ))
            fig_qq.update_layout(
                **CS, height=300,
                xaxis=dict(title="Theoretische Quantile", gridcolor=GRID, linecolor=LINE),
                yaxis=dict(title="Stichproben-Quantile [min]", gridcolor=GRID, linecolor=LINE, zeroline=False),
                title=dict(text=f"Q-Q — {nm}  (r = {r_qq:.3f})",
                           font=dict(family="Fraunces", size=13, color=INK)),
                showlegend=False, margin=dict(t=50, b=40, l=10, r=10),
            )
            st.plotly_chart(fig_qq, width="stretch")

            # Shapiro-Wilk
            W, p_sw = shapiro(vals)
            significant = p_sw < 0.05
            sw_color  = "#ef4444" if significant else "#10b981"
            sw_label  = "⚠️ Abweichung (p < 0.05)" if significant else "✓ Kein Befund (p ≥ 0.05)"
            st.markdown(
                f'<div style="background:{sw_color}10;border:1px solid {sw_color}40;'
                f'border-radius:10px;padding:.7rem 1rem;font-size:.84rem">'
                f'<b>Shapiro-Wilk:</b> &nbsp; W = {W:.4f} &nbsp;·&nbsp; '
                f'p = {p_sw:.4f} &nbsp;'
                f'<span style="background:{sw_color};color:#fff;padding:2px 9px;'
                f'border-radius:12px;font-size:.76rem;font-weight:600">{sw_label}</span>'
                f'<br><span style="color:#9ca3af;font-size:.76rem;margin-top:.25rem;display:block">'
                f'n = {len(vals)} · Q-Q-Korrelation r = {r_qq:.3f}</span></div>',
                unsafe_allow_html=True,
            )
    st.caption(
        "Interpretation: Punkte nahe der gestrichelten Diagonalen → normalverteilungsähnlich. "
        "Starke S-Kurve → schwere Ränder. Systematische Biegung → Schiefe. "
        f"Bei n = {n_dist_str} sind leichte Abweichungen normal und nicht besorgniserregend."
    )

    # ── Ausreißer-Tabelle ────────────────────────────────────────────────────────
    section("Ausreißer-Tage (Tukey-IQR 1.5 ×)")
    all_out = pd.concat(list(outlier_store.values()), ignore_index=True)
    if all_out.empty:
        st.success("✓ Keine Ausreißer gefunden — beide Verteilungen liegen im IQR-Bereich.")
    else:
        # Tabelle mit lesbaren Spaltenbezeichnungen
        tbl_out = all_out.copy()
        if "dauer_minuten" in tbl_out.columns:
            tbl_out["Bildschirmzeit"] = tbl_out["dauer_minuten"].apply(mins_to_hm)
        rename = {"person": "Person", "datum": "Datum", "wochentag": "Wochentag",
                  "woche": "KW", "richtung": "Typ"}
        tbl_out = tbl_out.rename(columns=rename)
        show_cols = [c for c in ["Person","Datum","Wochentag","KW","Bildschirmzeit",
                                  "Typ","IQR-Abstand"] if c in tbl_out.columns]
        st.dataframe(tbl_out[show_cols], width="stretch", hide_index=True)
        st.caption(
            "IQR-Abstand: Wie weit der Wert außerhalb der Tukey-Fence liegt. "
            "+2.0× bedeutet: 2 IQR über der oberen Grenze. "
            "Ausreißer sind statistische Extremwerte, keine Datenfehler."
        )

    # ── Kategorie-Violinen (Expander) ────────────────────────────────────────────
    with st.expander("🏷️ Verteilung pro Kategorie"):
        st.markdown('<p class="lead">Wie ist die Zeit pro Kategorie verteilt? '
                    'Ist Lernen sehr gleichmäßig, während Unterhaltung stark streut? '
                    'Basiert auf Top-App-Daten (nur erfasste Apps).</p>',
                    unsafe_allow_html=True)
        sel_dist_p = st.radio("Person", [name1, name2],
                               horizontal=True, key="dist_cat_p")
        d_dist = d1 if sel_dist_p == name1 else d2

        # Tageszeit pro Kategorie aggregieren
        cat_day = (d_dist["apps"]
                   .groupby(["datum","kategorie"])["dauer_minuten"]
                   .sum().reset_index())
        cat_day = cat_day[cat_day["dauer_minuten"] > 0]   # nur Tage mit Nutzung

        if cat_day.empty:
            st.info("Keine Kategorie-Daten vorhanden.")
        else:
            fig_catv = go.Figure()
            for cat in CATEGORIES:
                sub = cat_day[cat_day["kategorie"] == cat]["dauer_minuten"].values
                if len(sub) < 2:
                    continue
                c = CAT_COLORS[cat]
                rgba_fill = (f"rgba({int(c[1:3],16)},"
                             f"{int(c[3:5],16)},"
                             f"{int(c[5:7],16)},0.2)")
                fig_catv.add_trace(go.Violin(
                    x=[cat] * len(sub), y=sub, name=cat,
                    line_color=c, fillcolor=rgba_fill,
                    box_visible=True, meanline_visible=True,
                    points="all", jitter=0.4, pointpos=0,
                    marker=dict(size=5, color=c, opacity=0.5),
                    showlegend=False,
                    hovertemplate=f"<b>{cat}</b><br>%{{y:.0f}} min<extra></extra>",
                ))
            fig_catv.update_layout(
                **CS, height=400, violinmode="group",
                xaxis=dict(title="Kategorie", gridcolor="rgba(0,0,0,0)", linecolor=LINE),
                yaxis=dict(title="Minuten / Tag", gridcolor=GRID, linecolor=LINE, zeroline=False),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_catv, width="stretch")

            # KDE-Kurven pro Kategorie (eine Dichtekurve je Kategorie)
            st.markdown('<p class="lead">KDE-Dichtekurven je Kategorie — '
                        'zeigt die Form der Verteilung (schmal = gleichmäßig, '
                        'breit/mehrgipflig = stark schwankend).</p>',
                        unsafe_allow_html=True)
            fig_catk = go.Figure()
            any_kde = False
            for cat in CATEGORIES:
                sub = cat_day[cat_day["kategorie"] == cat]["dauer_minuten"].values
                # KDE braucht ≥ 2 Punkte UND Streuung > 0 (sonst singuläre Matrix)
                if len(sub) < 3 or np.std(sub) == 0:
                    continue
                c = CAT_COLORS[cat]
                kde_fn = gaussian_kde(sub)
                x_grid = np.linspace(max(0, sub.min() * 0.8), sub.max() * 1.15, 250)
                fig_catk.add_trace(go.Scatter(
                    x=x_grid, y=kde_fn(x_grid), mode="lines",
                    name=cat, line=dict(color=c, width=2.4),
                    hovertemplate=f"<b>{cat}</b><br>~%{{x:.0f}} min<br>Dichte: %{{y:.5f}}<extra></extra>",
                ))
                any_kde = True
            if any_kde:
                fig_catk.update_layout(
                    **CS, height=320, legend_title_text="",
                    xaxis=dict(title="Minuten / Tag", gridcolor=GRID, linecolor=LINE),
                    yaxis=dict(title="Dichte", gridcolor=GRID, linecolor=LINE, zeroline=False),
                    margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig_catk, width="stretch")
            else:
                st.caption("Zu wenige Datenpunkte pro Kategorie für KDE-Kurven.")
    section("Statistische Kennzahlen (Tageszeiten)")
    for d, name, color in [(d1, name1, P1_COLOR), (d2, name2, P2_COLOR)]:
        vals = d["tag"]["dauer_minuten"]
        stats = {"Mittelwert": vals.mean(), "Median": vals.median(),
                 "Std.-Abw.": vals.std(), "Min": vals.min(), "Max": vals.max()}
        st.markdown(f'<div class="section-title" style="color:{color};border-color:{color}33">{name}</div>',
                    unsafe_allow_html=True)
        cols = st.columns(5)
        for col, (k, v) in zip(cols, stats.items()):
            col.markdown(f"""<div class="metric-card">
              <div class="metric-label">{k}</div>
              <div class="metric-value" style="font-size:1.35rem;color:{color}">{mins_to_hm(v)}</div>
            </div>""", unsafe_allow_html=True)

    section("Boxplot — Tageszeiten pro Woche")
    fig6 = px.box(tag_all, x="woche", y="dauer_minuten", color="person",
                  color_discrete_sequence=[P1_COLOR, P2_COLOR],
                  labels={"dauer_minuten": "Minuten / Tag", "woche": "KW"})
    fig6.update_layout(**CS, height=380, margin=dict(t=20), legend_title_text="")
    st.plotly_chart(fig6, width="stretch")

    section("Heatmap — Tageszeit je Woche und Wochentag")
    hp  = st.radio("Person", [name1, name2], horizontal=True, key="hp")
    dfh = d1["tag"] if hp == name1 else d2["tag"]
    try:
        pivot = dfh.pivot_table(index="woche", columns="wochentag",
                                values="dauer_minuten", aggfunc="mean")
        pivot = pivot.reindex(columns=[w for w in WD_ORDER if w in pivot.columns])
        fig7 = px.imshow(pivot, color_continuous_scale="Teal",
                         labels={"color": "Min / Tag"}, aspect="auto")
        fig7.update_layout(paper_bgcolor=PAPER, font_color=INK,
                           font_family="Hanken Grotesk", height=320, margin=dict(t=20))
        st.plotly_chart(fig7, width="stretch")
    except Exception:
        st.info("Nicht genug Datenpunkte für die Heatmap.")

    # ── Bootstrap-Konfidenzintervalle pro Wochentag ────────────────────────────
    section("Konfidenzintervalle pro Wochentag (Bootstrap 95%-CI)")

    wd1 = weekday_ci(d1["tag"], WD_ORDER)
    wd2 = weekday_ci(d2["tag"], WD_ORDER)
    wd_present = [w for w in WD_ORDER if
                  w in wd1["wochentag"].values or w in wd2["wochentag"].values]

    # Kontextinfo: n pro Wochentag
    n_info_parts = []
    for wd in wd_present:
        n1 = int(wd1[wd1["wochentag"]==wd]["n"].values[0]) if wd in wd1["wochentag"].values else 0
        n2 = int(wd2[wd2["wochentag"]==wd]["n"].values[0]) if wd in wd2["wochentag"].values else 0
        n_info_parts.append(f"<b>{wd}</b>: {name1} n={n1}, {name2} n={n2}")

    st.markdown(
        f'<div style="background:#f6f5f1;border-radius:10px;padding:.75rem 1.1rem;'
        f'margin-bottom:.9rem;font-size:.84rem;color:#4b5563">'
        f'📅 Stichproben pro Wochentag: {" &nbsp;·&nbsp; ".join(n_info_parts)} &nbsp;·&nbsp; '
        f'Bootstrap-Resamplings: 2 000 &nbsp;·&nbsp; '
        f'<span style="color:#9ca3af">Breitere Bänder = mehr Streuung oder weniger Wochen</span>'
        f'</div>', unsafe_allow_html=True)

    # ── Hauptchart: Linie + CI-Band ────────────────────────────────────────────
    if not wd1.empty or not wd2.empty:
        fig_ci = go.Figure()
        configs = [
            (name1, wd1, P1_COLOR, "rgba(23,179,166,0.15)"),
            (name2, wd2, P2_COLOR, "rgba(236,108,166,0.15)"),
        ]
        for person, wd_df, color, fill_rgba in configs:
            if wd_df.empty:
                continue
            has_ci = not wd_df["ci_lo"].isna().all()

            if has_ci:
                # CI-Band als Polygon (vorwärts ci_hi, rückwärts ci_lo)
                x_band = wd_df["wochentag"].tolist() + wd_df["wochentag"].tolist()[::-1]
                y_band = wd_df["ci_hi"].tolist() + wd_df["ci_lo"].tolist()[::-1]
                fig_ci.add_trace(go.Scatter(
                    x=x_band, y=y_band,
                    fill="toself", fillcolor=fill_rgba,
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))

            # Mittellinie
            cd = wd_df[["n", "ci_lo", "ci_hi"]].values
            fig_ci.add_trace(go.Scatter(
                x=wd_df["wochentag"], y=wd_df["mean"],
                mode="lines+markers",
                name=person,
                line=dict(color=color, width=2.6),
                marker=dict(size=10, color=color,
                            line=dict(color="#fff", width=2)),
                customdata=cd,
                hovertemplate=(
                    f"<b>{person}</b> — %{{x}}<br>"
                    "Ø <b>%{y:.0f} min</b><br>"
                    "95%-CI: [%{customdata[1]:.0f}–%{customdata[2]:.0f}] min<br>"
                    "n = %{customdata[0]} Wochen<extra></extra>"
                ),
            ))

        fig_ci.update_layout(
            **CS, height=380, hovermode="x unified", legend_title_text="",
            xaxis=dict(title="Wochentag", categoryorder="array",
                       categoryarray=wd_present, gridcolor=GRID, linecolor=LINE),
            yaxis=dict(title="Minuten / Tag", gridcolor=GRID,
                       linecolor=LINE, zeroline=False),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_ci, width="stretch")
        st.caption(
            "Gefärbte Bänder = 95%-Bootstrap-Konfidenzintervall. "
            "Wo sich zwei Bänder überlappen, ist kein statistisch gesicherter "
            "Unterschied zwischen den Personen erkennbar — bei kleinem n ist das normal."
        )

        # ── Vergleichstabelle ──────────────────────────────────────────────────
        section("Tagesvergleich — Zahlen")
        tbl_rows = []
        for wd in wd_present:
            r1 = wd1[wd1["wochentag"]==wd].iloc[0] if wd in wd1["wochentag"].values else None
            r2 = wd2[wd2["wochentag"]==wd].iloc[0] if wd in wd2["wochentag"].values else None

            def fmt(row):
                if row is None: return "—"
                if np.isnan(row["ci_lo"]): return f"{row['mean']:.0f} min (n={row['n']})"
                return f"{row['mean']:.0f} min  [{row['ci_lo']:.0f}–{row['ci_hi']:.0f}]"

            def fmt_std(row):
                if row is None: return "—"
                return f"±{row['std']:.0f} min"

            if r1 is not None and r2 is not None and not np.isnan(r1["ci_lo"]) and not np.isnan(r2["ci_lo"]):
                overlap = max(r1["ci_lo"], r2["ci_lo"]) < min(r1["ci_hi"], r2["ci_hi"])
                diff = r1["mean"] - r2["mean"]
                diff_str = f"{diff:+.0f} min"
                ov_str = "✓ ja" if overlap else "✗ nein"
            else:
                diff_str = "—"
                ov_str = "—"

            tbl_rows.append({
                "Wochentag": wd,
                f"{name1}  Ø [95%-CI]": fmt(r1),
                f"{name1}  Streuung (σ)": fmt_std(r1),
                f"{name2}  Ø [95%-CI]": fmt(r2),
                f"{name2}  Streuung (σ)": fmt_std(r2),
                f"Differenz ({name1}−{name2})": diff_str,
                "CIs überschneiden sich": ov_str,
            })

        tbl = pd.DataFrame(tbl_rows)
        st.dataframe(tbl, width="stretch", hide_index=True)
        st.caption(
            "Ø [95%-CI] = Mittelwert mit Bootstrap-Konfidenzintervall. "
            "Streuung (σ) = Standardabweichung der Tageswerte (wie stark schwankt der "
            "Wochentag von Woche zu Woche). Differenz = Mittelwert P1 − P2. "
            "CIs überschneiden sich = statistisch kein gesicherter Unterschied "
            "(erwartbar bei kleinem n). Nicht-Überschneidung ≠ kausaler Unterschied."
        )

        # ── Klassische Balkendarstellung (Expander) ────────────────────────────
        with st.expander("📊 Klassische Balkendarstellung mit Fehlerbalken"):
            fig_bar = go.Figure()
            for person, wd_df, color in [(name1, wd1, P1_COLOR), (name2, wd2, P2_COLOR)]:
                if wd_df.empty: continue
                hi_arr = (wd_df["ci_hi"] - wd_df["mean"]).clip(lower=0).values
                lo_arr = (wd_df["mean"] - wd_df["ci_lo"]).clip(lower=0).values
                has_ci = not wd_df["ci_lo"].isna().all()
                fig_bar.add_trace(go.Bar(
                    name=person, x=wd_df["wochentag"], y=wd_df["mean"],
                    marker_color=color, marker_line_width=0,
                    error_y=dict(
                        type="data", symmetric=False,
                        array=hi_arr if has_ci else None,
                        arrayminus=lo_arr if has_ci else None,
                        visible=has_ci,
                        color="rgba(35,37,47,0.45)",
                        thickness=2.5, width=9,
                    ) if has_ci else dict(visible=False),
                    customdata=wd_df[["n", "ci_lo", "ci_hi"]].values,
                    hovertemplate=(
                        f"<b>{person}</b> — %{{x}}<br>"
                        "Ø <b>%{y:.0f} min</b><br>"
                        "95%-CI: [%{customdata[1]:.0f}–%{customdata[2]:.0f}] min<br>"
                        "n = %{customdata[0]} Wochen<extra></extra>"
                    ),
                ))
            fig_bar.update_layout(
                **CS, barmode="group", height=360,
                xaxis=dict(title="Wochentag", categoryorder="array",
                           categoryarray=wd_present, gridcolor=GRID, linecolor=LINE),
                yaxis=dict(title="Minuten / Tag", gridcolor=GRID,
                           linecolor=LINE, zeroline=False),
                legend_title_text="", margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_bar, width="stretch")
    else:
        st.info("Keine Tagesdaten mit Wochentag-Information vorhanden.")

# ════════ TAB · Kategorien ═══════════════════════════════════════════════════════
with tab_cat:
    section("App → Kategorie zuordnen")
    st.markdown('<p class="lead">Jede App bekommt eine funktionale Kategorie. Häufige Apps sind '
                'vorausgefüllt — du kannst alles per Dropdown überschreiben. Die Zuordnung gilt für '
                'beide Personen und wird für die ganze Sitzung gespeichert.</p>', unsafe_allow_html=True)

    cat_df = pd.DataFrame({
        "App": [display_name[a] for a in real_apps_lower],
        "_key": real_apps_lower,
        "Kategorie": [st.session_state["app_categories"][a] for a in real_apps_lower],
    })
    edited = st.data_editor(
        cat_df[["App", "Kategorie"]],
        column_config={
            "App": st.column_config.TextColumn("App", disabled=True, width="medium"),
            "Kategorie": st.column_config.SelectboxColumn("Kategorie", options=CATEGORIES, required=True),
        },
        hide_index=True, width="stretch", key="cat_editor",
    )
    changed = False
    for app_disp, kat in zip(cat_df["_key"], edited["Kategorie"]):
        if st.session_state["app_categories"].get(app_disp) != kat:
            st.session_state["app_categories"][app_disp] = kat
            changed = True
    if changed:
        st.rerun()

    section("Produktivitäts-Gewicht pro Kategorie")
    st.markdown('<p class="lead">Lege fest, wie „wertvoll" eine Kategorie zählt: −1 (klar unproduktiv) '
                'bis +1 (klar produktiv), 0 ist neutral. Diese Gewichte sind bewusst <b>subjektiv</b> — '
                'sie bilden später die Grundlage für den Produktivitäts-Score (Phase 1).</p>',
                unsafe_allow_html=True)
    wcols = st.columns(3)
    for i, cat in enumerate(CATEGORIES):
        with wcols[i % 3]:
            st.markdown(f'<span class="cat-chip" style="background:{CAT_COLORS[cat]}22;'
                        f'color:{CAT_COLORS[cat]}">{cat}</span>', unsafe_allow_html=True)
            st.slider(cat, -1.0, 1.0, step=0.1, key=f"w_{cat}", label_visibility="collapsed")

    section("Wie verteilt sich die Zeit auf die Kategorien?")
    st.markdown('<p class="lead">Vorschau: So setzt sich die Gesamtzeit beider Personen aktuell '
                'nach Kategorie zusammen.</p>', unsafe_allow_html=True)
    cat_rows = []
    for d, name in [(d1, name1), (d2, name2)]:
        g = d["apps"].groupby("kategorie")["dauer_minuten"].sum()
        for cat in CATEGORIES:
            cat_rows.append({"person": name, "kategorie": cat, "minuten": float(g.get(cat, 0))})
    cat_summary = pd.DataFrame(cat_rows)
    if cat_summary["minuten"].sum() > 0:
        fig_cat = px.bar(cat_summary, x="minuten", y="person", color="kategorie",
                         orientation="h", barmode="stack",
                         color_discrete_map=CAT_COLORS, category_orders={"kategorie": CATEGORIES},
                         labels={"minuten": "Minuten gesamt", "person": "", "kategorie": "Kategorie"})
        fig_cat.update_layout(**CS, height=240, margin=dict(t=10, b=10),
                              legend_title_text="", legend_orientation="h")
        st.plotly_chart(fig_cat, width="stretch")
    else:
        st.info("Noch keine App-Daten kategorisiert.")

# ════════ TAB · Lebenszeit ═══════════════════════════════════════════════════════
with tab_life:

    # ── Eröffnung ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero" style="margin-bottom:1.6rem">
  <h2 style="font-size:2rem;margin-bottom:.4rem">Zeit in Perspektive</h2>
  <p style="max-width:680px">
    Diese Seite rechnet eure Bildschirmzeit in andere Einheiten um —
    nicht um zu urteilen, sondern um Größenordnungen greifbar zu machen.
    <b>Alle Annahmen sind einstellbar.</b> Die App berechnet transparent
    aus euren Eingaben; keine der Zahlen ist eine gesicherte Statistik.
  </p>
</div>""", unsafe_allow_html=True)

    # ── Hochrechnung ─────────────────────────────────────────────────────────────
    section("Hochrechnung auf ein Jahr")

    hr_col1, hr_col2 = st.columns(2)
    with hr_col1:
        weeks_per_year = st.slider("Aktive Wochen pro Jahr (Hochrechnung)", 30, 52, 52, 1,
                                   key="life_wpy",
                                   help="52 = ganzes Jahr. Weniger, um z. B. Ferien auszunehmen.")
    with hr_col2:
        waking_h = st.slider("Wache Stunden pro Tag", 12, 18, 16, 1, key="life_waking",
                             help="Für die Umrechnung in wache Lebenstage.")

    def annual_stats(d: dict) -> dict:
        wdf = d["woche"]
        if wdf.empty:
            return {"avg_w": 0, "n_w": 0, "annual_min": 0, "annual_h": 0}
        avg_w = wdf["dauer_minuten"].mean()
        n_w   = len(wdf)
        ann   = avg_w * weeks_per_year
        return {"avg_w": avg_w, "n_w": int(n_w), "annual_min": ann, "annual_h": ann / 60}

    a1 = annual_stats(d1)
    a2 = annual_stats(d2)

    c1, c2, c3 = st.columns(3)
    for col, a, nm, color in [
        (c1, a1, name1, P1_COLOR),
        (c2, a2, name2, P2_COLOR),
    ]:
        col.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{nm} · Hochrechnung</div>
  <div class="metric-value" style="color:{color}">{mins_to_hm(a['annual_min'])}</div>
  <div class="metric-sub">pro Jahr (≈ {a['annual_h']:.0f} Stunden)</div>
  <div style="color:#b6bac4;font-size:.73rem;margin-top:.6rem;border-top:1px solid #f0ede5;padding-top:.5rem">
    Ø {mins_to_hm(a['avg_w'])} / Woche &nbsp;·&nbsp; {a['n_w']} Wochen Daten
  </div>
</div>""", unsafe_allow_html=True)

    with c3:
        diff_h = abs(a1["annual_h"] - a2["annual_h"])
        more   = name1 if a1["annual_h"] >= a2["annual_h"] else name2
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Jahres-Unterschied</div>
  <div class="metric-value">{diff_h:.0f} h</div>
  <div class="metric-sub">{more} verbringt jährlich mehr Zeit vor dem Bildschirm</div>
</div>""", unsafe_allow_html=True)

    st.caption(
        f"Hochrechnung: Wochenschnitt × {weeks_per_year} Wochen. Saisonale Effekte "
        f"(Ferien, Prüfungsphasen) sind nicht modelliert — die tatsächliche Jahressumme "
        f"kann abweichen."
    )

    # ── Vergleichsgrößen ─────────────────────────────────────────────────────────
    section("Was ließe sich in dieser Zeit erleben?")

    with st.expander("⚙️ Annahmen für die Vergleiche anpassen", expanded=False):
        va1, va2, va3 = st.columns(3)
        with va1:
            reading_wpm  = st.slider("Lesegeschwindigkeit (Wörter/min)", 100, 500, 250, 25,
                                     key="v_wpm")
            book_words   = st.slider("Ø Buchlänge (in 1000 Wörtern)", 20, 150, 80, 5,
                                     key="v_bw") * 1000
        with va2:
            walk_min     = st.slider("Dauer Spaziergang (min)", 15, 120, 45, 5, key="v_walk")
            evening_min  = st.slider("Dauer Abend mit Freunden (min)", 60, 240, 120, 15,
                                     key="v_eve")
        with va3:
            sleep_h      = st.slider("Schlaf pro Nacht (h)", 6, 10, 8, 1, key="v_sleep")

    book_h  = (book_words / reading_wpm) / 60        # Stunden pro Buch
    books1  = a1["annual_h"] / book_h  if book_h > 0 else 0
    books2  = a2["annual_h"] / book_h  if book_h > 0 else 0
    walks1  = a1["annual_min"] / walk_min  if walk_min > 0 else 0
    walks2  = a2["annual_min"] / walk_min  if walk_min > 0 else 0
    eves1   = a1["annual_min"] / evening_min if evening_min > 0 else 0
    eves2   = a2["annual_min"] / evening_min if evening_min > 0 else 0
    nights1 = a1["annual_h"] / sleep_h  if sleep_h > 0 else 0
    nights2 = a2["annual_h"] / sleep_h  if sleep_h > 0 else 0
    wdays1  = a1["annual_h"] / waking_h
    wdays2  = a2["annual_h"] / waking_h

    def cmp_card(icon, label, v1, v2, unit, note):
        return f"""
<div class="metric-card">
  <div style="font-size:1.5rem">{icon}</div>
  <div class="metric-label" style="margin:.35rem 0 .1rem">{label}</div>
  <div style="display:flex;gap:1.4rem;margin:.45rem 0 .4rem;align-items:flex-end">
    <div>
      <div style="font-family:'Fraunces',serif;font-size:1.8rem;
                  font-weight:600;color:{P1_COLOR};line-height:1">{v1}</div>
      <div style="color:#9ca3af;font-size:.72rem;margin-top:.1rem">{name1}</div>
    </div>
    <div>
      <div style="font-family:'Fraunces',serif;font-size:1.8rem;
                  font-weight:600;color:{P2_COLOR};line-height:1">{v2}</div>
      <div style="color:#9ca3af;font-size:.72rem;margin-top:.1rem">{name2}</div>
    </div>
    <div style="color:#d1d5db;font-size:.82rem;padding-bottom:.15rem">{unit}</div>
  </div>
  <div style="color:#b6bac4;font-size:.7rem;border-top:1px solid #f0ede5;padding-top:.4rem">{note}</div>
</div>"""

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.markdown(cmp_card("📅", "Wache Lebenstage",
        f"{wdays1:.1f}", f"{wdays2:.1f}", "Tage",
        f"Annahme: {waking_h} Stunden wach/Tag"), unsafe_allow_html=True)
    g2.markdown(cmp_card("📚", "Bücher lesen",
        f"~{int(books1)}", f"~{int(books2)}", "Bücher",
        f"{reading_wpm} WPM · {book_words//1000}k Wörter/Buch"), unsafe_allow_html=True)
    g3.markdown(cmp_card("🌿", "Spaziergänge",
        f"~{int(walks1)}", f"~{int(walks2)}", "Gänge",
        f"{walk_min} min / Spaziergang"), unsafe_allow_html=True)
    g4.markdown(cmp_card("🌙", "Schlafnächte",
        f"~{int(nights1)}", f"~{int(nights2)}", "Nächte",
        f"Annahme: {sleep_h}h pro Nacht"), unsafe_allow_html=True)
    g5.markdown(cmp_card("🧑‍🤝‍🧑", "Abende mit Freunden",
        f"~{int(eves1)}", f"~{int(eves2)}", "Abende",
        f"{evening_min} min / Abend"), unsafe_allow_html=True)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── The Tail End ─────────────────────────────────────────────────────────────
    section("The Tail End — Die Eltern-Perspektive")

    st.markdown("""
<div style="background:#fafaf9;border:1px solid #e8e6df;border-left:4px solid #6366f1;
            border-radius:12px;padding:.9rem 1.2rem;margin-bottom:1.1rem;font-size:.85rem">
  <b>Quelle &amp; Einordnung:</b> Das Konzept, verbleibende Elternzeit zu berechnen, wurde durch
  Tim Urbans Essay <em>„The Tail End"</em> (Wait But Why, 2015) bekannt gemacht.
  Die Kerneinsicht: Wer mit ~18 Jahren auszieht, hat den <em>Großteil</em> der
  gemeinsamen Zeit mit seinen Eltern bereits erlebt — weil danach nur noch wenige
  Besuche pro Jahr bleiben. <b>Diese App berechnet keine gesicherten Zahlen</b>,
  sondern rechnet transparent aus euren eigenen Annahmen. Alle Parameter sind einstellbar.
</div>""", unsafe_allow_html=True)

    sl_col, res_col = st.columns([1, 2])
    with sl_col:
        st.markdown("**Meine Situation**")
        my_age       = st.slider("Mein Alter",           16, 45, 22, 1,  key="te_myage")
        parent_age   = st.slider("Alter Elternteil",     40, 85, 55, 1,  key="te_page")
        life_exp     = st.slider("Lebenserwartung (Annahme)", 70, 100, 83, 1, key="te_lexp")
        visits_yr    = st.slider("Besuche pro Jahr",      1, 52,  5,  1,  key="te_vis")
        days_visit   = st.slider("Tage pro Besuch",       1, 21,  3,  1,  key="te_days")
        st.caption(
            "Lebenserwartung ist unsicher und variiert stark. "
            "Stelle die Zahl so ein, wie es für deine Situation realistisch erscheint."
        )

    with res_col:
        rem_years    = max(0, life_exp - parent_age)
        rem_visits   = rem_years * visits_yr
        rem_days     = rem_visits * days_visit
        rem_h        = rem_days * waking_h        # Wachstunden gemeinsam

        ann_h_avg    = (a1["annual_h"] + a2["annual_h"]) / 2
        ann_days_avg = ann_h_avg / waking_h

        pct1 = (a1["annual_h"] / rem_h * 100) if rem_h > 0 else 0
        pct2 = (a2["annual_h"] / rem_h * 100) if rem_h > 0 else 0

        st.markdown(f"""
<div style="display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap">
  <div class="metric-card" style="flex:1;min-width:160px">
    <div class="metric-label">Verbleibende Jahre</div>
    <div class="metric-value" style="color:#6366f1">{rem_years}</div>
    <div class="metric-sub">mit diesem Elternteil</div>
  </div>
  <div class="metric-card" style="flex:1;min-width:160px">
    <div class="metric-label">Verbleibende Besuche</div>
    <div class="metric-value" style="color:#6366f1">{rem_visits}</div>
    <div class="metric-sub">{visits_yr}×/Jahr · {days_visit} Tage</div>
  </div>
  <div class="metric-card" style="flex:1;min-width:160px">
    <div class="metric-label">Verbleibende Tage</div>
    <div class="metric-value" style="color:#6366f1">{rem_days}</div>
    <div class="metric-sub">gemeinsam (geschätzt)</div>
  </div>
</div>""", unsafe_allow_html=True)

        # Vergleichs-Balken
        bar_data = pd.DataFrame({
            "Kategorie": [
                f"Verbleibende Tage\nmit Elternteil",
                f"Bildschirmzeit/Jahr\n{name1}",
                f"Bildschirmzeit/Jahr\n{name2}",
            ],
            "Tage": [rem_days,
                     round(a1["annual_h"] / waking_h, 1),
                     round(a2["annual_h"] / waking_h, 1)],
            "farbe": ["#6366f1", P1_COLOR, P2_COLOR],
        })
        fig_tail = go.Figure()
        for _, row in bar_data.iterrows():
            fig_tail.add_trace(go.Bar(
                y=[row["Kategorie"]], x=[row["Tage"]],
                orientation="h", marker_color=row["farbe"], marker_line_width=0,
                text=[f"  {row['Tage']:.0f} Tage"],
                textposition="outside", textfont=dict(size=11, color=INK),
                hovertemplate=f"<b>{row['Kategorie']}</b><br>{row['Tage']:.0f} Tage<extra></extra>",
                showlegend=False,
            ))
        fig_tail.update_layout(
            **CS, height=220,
            xaxis=dict(title=f"Tage (Wachstunden ÷ {waking_h})", gridcolor=GRID,
                       linecolor=LINE, zeroline=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=LINE,
                       tickfont=dict(size=11)),
            bargap=0.38, margin=dict(t=10, b=20, l=10, r=80),
        )
        st.plotly_chart(fig_tail, width="stretch")

        st.markdown(
            f'<div style="display:flex;gap:1rem;flex-wrap:wrap">'
            f'<div style="background:{P1_COLOR}12;border:1px solid {P1_COLOR}44;'
            f'border-radius:10px;padding:.65rem 1rem;flex:1;min-width:140px;font-size:.85rem">'
            f'<b style="color:{P1_COLOR}">{name1}</b><br>'
            f'Jährliche Bildschirmzeit entspricht<br>'
            f'<b style="font-family:Fraunces,serif;font-size:1.3rem;color:{P1_COLOR}">'
            f'{pct1:.1f}%</b> der gesch. Elternzeit</div>'
            f'<div style="background:{P2_COLOR}12;border:1px solid {P2_COLOR}44;'
            f'border-radius:10px;padding:.65rem 1rem;flex:1;min-width:140px;font-size:.85rem">'
            f'<b style="color:{P2_COLOR}">{name2}</b><br>'
            f'Jährliche Bildschirmzeit entspricht<br>'
            f'<b style="font-family:Fraunces,serif;font-size:1.3rem;color:{P2_COLOR}">'
            f'{pct2:.1f}%</b> der gesch. Elternzeit</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Die %-Angabe setzt jährliche Bildschirmzeit (als Tage) ins Verhältnis zur "
            "verbleibenden gemeinsamen Zeit mit einem Elternteil. "
            "Beides sind Schätzungen auf Basis deiner Annahmen — keine Fakten."
        )

    # ── Schlussbemerkung ──────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#f6f5f1;border-radius:16px;padding:1.5rem 1.8rem;
            margin-top:1.6rem;max-width:720px">
  <div style="font-family:'Fraunces',serif;font-size:1.15rem;color:#374151;
              font-weight:600;margin-bottom:.5rem">
    Zahlen schaffen Perspektive — keine Urteile.
  </div>
  <div style="color:#6b7280;font-size:.9rem;line-height:1.65">
    Diese Zahlen sagen nichts darüber aus, ob Zeit gut oder schlecht verbracht wurde.
    Bildschirmzeit umfasst Video-Calls mit Freunden, das Lesen von Nachrichten,
    kreative Projekte — und eben auch passives Scrollen.
    Was diese Perspektive leisten kann: ein Bewusstsein für Größenordnungen schaffen,
    das im Alltag leicht verloren geht.
  </div>
</div>""", unsafe_allow_html=True)

# ════════ TAB 5 · Daten ══════════════════════════════════════════════════════════
with tab5:
    if show_raw:
        section("Rohdaten (kombiniert)")
        raw_all = pd.concat([d1["raw"], d2["raw"]], ignore_index=True)
        st.dataframe(raw_all, width="stretch", height=400)
        st.download_button("Kombinierte CSV herunterladen",
                           raw_all.to_csv(index=False).encode("utf-8"),
                           "screentime_combined.csv", "text/csv")
    else:
        st.info('Aktiviere "Rohdaten anzeigen" in der Sidebar.')

    section("CSV-Format (fix)")
    st.markdown('<p class="lead">Pro Woche und Person eine CSV. Die drei Tage sind immer '
                '<b>Montag, Dienstag, Mittwoch</b>. Spalten bleiben unverändert.</p>',
                unsafe_allow_html=True)
    example = pd.DataFrame([
        {"woche": "8", "datum": "16.02.-18.02.", "daten_kategorie": "woche_gesamt", "name": "gesamt",    "dauer_minuten": 1765},
        {"woche": "8", "datum": "16.02.2026",    "daten_kategorie": "tag_gesamt",   "name": "gesamt",    "dauer_minuten": 225},
        {"woche": "8", "datum": "16.02.2026",    "daten_kategorie": "top_app",      "name": "WhatsApp",  "dauer_minuten": 54},
        {"woche": "8", "datum": "16.02.2026",    "daten_kategorie": "top_app",      "name": "Instagram", "dauer_minuten": 47},
        {"woche": "8", "datum": "16.02.2026",    "daten_kategorie": "top_app",      "name": "YouTube",   "dauer_minuten": 38},
    ])
    st.dataframe(example, width="stretch", hide_index=True)
    st.download_button("Vorlage herunterladen",
                       example.to_csv(index=False).encode("utf-8"),
                       "vorlage.csv", "text/csv")
