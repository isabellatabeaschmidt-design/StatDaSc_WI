import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

from scipy.stats import pearsonr, t as t_dist, shapiro, gaussian_kde, probplot as scipy_probplot

# ════════════════════════════════════════════════════════════════════════════════
# ScreenTime Duo · Bildschirmzeit-Vergleich zweier Personen
# Phase 0–2: UI/UX · Kategorien · Produktivitäts-Score · Verdrängungsanalyse
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Proof of Life",
    page_icon=None,
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
    "phone": "Kommunikation",
    # Produktivität / Tools
    "slack": "Produktivität", "teams": "Produktivität", "microsoft teams": "Produktivität",
    "zoom": "Produktivität", "gmail": "Produktivität", "mail": "Produktivität",
    "outlook": "Produktivität", "notion": "Produktivität", "obsidian": "Produktivität",
    "google docs": "Produktivität", "docs": "Produktivität", "word": "Produktivität",
    "excel": "Produktivität", "onenote": "Produktivität", "maps": "Produktivität",
    "google maps": "Produktivität", "calendar": "Produktivität",
    # Produktivität — Kalender & Notizen
    "calender": "Produktivität", "kalender": "Produktivität", "n kalender": "Produktivität",
    "erinnerungen": "Produktivität", "notizen": "Produktivität",
    "calculator": "Produktivität", "rechner": "Produktivität",
    # Produktivität — Navigation & Reise
    "db navigator": "Produktivität", "karten": "Produktivität",
    "google.maps": "Produktivität", "googlemaps": "Produktivität",
    # Produktivität — Finanzen & Banking
    "paypal": "Produktivität", "sparkasse": "Produktivität",
    "s-pushtan": "Produktivität", "trade_republik": "Produktivität",
    "trade republik": "Produktivität",
    # Produktivität — Kommunikation & Utilities
    "fraenk": "Produktivität", "fraenk: die mobilfunk app": "Produktivität",
    "neuland": "Produktivität",
    # Soziale Medien
    "instagram": "Soziale Medien", "tiktok": "Soziale Medien", "snapchat": "Soziale Medien",
    "facebook": "Soziale Medien", "twitter": "Soziale Medien", "x": "Soziale Medien",
    "reddit": "Soziale Medien", "pinterest": "Soziale Medien", "linkedin": "Soziale Medien",
    # Unterhaltung
    "netflix": "Unterhaltung", "spotify": "Unterhaltung", "spotofy": "Unterhaltung",
    "twitch": "Unterhaltung", "disney+": "Unterhaltung", "prime video": "Unterhaltung",
    "amazon prime video": "Unterhaltung",
    "dice": "Unterhaltung", "faceapp": "Unterhaltung", "pon": "Unterhaltung",
    # Soziale Medien
    "youtube": "Soziale Medien",
    # Lernen
    "duolingo": "Lernen", "anki": "Lernen", "coursera": "Lernen", "udemy": "Lernen",
    "khan academy": "Lernen", "kindle": "Lernen", "books": "Lernen",
    "bücher": "Lernen", "chatgpt": "Lernen", "claude": "Lernen",
    "moodle.thi.de": "Lernen", "moodle": "Lernen",
    "strava": "Lernen", "wellpass": "Lernen",
    "einfachbacken.de": "Lernen", "marcelpaa.com": "Lernen",
    # Sonstiges — Shopping
    "bestsecret": "Sonstiges", "breuninger": "Sonstiges", "chrono24": "Sonstiges",
    "vestiaire": "Sonstiges", "vinted": "Sonstiges",
    "zalando": "Sonstiges", "zalandp": "Sonstiges",
    # Sonstiges — Spiele & Utilities
    "sudoku": "Sonstiges", "sudoku.com": "Sonstiges", "sodoku.com": "Sonstiges",
    "chrome": "Sonstiges", "safari": "Sonstiges", "firefox": "Sonstiges",
    "google.com": "Sonstiges",
    "fotos": "Sonstiges", "photos": "Sonstiges", "kamera": "Sonstiges",
    "settings": "Sonstiges", "uhr": "Sonstiges", "wetter": "Sonstiges",
    "wg-gesucht.de": "Sonstiges", "next": "Sonstiges",
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
    # Neue Display-Namen
    "chatgpt": "ChatGPT", "claude": "Claude",
    "db navigator": "DB Navigator", "s-pushtan": "S-pushTAN",
    "n kalender": "N Kalender", "trade_republik": "Trade Republik",
    "wg-gesucht.de": "WG-Gesucht.de", "moodle.thi.de": "Moodle THI",
    "bestsecret": "BestSecret", "chrono24": "Chrono24",
    "vestiaire": "Vestiaire", "vinted": "Vinted",
    "zalando": "Zalando", "breuninger": "Breuninger",
    "sparkasse": "Sparkasse", "paypal": "PayPal",
    "strava": "Strava", "wellpass": "Wellpass",
    "spotofy": "Spotify",   # typo → korrekter Name
    "zalandp": "Zalando",   # typo → korrekter Name
    "sodoku.com": "Sudoku.com",  # typo → korrekter Name
    "sudoku.com": "Sudoku.com",
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

  /* Nav-Group Pills — alle horizontalen Radios ────────────────────────────────── */
  div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: inline-flex; gap: 3px; flex-wrap: nowrap;
    background: #eceae4; border-radius: 13px; padding: 4px 5px;
    border: 1px solid #dedad2;
  }
  div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    display: flex; align-items: center;
    padding: 5px 18px; border-radius: 9px; cursor: pointer;
    font-family: 'Hanken Grotesk', sans-serif;
    font-size: .84rem; font-weight: 600; color: #9ca3af;
    transition: background .12s, color .12s;
    white-space: nowrap;
  }
  div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: #ffffff; color: #23252f;
    box-shadow: 0 1px 4px rgba(35,37,47,.10);
  }
  /* Radio label (widget title) ausblenden wenn collapsed */
  div[data-testid="stRadio"] > label { display: none; }

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


def cmp_card(label, v1, v2, unit, note):
    """Vergleichskarte für zwei Personen (Lebenszeit-Tab)."""
    return f"""
<div class="metric-card">
  <div class="metric-label" style="margin:0 0 .45rem">{label}</div>
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

# ── Leere Datenstruktur (kein Upload) ────────────────────────────────────────────
def empty_data(name: str) -> dict:
    cols = ["woche", "datum", "daten_kategorie", "name", "dauer_minuten", "person"]
    empty = pd.DataFrame(columns=cols)
    return {
        "tag":            empty.copy(),
        "woche":          empty.copy(),
        "apps":           empty.copy(),
        "raw":            empty.copy(),
        "_source":        "empty",
        "_error":         None,
        "_date_failures": 0,
        "_dated_rows":    0,
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
    if not (st.session_state.get("f1") or st.session_state.get("f2")):
        st.markdown(
            f'<div style="background:linear-gradient(135deg,{P1_COLOR}18,{P2_COLOR}18);'
            f'border:1.5px dashed {P1_COLOR}55;border-radius:12px;'
            f'padding:.75rem 1rem;margin-bottom:1rem;font-size:.82rem;color:#374151;'
            f'line-height:1.5">'
            f'<b>Hier hochladen</b><br>'
            f'CSV-Dateien pro Person unterhalb eintragen — alle Wochen auf einmal auswählbar.'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("### Daten")
    st.markdown('<span class="tag-p1">Person 1</span>', unsafe_allow_html=True)
    name1  = st.text_input("Name", value="Sarah", key="n1")
    files1 = st.file_uploader("CSV hochladen (alle Wochen auf einmal)",
                               type="csv", key="f1", accept_multiple_files=True)
    if files1:
        st.caption(f"{len(files1)} Datei{'en' if len(files1)>1 else ''} geladen")
    st.markdown("---")
    st.markdown('<span class="tag-p2">Person 2</span>', unsafe_allow_html=True)
    name2  = st.text_input("Name", value="Bella", key="n2")
    files2 = st.file_uploader("CSV hochladen (alle Wochen auf einmal)",
                               type="csv", key="f2", accept_multiple_files=True)
    if files2:
        st.caption(f"{len(files2)} Datei{'en' if len(files2)>1 else ''} geladen")
    st.markdown("---")
    st.markdown("### Optionen")
    pseudonym = st.toggle("App-Namen pseudonymisieren", False)
    show_raw  = st.toggle("Rohdaten anzeigen", False)
    st.markdown("---")
    st.caption("Proof of Life · Semesterprojekt")

# ── Daten laden ──────────────────────────────────────────────────────────────────
def load(files, name: str) -> dict:
    """Lädt CSV-Dateien; gibt leere Struktur zurück wenn keine Datei vorhanden."""
    if not files:
        return empty_data(name)
    try:
        d = parse_csv(files, name)
    except Exception as e:
        st.error(
            f"**{name}:** CSV konnte nicht gelesen werden (`{e}`). "
            f"Prüfe Spalten: woche, datum, daten_kategorie, name, dauer_minuten."
        )
        return empty_data(name)

    d["tag"],  nf_tag  = add_weekday(d["tag"])
    d["apps"], nf_apps = add_weekday(d["apps"])
    d["apps"]["name"] = d["apps"]["name"].apply(
        lambda n: DISPLAY_NAMES.get(n.lower(), n if any(c.isupper() for c in n) else n.title())
    )
    d["_source"]        = "upload"
    d["_error"]         = None
    d["_date_failures"] = nf_tag + nf_apps
    d["_dated_rows"]    = int(len(d["tag"]) + len(d["apps"]))
    return d

init_category_state()

d1 = load(files1, name1)
d2 = load(files2, name2)

# Datumsformat-Warnung (nur bei echten Uploads mit Problemen)
for d, nm in [(d1, name1), (d2, name2)]:
    if d["_source"] == "upload" and d["_date_failures"] > 0:
        st.warning(
            f"**{nm}:** {d['_date_failures']} von {d['_dated_rows']} Datumswerten "
            f"konnten nicht als **TT.MM.JJJJ** gelesen werden (z. B. 16.03.2026). "
            f"Bitte Datumsformat in der CSV prüfen."
        )

# Warnung wenn nur eine Person Daten hat
_has1 = d1["_source"] == "upload"
_has2 = d2["_source"] == "upload"
if _has1 and not _has2:
    st.warning(f"Nur **{name1}** hat Daten hochgeladen — lade auch für **{name2}** eine CSV hoch, damit alle Vergleiche funktionieren.")
elif _has2 and not _has1:
    st.warning(f"Nur **{name2}** hat Daten hochgeladen — lade auch für **{name1}** eine CSV hoch, damit alle Vergleiche funktionieren.")

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
<div style="padding:1.6rem 0 .4rem">
  <div style="font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;
              color:#9ca3af;font-weight:700;margin-bottom:.45rem;
              font-family:'Hanken Grotesk',sans-serif">Proof of Life</div>
  <h1 style="font-family:'Fraunces',serif;font-size:2.4rem;font-weight:600;
             color:#23252f;margin:0 0 .35rem;line-height:1.08">
    Was hast du wirklich gemacht?
  </h1>
  <div style="color:#9ca3af;font-size:.9rem;font-family:'Hanken Grotesk',sans-serif">
    <span style="color:{P1_COLOR};font-weight:600">{name1}</span>
    &ensp;&middot;&ensp;
    <span style="color:{P2_COLOR};font-weight:600">{name2}</span>
    &ensp;&middot;&ensp;{len(weeks)} Wochen
  </div>
</div>
<div style="height:3px;background:linear-gradient(90deg,{P1_COLOR} 0%,{P2_COLOR} 100%);
            border-radius:3px;margin-bottom:1.6rem;width:120px"></div>
""", unsafe_allow_html=True)

if not files1 and not files2:

    st.markdown(
        f'<div style="background:{P1_COLOR}12;border:1.5px solid {P1_COLOR}44;'
        f'border-radius:12px;padding:.8rem 1.2rem;margin-bottom:1.4rem;'
        f'max-width:460px;display:flex;align-items:center;gap:.8rem">'
        f'<div style="font-size:1.4rem">←</div>'
        f'<div style="font-size:.88rem;color:#374151;line-height:1.45">'
        f'<b>Sidebar öffnen</b> (Pfeil oben links) und dort die CSV-Dateien hochladen.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="color:#6b7280;font-size:.95rem;line-height:1.7;max-width:600px;margin-bottom:1.8rem">'
        "Diese App vergleicht die Bildschirmzeit von zwei Personen über mehrere Wochen — "
        "und erzählt daraus eine Geschichte. Ladet eure CSV-Dateien hoch und los geht's."
        "</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    step_style = (
        "background:#ffffff;border:1px solid #e8e6df;border-radius:16px;"
        "padding:1.3rem 1.4rem;height:100%"
    )
    num_style = (
        f"display:inline-flex;align-items:center;justify-content:center;"
        f"width:26px;height:26px;border-radius:50%;"
        f"background:linear-gradient(135deg,{P1_COLOR},{P2_COLOR});"
        f"font-family:'Fraunces',serif;font-size:.8rem;font-weight:700;"
        f"color:#fff;margin-bottom:.7rem"
    )

    with col1:
        st.markdown(f"""
<div style="{step_style}">
  <div style="{num_style}">1</div>
  <div style="font-weight:600;color:#23252f;font-size:.9rem;margin-bottom:.3rem">Namen eintragen</div>
  <div style="color:#6b7280;font-size:.84rem;line-height:1.55">
    In der <b>Sidebar links</b> eure Namen für Person 1 und Person 2 eintragen.
  </div>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div style="{step_style}">
  <div style="{num_style}">2</div>
  <div style="font-weight:600;color:#23252f;font-size:.9rem;margin-bottom:.3rem">CSV hochladen</div>
  <div style="color:#6b7280;font-size:.84rem;line-height:1.55">
    Pro Person eine oder mehrere CSV-Dateien — ihr könnt <b>alle Wochen gleichzeitig</b> auswählen.
  </div>
</div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
<div style="{step_style}">
  <div style="{num_style}">3</div>
  <div style="font-weight:600;color:#23252f;font-size:.9rem;margin-bottom:.3rem">Kategorien anpassen</div>
  <div style="color:#6b7280;font-size:.84rem;line-height:1.55">
    Unter <b>Daten &amp; Annahmen → Kategorien</b> legt ihr fest, was produktiv zählt.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    section("So muss die CSV aussehen")

    st.markdown("""
<div style="background:#ffffff;border:1px solid #e8e6df;border-radius:14px;padding:1.2rem 1.5rem;max-width:580px">
  <div style="font-family:monospace;font-size:.82rem;color:#374151;line-height:2">
    woche, datum, daten_kategorie, name, dauer_minuten<br>
    12, 16.03.2026, tag_gesamt, gesamt, 225<br>
    12, 16.03.2026, top_app, WhatsApp, 54<br>
    12, 16.03.2026, top_app, Instagram, 47<br>
    12, 16.03.-22.03., woche_gesamt, gesamt, 1765
  </div>
  <div style="color:#9ca3af;font-size:.78rem;margin-top:.8rem;border-top:1px solid #f0ede5;padding-top:.6rem">
    Drei Tage pro Woche: <b>Montag, Dienstag, Mittwoch</b> &nbsp;·&nbsp; Datum: <b>TT.MM.JJJJ</b> &nbsp;·&nbsp; Leerzeilen zwischen Blöcken sind okay.
  </div>
</div>""", unsafe_allow_html=True)

    st.stop()

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

# ── Zwei-Ebenen-Navigation ────────────────────────────────────────────────────────
st.markdown("<div style='margin-bottom:.7rem'></div>", unsafe_allow_html=True)
nav_group = st.radio(
    "Bereich",
    options=["Story", "Analysen", "Daten & Annahmen"],
    horizontal=True,
    label_visibility="collapsed",
    key="nav_group",
)
st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# STORY — Lebenszeit · Produktivität · Vergleich
# ══════════════════════════════════════════════════════════════════════════════════
if nav_group == "Story":
    s_life, s_prod, s_cmp = st.tabs(["Lebenszeit", "Produktivität", "Vergleich"])

    # ════════ STORY · Lebenszeit ═════════════════════════════════════════════════
    with s_life:

        st.markdown("""
<div style="background:#f6f5f1;border-radius:16px;padding:1.3rem 1.6rem;
            margin-bottom:1.6rem;max-width:680px">
  <div style="font-family:'Fraunces',serif;font-size:1.2rem;color:#374151;
              font-weight:600;margin-bottom:.35rem">Zeit in Perspektive</div>
  <div style="color:#6b7280;font-size:.88rem;line-height:1.6">
    Diese Seite rechnet eure Bildschirmzeit in andere Einheiten um —
    nicht um zu urteilen, sondern um Größenordnungen greifbar zu machen.
    <b>Alle Annahmen sind einstellbar.</b>
  </div>
</div>""", unsafe_allow_html=True)

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

        ann1 = annual_stats(d1)
        ann2 = annual_stats(d2)

        lc1, lc2, lc3 = st.columns(3)
        for col, a, nm, color in [
            (lc1, ann1, name1, P1_COLOR),
            (lc2, ann2, name2, P2_COLOR),
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

        with lc3:
            diff_h = abs(ann1["annual_h"] - ann2["annual_h"])
            more   = name1 if ann1["annual_h"] >= ann2["annual_h"] else name2
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

        # ── Vergleichsgrößen ──────────────────────────────────────────────────────
        section("Was ließe sich in dieser Zeit erleben?")

        with st.expander("Annahmen für die Vergleiche anpassen", expanded=False):
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

        book_h  = (book_words / reading_wpm) / 60
        books1  = ann1["annual_h"] / book_h  if book_h > 0 else 0
        books2  = ann2["annual_h"] / book_h  if book_h > 0 else 0
        walks1  = ann1["annual_min"] / walk_min  if walk_min > 0 else 0
        walks2  = ann2["annual_min"] / walk_min  if walk_min > 0 else 0
        eves1   = ann1["annual_min"] / evening_min if evening_min > 0 else 0
        eves2   = ann2["annual_min"] / evening_min if evening_min > 0 else 0
        nights1 = ann1["annual_h"] / sleep_h  if sleep_h > 0 else 0
        nights2 = ann2["annual_h"] / sleep_h  if sleep_h > 0 else 0
        wdays1  = ann1["annual_h"] / waking_h
        wdays2  = ann2["annual_h"] / waking_h



        g1, g2, g3, g4, g5 = st.columns(5)
        g1.markdown(cmp_card("Wache Lebenstage",
            f"{wdays1:.1f}", f"{wdays2:.1f}", "Tage",
            f"Annahme: {waking_h} Stunden wach/Tag"), unsafe_allow_html=True)
        g2.markdown(cmp_card("Bücher lesen",
            f"~{int(books1)}", f"~{int(books2)}", "Bücher",
            f"{reading_wpm} WPM · {book_words//1000}k Wörter/Buch"), unsafe_allow_html=True)
        g3.markdown(cmp_card("Spaziergänge",
            f"~{int(walks1)}", f"~{int(walks2)}", "Gänge",
            f"{walk_min} min / Spaziergang"), unsafe_allow_html=True)
        g4.markdown(cmp_card("Schlafnächte",
            f"~{int(nights1)}", f"~{int(nights2)}", "Nächte",
            f"Annahme: {sleep_h}h pro Nacht"), unsafe_allow_html=True)
        g5.markdown(cmp_card("Abende mit Freunden",
            f"~{int(eves1)}", f"~{int(eves2)}", "Abende",
            f"{evening_min} min / Abend"), unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        # ── The Tail End ──────────────────────────────────────────────────────────
        section("The Tail End — Die Eltern-Perspektive")

        st.markdown("""
<div style="background:#fafaf9;border:1px solid #e8e6df;border-left:4px solid #6366f1;
            border-radius:12px;padding:.9rem 1.2rem;margin-bottom:1.1rem;font-size:.85rem">
  <b>Quelle &amp; Einordnung:</b> Das Konzept, verbleibende Elternzeit zu berechnen, wurde durch
  Tim Urbans Essay <em>„The Tail End"</em> (Wait But Why, 2015) bekannt gemacht.
  Die Kerneinsicht: Wer mit ~18 Jahren auszieht, hat den <em>Großteil</em> der
  gemeinsamen Zeit mit seinen Eltern bereits erlebt. <b>Diese App berechnet keine gesicherten Zahlen</b>,
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
            rem_h        = rem_days * waking_h

            ann_h_avg    = (ann1["annual_h"] + ann2["annual_h"]) / 2
            ann_days_avg = ann_h_avg / waking_h

            pct1 = (ann1["annual_h"] / rem_h * 100) if rem_h > 0 else 0
            pct2 = (ann2["annual_h"] / rem_h * 100) if rem_h > 0 else 0

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

            bar_data = pd.DataFrame({
                "Kategorie": [
                    f"Verbleibende Tage\nmit Elternteil",
                    f"Bildschirmzeit/Jahr\n{name1}",
                    f"Bildschirmzeit/Jahr\n{name2}",
                ],
                "Tage": [rem_days,
                         round(ann1["annual_h"] / waking_h, 1),
                         round(ann2["annual_h"] / waking_h, 1)],
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
  </div>
</div>""", unsafe_allow_html=True)

    # ════════ STORY · Produktivität ══════════════════════════════════════════════
    with s_prod:

        st.markdown("""
<div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;
            border-radius:12px;padding:1rem 1.3rem;margin-bottom:1.2rem">
  <div style="font-weight:700;color:#92400e;margin-bottom:.25rem">Dieser Score ist subjektiv</div>
  <div style="color:#78350f;font-size:.88rem;line-height:1.55">
    Der Produktivitäts-Score berechnet sich aus den Gewichten, die du unter
    <b>Daten &amp; Annahmen → Kategorien</b> eingestellt hast. Gleiche Nutzung mit anderen Gewichten
    ergibt einen anderen Score. Er misst also nicht <em>objektive Produktivität</em>,
    sondern wie gut die Bildschirmzeit mit deinen eigenen Prioritäten übereinstimmt.
    Die Formel: <code>Score = (Σ App-Zeit × Gewicht / Σ App-Zeit + 1) / 2 × 100</code>
  </div>
</div>""", unsafe_allow_html=True)

        res1 = compute_score(d1["apps"], d1["tag"])
        res2 = compute_score(d2["apps"], d2["tag"])

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

        section("Wöchentlicher Score-Trend")
        score_df1 = compute_weekly_scores(d1["apps"], d1["tag"], weeks)
        score_df2 = compute_weekly_scores(d2["apps"], d2["tag"], weeks)

        if not score_df1.empty or not score_df2.empty:
            trend_frames = []
            if not score_df1.empty:
                score_df1["person"] = name1
                trend_frames.append(score_df1)
            if not score_df2.empty:
                score_df2["person"] = name2
                trend_frames.append(score_df2)
            trend = pd.concat(trend_frames, ignore_index=True)

            fig_trend = px.line(
                trend, x="woche", y="score", color="person",
                color_discrete_map={name1: P1_COLOR, name2: P2_COLOR},
                markers=True,
                labels={"score": "Score (0 – 100)", "woche": "KW", "person": ""},
            )
            fig_trend.update_traces(line_width=2.6, marker_size=9)
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

            cw_both = sorted(set(score_df1["woche"].tolist()) & set(score_df2["woche"].tolist()),
                             key=lambda x: int(x) if str(x).isdigit() else x)
            if cw_both and not score_df1.empty and not score_df2.empty:
                s1_map = score_df1.set_index("woche")["score"]
                s2_map = score_df2.set_index("woche")["score"]
                deltas  = [s1_map[w] - s2_map[w] for w in cw_both]
                avg_delta = sum(deltas) / len(deltas)
                winner = name1 if avg_delta > 0 else name2
                st.caption(
                    f"Ø Score-Differenz über alle Wochen: {abs(avg_delta):.1f} Punkte "
                    f"zugunsten von **{winner}**."
                )
        else:
            st.info("Noch keine Daten für den Score-Trend vorhanden.")

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

        section("Beste & schlechteste Woche")
        score_pair = [(score_df1, name1, P1_COLOR), (score_df2, name2, P2_COLOR)]
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

    # ════════ STORY · Vergleich ══════════════════════════════════════════════════
    with s_cmp:
        section("Gemeinsame Apps — Radardiagramm")
        app_df1 = d1["apps"].copy(); app_df1["name_l"] = app_df1["name"].str.lower()
        app_df2 = d2["apps"].copy(); app_df2["name_l"] = app_df2["name"].str.lower()
        shared_l = sorted(set(app_df1["name_l"]) & set(app_df2["name_l"]))[:7]

        if len(shared_l) >= 3:
            r1 = app_df1.groupby("name_l")["dauer_minuten"].sum()
            r2 = app_df2.groupby("name_l")["dauer_minuten"].sum()
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

# ══════════════════════════════════════════════════════════════════════════════════
# ANALYSEN — Zeitverlauf · App-Analyse · Verdrängung
# ══════════════════════════════════════════════════════════════════════════════════
elif nav_group == "Analysen":
    a_zeit, a_app, a_verd = st.tabs(["Zeitverlauf", "App-Analyse", "Verdrängung"])

    # ════════ ANALYSEN · Zeitverlauf ═════════════════════════════════════════════
    with a_zeit:
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

        section("Trendwende-Analyse (lokale Hoch- & Tiefpunkte)")
        smooth_w = st.slider("Glättungsfenster (Wochen)", 2, 4, 3, key="tp_window",
                             help="Größeres Fenster → weniger, robustere Trendwenden")
        st.markdown(
            f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:'
            f'4px solid #10b981;border-radius:12px;padding:.85rem 1.2rem;'
            f'margin-bottom:1rem;font-size:.84rem;color:#065f46">'
            f'<b>Methodik (Rolling-Differenzen, Fenster = {smooth_w} Wochen)</b>'
            f'&ensp;·&ensp;Wochenwerte glätten → <b>erste</b> Differenz → Vorzeichenwechsel = '
            f'lokales Maximum/Minimum (Trendwende).<br>'
            f'<span style="color:#047857">Begriffs-Hinweis:</span> Das sind '
            f'<b>Trendwenden</b> (die Reihe kehrt ihre Richtung um), nicht „Wendepunkte" '
            f'im mathematischen Sinn. Kein PELT/BINSEG, weil formale Change-Point-'
            f'Verfahren bei n &lt; 20 unzuverlässig sind.</div>',
            unsafe_allow_html=True,
        )

        def make_weekly_series(d: dict) -> pd.Series:
            s = d["woche"].groupby("woche")["dauer_minuten"].sum()
            return s.reindex(weeks).fillna(np.nan)

        series1 = make_weekly_series(d1)
        series2 = make_weekly_series(d2)
        tp1 = detect_turning_points(series1.dropna(), smooth_window=smooth_w)
        tp2 = detect_turning_points(series2.dropna(), smooth_window=smooth_w)

        fig_tp = go.Figure()
        configs = [
            (series1, tp1, name1, P1_COLOR, "rgba(23,179,166,0.18)",  -55),
            (series2, tp2, name2, P2_COLOR, "rgba(236,108,166,0.18)", +55),
        ]
        for ws, tp, nm, color, fill_rgba, ay_base in configs:
            ws_clean = ws.dropna()
            if ws_clean.empty:
                continue
            weeks_nm = ws_clean.index.tolist()
            smoothed  = ws_clean.rolling(window=smooth_w, center=True, min_periods=1).mean()

            fig_tp.add_trace(go.Scatter(
                x=weeks_nm, y=ws_clean.values,
                name=nm, mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color, line=dict(color="white", width=1.2)),
                hovertemplate=f"<b>{nm}</b> KW%{{x}}<br>%{{y:.0f}} min<extra></extra>",
            ))
            fig_tp.add_trace(go.Scatter(
                x=weeks_nm, y=smoothed.values,
                name=f"{nm} (geglättet)", mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.55, showlegend=False,
                hovertemplate=f"<b>{nm} glatt</b> KW%{{x}}<br>%{{y:.0f}} min<extra></extra>",
            ))
            if tp.empty:
                continue
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
            for _, row in tp.iterrows():
                ay = ay_base
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
            "Sterne markieren lokale Extrema (▼ Hochpunkt, ▲ Tiefpunkt). "
            "P1-Annotationen erscheinen oberhalb, P2-Annotationen unterhalb."
        )

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

        # ── Kennzahlen & Boxplot ──────────────────────────────────────────────
        section("Kennzahlen & Streuung")
        kz_cols = st.columns(2)
        for col, d, name, color in [(kz_cols[0], d1, name1, P1_COLOR),
                                     (kz_cols[1], d2, name2, P2_COLOR)]:
            vals = d["tag"]["dauer_minuten"]
            if vals.empty:
                continue
            with col:
                st.markdown(
                    f'<div style="font-weight:600;color:{color};'
                    f'font-size:.88rem;margin-bottom:.5rem">{name}</div>',
                    unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                for mc, label, v in [
                    (m1, "Mittelwert", mins_to_hm(vals.mean())),
                    (m2, "Median",     mins_to_hm(vals.median())),
                    (m3, "Std.-Abw.",  mins_to_hm(vals.std())),
                    (m4, "Max",        mins_to_hm(vals.max())),
                ]:
                    mc.markdown(f"""<div class="metric-card">
                      <div class="metric-label">{label}</div>
                      <div class="metric-value" style="font-size:1.1rem;color:{color}">{v}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        fig_box = px.box(tag_all, x="woche", y="dauer_minuten", color="person",
                         color_discrete_sequence=[P1_COLOR, P2_COLOR],
                         labels={"dauer_minuten": "Minuten / Tag", "woche": "KW", "person": ""})
        fig_box.update_layout(**CS, height=340, margin=dict(t=10, b=10), legend_title_text="")
        st.plotly_chart(fig_box, width="stretch")
        st.caption("Boxplot zeigt Median, IQR (Box), Whisker (1.5 × IQR) und Ausreißerpunkte pro Woche.")

    # ════════ ANALYSEN · App-Analyse ═════════════════════════════════════════════
    with a_app:
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

    # ════════ ANALYSEN · Verdrängung ═════════════════════════════════════════════
    with a_verd:

        st.markdown("""
<div style="background:#fff7ed;border:1px solid #fed7aa;border-left:4px solid #f97316;
            border-radius:12px;padding:1rem 1.3rem;margin-bottom:1.2rem">
  <div style="font-weight:700;color:#9a3412;margin-bottom:.3rem">
    Statistische Einschränkungen — bitte lesen
  </div>
  <div style="color:#7c2d12;font-size:.875rem;line-height:1.6">
    <b>Korrelation ≠ Kausalität.</b> Selbst eine starke negative Korrelation zwischen
    App A und App B bedeutet nicht, dass A die Nutzung von B verdrängt.<br><br>
    <b>Kleines n = breite Konfidenzintervalle.</b> Mit 10–15 Wochen als Datenpunkte sind
    fast alle Korrelationen statistisch nicht signifikant (p ≥ 0.05). Das ist keine
    Fehlfunktion, sondern ehrliche Statistik. Für belastbare Ergebnisse wären ≥ 30 Wochen nötig.
  </div>
</div>""", unsafe_allow_html=True)

        col_ps, col_mp = st.columns([3, 1])
        with col_ps:
            sel_verd = st.radio("Person", [name1, name2], horizontal=True, key="verd_p")
        with col_mp:
            min_pres = st.number_input("Mindest-Wochen für App", 2, 10, 3, 1,
                                       key="verd_min",
                                       help="App muss in mind. N Wochen aktiv gewesen sein.")

        dv = d1 if sel_verd == name1 else d2
        clr_verd = P1_COLOR if sel_verd == name1 else P2_COLOR

        section("App-zu-App-Korrelationen")
        corr_df, pval_df, n_weeks = compute_corr_matrix(dv["apps"], min_presence=min_pres)

        if corr_df is None:
            st.info(f"Zu wenige Apps oder Wochen für die Analyse "
                    f"(Mindest-Wochen: {min_pres}). Passe den Regler oben an.")
        else:
            k = len(corr_df)
            r_crit = critical_r(n_weeks)
            st.markdown(
                f'<div style="background:#f6f5f1;border-radius:10px;padding:.75rem 1.1rem;'
                f'margin-bottom:1rem;font-size:.84rem;color:#4b5563">'
                f'<b>n = {n_weeks} Wochen</b> &nbsp;·&nbsp; '
                f'{k} Apps (mind. {min_pres}× aktiv) &nbsp;·&nbsp; '
                f'Signifikanzschwelle: |r| &gt; <b>{r_crit:.2f}</b> für p &lt; 0.05 '
                f'&nbsp;·&nbsp; <span style="color:#6b7280">Sterne: * p&lt;0.05 &nbsp;'
                f'** p&lt;0.01 &nbsp; *** p&lt;0.001 &nbsp; ns = nicht signifikant</span>'
                f'</div>', unsafe_allow_html=True)

            fig_hm = corr_heatmap(corr_df, pval_df, height=max(380, k * 46))
            st.plotly_chart(fig_hm, width="stretch")

            pairs_df = build_pairs_df(corr_df, pval_df, top_n=6)
            if not pairs_df.empty:
                section("Stärkste Substitute & Komplemente")
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

                with st.expander("Vollständige Tabelle aller Paare"):
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

        section("Kategorie-zu-Kategorie-Korrelationen")
        st.markdown('<p class="lead">Verdrängt Unterhaltung das Lernen? '
                    'Gehen Kommunikation und Soziale Medien Hand in Hand?</p>',
                    unsafe_allow_html=True)

        cat_apps = dv["apps"].copy()
        cat_apps["name"] = cat_apps["kategorie"]
        corr_cat, pval_cat, n_cat = compute_corr_matrix(cat_apps, min_presence=1)

        if corr_cat is None:
            st.info("Zu wenige Kategorien oder Wochen für die Kategorie-Analyse.")
        else:
            k_cat = len(corr_cat)
            fig_cat_hm = corr_heatmap(corr_cat, pval_cat, height=max(340, k_cat * 52))
            st.plotly_chart(fig_cat_hm, width="stretch")

            cat_pairs = build_pairs_df(corr_cat, pval_cat, top_n=4)
            if not cat_pairs.empty:
                col_pos, col_neg = st.columns(2)
                pos_pairs = cat_pairs[cat_pairs["r"] > 0]
                neg_pairs = cat_pairs[cat_pairs["r"] < 0]

                with col_pos:
                    st.markdown(
                        '<div style="font-weight:700;color:#10b981;margin-bottom:.4rem">'
                        'Komplementäre Kategorien</div>', unsafe_allow_html=True)
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
                        'Substituierende Kategorien</div>', unsafe_allow_html=True)
                    for _, row in neg_pairs.iterrows():
                        st.markdown(
                            f'<div style="background:#fef2f2;border-radius:8px;'
                            f'padding:.6rem .9rem;margin-bottom:.4rem;font-size:.85rem">'
                            f'<b>{row["a"]}</b> vs. <b>{row["b"]}</b><br>'
                            f'<span style="color:#ef4444">r = {row["r"]:+.2f}</span> &nbsp; '
                            f'<span style="color:#9ca3af">{row["sig"]}</span></div>',
                            unsafe_allow_html=True)

        st.markdown("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
            padding:.9rem 1.1rem;margin-top:1.2rem;font-size:.82rem;color:#64748b">
  <b>Methodik:</b> Pearson-Korrelation der wöchentlichen App-Zeiten (Σ Minuten/Woche).
  Wochen ohne Nutzung einer App zählen als 0. Apps mit konstantem Wert über alle Wochen
  (Std. = 0) werden ausgeschlossen. Für belastbare Inferenz werden ≥ 30 unabhängige
  Beobachtungen empfohlen; diese Analyse ist explorativer Natur.
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# DATEN & ANNAHMEN — Daten · Kategorien
# ══════════════════════════════════════════════════════════════════════════════════
elif nav_group == "Daten & Annahmen":
    d_daten, d_cat = st.tabs(["Daten", "Kategorien — Produktivitätsranking"])

    # ════════ DATEN & ANNAHMEN · Daten ═══════════════════════════════════════════
    with d_daten:
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

    # ════════ DATEN & ANNAHMEN · Kategorien ══════════════════════════════════════
    with d_cat:
        section("App → Kategorie zuordnen")
        st.markdown("""
<div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;
            border-radius:12px;padding:.9rem 1.2rem;margin-bottom:1.1rem;font-size:.85rem">
  <b style="color:#92400e">Diese Einstellungen bestimmen den Produktivitäts-Score</b>
  <span style="color:#78350f"> — du findest die Auswertung unter <b>Story → Produktivität</b>.</span>
</div>""", unsafe_allow_html=True)

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
                    'sie bilden die Grundlage für den Produktivitäts-Score unter Story → Produktivität.</p>',
                    unsafe_allow_html=True)
        wcols = st.columns(3)
        for i, cat in enumerate(CATEGORIES):
            with wcols[i % 3]:
                st.markdown(f'<span class="cat-chip" style="background:{CAT_COLORS[cat]}22;'
                            f'color:{CAT_COLORS[cat]}">{cat}</span>', unsafe_allow_html=True)
                st.slider(cat, -1.0, 1.0, step=0.1, key=f"w_{cat}", label_visibility="collapsed")

        section("Zeitverteilung nach Kategorie")
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
