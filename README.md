# Proof of Life
**Was hast du wirklich gemacht?**

Bildschirmzeit-Vergleich zweier Personen über mehrere Wochen — als interaktives Streamlit-Dashboard.

---

## Schnellstart

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. App starten
streamlit run app.py
```

Der Browser öffnet sich automatisch unter `http://localhost:8501`

---

## Voraussetzungen

- Python 3.10 oder neuer
- pip

```
streamlit>=1.49.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.26.0
scipy>=1.11.0
statsmodels>=0.14.0
```

---

## Daten hochladen

1. App starten — Sidebar links öffnen
2. Namen für Person 1 und Person 2 eintragen
3. Pro Person eine oder mehrere CSV-Dateien hochladen (alle Wochen auf einmal auswählbar)

### CSV-Format

Jede Datei entspricht einer Woche. Die drei erfassten Tage sind immer **Montag, Dienstag, Mittwoch**.

| Spalte | Beispiel | Beschreibung |
|---|---|---|
| `woche` | `12` | ISO-Kalenderwoche |
| `datum` | `16.03.2026` | Format TT.MM.JJJJ (Tageszeilen) |
| `daten_kategorie` | `tag_gesamt` | `woche_gesamt`, `tag_gesamt` oder `top_app` |
| `name` | `WhatsApp` | `gesamt` für Summen, App-Name für top_app |
| `dauer_minuten` | `54` | Zeit in Minuten |

**Beispiel-CSV:**
```
woche,datum,daten_kategorie,name,dauer_minuten
12,16.03.-22.03.,woche_gesamt,gesamt,1765
12,16.03.2026,tag_gesamt,gesamt,225
12,16.03.2026,top_app,WhatsApp,54
12,16.03.2026,top_app,Instagram,47
12,17.03.2026,tag_gesamt,gesamt,196
12,17.03.2026,top_app,WhatsApp,58
```

---

## Aufbau der App

Die Navigation ist zweistufig — oben Gruppe wählen, dann Tab.

### Story
| Tab | Inhalt |
|---|---|
| Lebenszeit | Jahreshochrechnung, Vergleichsgrößen (Bücher, Spaziergänge …), The Tail End |
| Produktivität | Score 0–100 basierend auf Kategorie-Gewichten, Wochentrend, beste/schlechteste Woche |
| Vergleich | Radar gemeinsamer Apps, Differenzdiagramm, Korrelation |

### Analysen
| Tab | Inhalt |
|---|---|
| Zeitverlauf | Wochenverlauf, Tagesdetail, Kennzahlen & Boxplot, Trendwende-Analyse |
| App-Analyse | Top-Apps pro Person, App-Zeitverlauf |
| Verdrängung | Korrelationsmatrix (Substitute vs. Komplemente), p-Werte |

### Daten & Annahmen
| Tab | Inhalt |
|---|---|
| Daten | Rohdaten-Tabelle, CSV-Vorlage herunterladen |
| Kategorien | App → Kategorie zuordnen, Produktivitäts-Gewichte einstellen |

---

## Hinweise

**Produktivitäts-Score** ist subjektiv — er hängt vollständig von den Kategorie-Gewichten ab, die du unter *Daten & Annahmen → Kategorien* einstellst.

**Korrelationen** (Verdrängung): Bei 10–15 Wochen ist n klein. Die meisten Werte sind statistisch nicht signifikant. p-Werte werden angezeigt.

**Trendwenden**: Lokale Hoch-/Tiefpunkte über Vorzeichenwechsel der ersten Differenz — kein formales Change-Point-Verfahren.

---

## Projektstruktur

```
proof-of-life/
├── app.py            ← Hauptanwendung
├── requirements.txt  ← Abhängigkeiten
└── README.md         ← Diese Datei
```
