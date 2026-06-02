# 📱 ScreenTime Duo

Bildschirmzeit-Vergleich zweier Personen über 10–15 Wochen — statistisches Storytelling mit Streamlit.

## Schnellstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ohne hochgeladene Dateien startet die App mit **Demo-Daten** (13 Wochen, je Mo/Di/Mi).

---

## CSV-Format (fix, ändert sich nicht)

Pro Person werden beliebig viele CSV-Dateien hochgeladen — typischerweise eine pro Woche.
Alle Dateien folgen exakt diesem Format:

| Spalte            | Typ    | Beschreibung                                              |
|-------------------|--------|-----------------------------------------------------------|
| `woche`           | Zahl   | ISO-Kalenderwoche, z. B. `12`                             |
| `datum`           | Text   | Tageszeilen: `TT.MM.JJJJ` · Wochenzeile: `TT.MM.-TT.MM.` |
| `daten_kategorie` | Text   | `woche_gesamt`, `tag_gesamt` oder `top_app`               |
| `name`            | Text   | `gesamt` für Tages-/Wochensummen, App-Name für top_app    |
| `dauer_minuten`   | Zahl   | Zeit in Minuten                                           |

**Tage:** Jede Woche enthält genau **3 Tage: Montag, Dienstag, Mittwoch**.
**Datumsformat:** Tageszeilen müssen `TT.MM.JJJJ` sein (z. B. `16.03.2026`). Abweichende
Formate werden erkannt und in der App als Warnung gemeldet (keine stillen Fehler).
**Leerzeilen** zwischen Blöcken sind erlaubt und werden ignoriert.
**App-Namen** dürfen klein oder groß geschrieben sein (werden automatisch normalisiert).
**Leerzeichen** vor Spaltennamen werden automatisch entfernt.
Doppelt hochgeladene Wochen werden automatisch dedupliziert.

### Beispiel (eine Woche = eine Datei)

```
woche,datum,daten_kategorie,name, dauer_minuten
12,16.03.-22.03.,woche_gesamt,gesamt,1765

12,16.03.2026,tag_gesamt,gesamt,225
12,16.03.2026,top_app,whatsapp,54
12,16.03.2026,top_app,instagram,47
12,16.03.2026,top_app,youtube,38

12,17.03.2026,tag_gesamt,gesamt,196
12,17.03.2026,top_app,whatsapp,58
12,17.03.2026,top_app,slack,43

12,18.03.2026,tag_gesamt,gesamt,290
12,18.03.2026,top_app,whatsapp,140
12,18.03.2026,top_app,sudoku,41
```

---

## Funktionen (Tabs)

| Tab             | Inhalt                                                                       |
|-----------------|------------------------------------------------------------------------------|
| Zeitverlauf     | Wochenverlauf, Tagesvergleich (Mo/Di/Mi), **Trendwende-Analyse**             |
| App-Analyse     | Top-Apps pro Person (Farbe = Kategorie), App-Zeitverlauf                      |
| Vergleich       | Radar gemeinsamer Apps, Differenzdiagramm, Korrelation                       |
| Produktivität   | Score 0–100 aus Kategorie-Gewichten, Wochentrend, beste/schlechteste Woche   |
| Verdrängung     | App- & Kategorie-Korrelationsmatrix (Substitute vs. Komplemente), p-Werte    |
| Verteilung      | Violin, Histogramm+KDE, Q-Q-Plot, Shapiro-Wilk, IQR-Ausreißer, KDE/Kategorie |
| Statistik       | Kennzahlen, Boxplot, Heatmap, **Bootstrap-Konfidenzintervalle je Wochentag** |
| Kategorien      | App→Kategorie zuordnen (editierbar), Gewichte einstellen, Verteilung         |
| Lebenszeit      | Hochrechnung, Vergleichsgrößen, „The Tail End"-Eltern-Perspektive            |
| Daten           | Rohdaten, CSV-Vorlage herunterladen                                          |

### Statistische Hinweise (Ehrlichkeit)

- **Produktivitäts-Score** ist subjektiv — er hängt vollständig von den eingestellten
  Kategorie-Gewichten ab. Das UI kommuniziert das offen.
- **Korrelationen** (Verdrängung): bei 10–15 Wochen ist *n* klein; die meisten Werte
  sind statistisch nicht signifikant. Korrelation ≠ Kausalität. p-Werte werden angezeigt.
- **Konfidenzintervalle** je Wochentag: Bootstrap (keine Verteilungsannahme); zusätzlich
  wird die Standardabweichung (Streuung) gezeigt.
- **Normalitätstest** (Shapiro-Wilk): bei kleinem n geringe Teststärke — p > 0.05 heißt
  nicht „normalverteilt". Der Q-Q-Plot ist oft informativer.
- **Trendwenden** (Zeitverlauf): lokale Hoch-/Tiefpunkte über Vorzeichenwechsel der
  *ersten* Differenz — bewusst nicht „Wendepunkte" im mathematischen Sinn (zweite
  Ableitung) und kein formales Change-Point-Verfahren (instabil bei n < 20).
- **Lebenszeit**: alle Annahmen (Lesetempo, Besuche/Jahr, Wachstunden, Wochen/Jahr …)
  sind einstellbar. Nichts wird als gesicherte Statistik behauptet. Das „Tail End"-Konzept
  geht auf Tim Urbans Essay *„The Tail End"* (Wait But Why, 2015) zurück.

---

## Abhängigkeiten

```
streamlit · pandas · plotly · numpy · scipy · statsmodels
```

- `scipy` — Bootstrap-Hilfen, Shapiro-Wilk, Q-Q (probplot), KDE, Pearson-Korrelation
- `statsmodels` — OLS-Trendlinie im Korrelations-Scatter (degradiert sauber, falls nicht installiert)

## Projektstruktur

```
screentime_app/
├── app.py            ← Hauptanwendung
├── requirements.txt  ← Python-Abhängigkeiten
└── README.md         ← Diese Datei
```
