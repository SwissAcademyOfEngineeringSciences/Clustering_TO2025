
from pathlib import Path
from collections import Counter
import re
import unicodedata

from docx import Document
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ============================================================
# 1. EINSTELLUNGEN
# ============================================================

# Hauptordner liegt im gleichen Verzeichnis wie dieses Script
HAUPTORDNER = Path(__file__).parent / "C:/Users/TobiasSchlegel/Desktop/texte"

# Speicherort der Excel-Datei
EXCEL_DATEI = HAUPTORDNER / "Word_Auswertung.xlsx"


# Metadatenfelder
METADATEN = {
    "Untertitel": ["Untertitel"],
    "Forschungsfeld": ["Forschungsfeld"],
    "Autor:in": [
        "Autor:innen",
        "Autorinnen",
        "Autor:in",
        "Autoren",
        "Autorin",
        "Autor"
    ],
    "Expert:innen": [
        "Expert:innen",
        "Expertinnen",
        "Expert:in",
        "Experten",
        "Expertin",
        "Experte"
    ]
}


# Feste Zuordnung der drei Textabschnitte
ABSCHNITTE = {
        "Definition": [
        "Definition",
        "Definition (600 Zeichen inkl. Leerzeichen)",
        "Definition (600 Zeichen)"
    ],
    "Anwendungen": [
        "Heutige und zukünftige Anwendungen sowie Chancen",
        "Heutige und zukünftige Anwendungen sowie Chancen (1700 Zeichen)"
    ],
    "Herausforderungen": [
        "Technische Herausforderungen sowie regulatorische und gesellschaftliche Hemmnisse",
        "Technische Herausforderungen sowie regulatorische und gesellschaftliche Hemmnisse (1300 Zeichen)"
    ]
}

# ============================================================
# 2. HILFSFUNKTIONEN
# ============================================================

def normalisieren(text):
    """Vereinheitlicht Texte für Vergleiche."""

    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(":")

    return text


def text_ohne_formatierung(paragraph):
    """Liest den Text eines Word-Absatzes aus."""

    return paragraph.text.strip()


def absatz_ist_fett(paragraph):
    """
    Prüft, ob ein gesamter Absatz fett formatiert ist.
    Wird für die bisherige Abschnittszuordnung verwendet.
    """

    runs = [
        run for run in paragraph.runs
        if run.text.strip()
    ]

    if not runs:
        return False

    return all(run.bold is True for run in runs)


def absatz_ist_kursiv(paragraph):
    """Prüft, ob ein gesamter Absatz kursiv formatiert ist."""

    runs = [
        run for run in paragraph.runs
        if run.text.strip()
    ]

    if not runs:
        return False

    return all(run.italic is True for run in runs)


def absatz_ist_kontrolltitel(paragraph):
    """
    Erkennt fett formatierte Zwischentitel für das Kontrollblatt.

    Voraussetzungen:
    - eigener Word-Absatz (Enter)
    - gesamter Absatz fett
    - keine manuellen Zeilenumbrüche (Shift + Enter)
    - keine einzelnen fettgedruckten Wörter im Fliesstext
    """

    text = paragraph.text.strip()

    if not text:
        return False

    # Manuelle Zeilenumbrüche ausschliessen
    if "\n" in text or "\r" in text:
        return False

    runs = [
        run for run in paragraph.runs
        if run.text.strip()
    ]

    if not runs:
        return False

    # Alle sichtbaren Textteile müssen fett sein
    return all(run.bold is True for run in runs)


def metadaten_auslesen(paragraph, feld):
    """
    Sucht die Metadatenbezeichnung am Absatzanfang
    und übernimmt den nachfolgenden nicht-fetten Text.
    """

    text = paragraph.text.strip()

    if not normalisieren(text).startswith(normalisieren(feld)):
        return None

    match = re.match(
        r"^\s*" + re.escape(feld) + r"\s*:?\s*",
        text,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    rest_start = match.end()

    position = 0
    ergebnis = ""

    for run in paragraph.runs:

        for zeichen in run.text:

            if position >= rest_start:

                if run.bold is not True:
                    ergebnis += zeichen

            position += 1

    return ergebnis.strip() or None

def abschnitt_zuordnen(ueberschrift):
    vergleich = ueberschrift.strip()

    for excel_spalte, word_titel in ABSCHNITTE.items():
        if isinstance(word_titel, list):
            if vergleich in word_titel:
                return excel_spalte
        elif vergleich == word_titel:
            return excel_spalte

    return None

# ============================================================
# 3. WORD-DOKUMENT AUSLESEN
# ============================================================

def dokument_auslesen(dateipfad):

    doc = Document(dateipfad)

    ergebnis = {
        "Dateiname": dateipfad.name,
        "Titel": "",
        "Untertitel": "",
        "Forschungsfeld": "",
        "Autor:in": "",
        "Expert:innen": "",
        "Lead": "",
        "Definition": "",
        "Anwendungen": "",
        "Herausforderungen": "",
        "Dateipfad": str(dateipfad)
    }

    absätze = doc.paragraphs

    # --------------------------------------------------------
    # A. TITEL FINDEN
    # --------------------------------------------------------
    # TITEL AUSLESEN
    for p in absätze:
        text = text_ohne_formatierung(p).strip()

        # Leere Absätze am Anfang überspringen
        if not text:
            continue

        # Ersten Textabsatz als Titel übernehmen
        # Unabhängig von Fett- oder Titel-Formatierung
        ergebnis["Titel"] = text
        break
    # --------------------------------------------------------
    # B. METADATEN FINDEN
    # --------------------------------------------------------
    # METADATEN AUSLESEN
    for p in absätze:
        text = text_ohne_formatierung(p).strip()

        if not text:
            continue

        for excel_spalte, bezeichnungen in METADATEN.items():
            gefunden = False

            for bezeichnung in bezeichnungen:
                muster = r"^\s*" + re.escape(bezeichnung) + r"\s*:?\s*(.*)$"

                treffer = re.match(
                    muster,
                    text,
                    flags=re.IGNORECASE
                )

                if treffer:
                    wert = treffer.group(1).strip()

                    ergebnis[excel_spalte] = wert
                    gefunden = True
                    break

            if gefunden:
                break
    # --------------------------------------------------------
    # C. LEAD FINDEN
    # --------------------------------------------------------

    lead_aktiv = False
    lead_gefunden = False
    lead_texte = []

    for p in absätze:
        text = text_ohne_formatierung(p).strip()

        if not text:
            continue

        # Formatierung der Textteile prüfen
        runs_mit_text = [
            run for run in p.runs
            if run.text.strip()
        ]

        ist_kursiv = (
                runs_mit_text
                and all(run.italic for run in runs_mit_text)
        )

        ist_fett = (
                runs_mit_text
                and all(run.bold for run in runs_mit_text)
        )

        # Variante 1 und 2: Teaser-Zwischentitel erkennen
        if normalisieren(text).strip() in [
            "teaser",
            "teaser (400 zeichen inkl. leerzeichen)"
        ]:
            lead_aktiv = True
            lead_gefunden = True
            continue

        # Wenn ein Teaser begonnen hat:
        if lead_aktiv:

            # Fett-kursive Absätze gehören zum Lead!
            if ist_fett and ist_kursiv:
                lead_texte.append(text)
                continue

            # Andere fett formatierte Absätze beenden den Lead
            if absatz_ist_fett(p):
                lead_aktiv = False
                continue

            lead_texte.append(text)

        # Variante 3: Kursiver oder fett-kursiver Text ohne Zwischentitel
        elif not lead_gefunden and ist_kursiv:
            lead_texte.append(text)

    if lead_texte:
        ergebnis["Lead"] = "\n".join(lead_texte)
    # --------------------------------------------------------
    # D. DEFINITION, ANWENDUNGEN, HERAUSFORDERUNGEN
    # --------------------------------------------------------

    aktuelle_kategorie = None

    for p in absätze:

        text = text_ohne_formatierung(p)

        if not text:
            continue

        # Dokumenttitel überspringen
        if text == ergebnis["Titel"]:
            continue

        # Prüfen, ob der Absatz exakt einem der drei
        # festgelegten Hauptabschnittstitel entspricht
        gefundene_kategorie = abschnitt_zuordnen(text)

        if gefundene_kategorie:
            aktuelle_kategorie = gefundene_kategorie
            continue

        # Jeden anderen Absatz dem aktuellen Abschnitt zuordnen
        # Formatierung spielt dabei keine Rolle!
        if aktuelle_kategorie:
            if ergebnis[aktuelle_kategorie]:
                ergebnis[aktuelle_kategorie] += "\n"

            ergebnis[aktuelle_kategorie] += text

    return ergebnis


# ============================================================
# 4. ALLE WORD-DATEIEN REKURSIV DURCHSUCHEN
# ============================================================

auswertungen = []
kontrolle = []

if not HAUPTORDNER.exists():
    raise FileNotFoundError(
        f"Der Hauptordner wurde nicht gefunden: {HAUPTORDNER}"
    )

word_dateien = list(HAUPTORDNER.rglob("*.docx"))

# Temporäre Word-Dateien ausschliessen
word_dateien = [
    f for f in word_dateien
    if not f.name.startswith("~$")
    and f != EXCEL_DATEI
]

print(f"{len(word_dateien)} Word-Dateien gefunden.")


for datei in word_dateien:

    try:

        # Dokument auslesen
        daten = dokument_auslesen(datei)

        auswertungen.append(daten)

        # ----------------------------------------------------
        # Überschriften für das Kontrollblatt separat erkennen
        # ----------------------------------------------------

        doc = Document(datei)

        erkannte_titel = []

        for p in doc.paragraphs:

            if absatz_ist_kontrolltitel(p):

                erkannte_titel.append(
                    text_ohne_formatierung(p)
                )

        # Mehrfach vorkommende Überschriften zählen
        zaehlung = Counter(erkannte_titel)

        for ueberschrift, anzahl in zaehlung.items():

            kontrolle.append({
                "Dateiname": datei.name,
                "Überschrift": ueberschrift,
                "Anzahl": anzahl
            })

        print(f"Erfolgreich gelesen: {datei.name}")

    except Exception as e:

        print(f"FEHLER bei {datei.name}: {e}")


# ============================================================
# 5. EXCEL ERSTELLEN
# ============================================================

wb = Workbook()

# ------------------------------------------------------------
# Tabellenblatt 1: Auswertung
# ------------------------------------------------------------

ws = wb.active
ws.title = "Auswertung"

spalten = [
    "Dateiname",
    "Titel",
    "Untertitel",
    "Forschungsfeld",
    "Autor:in",
    "Expert:innen",
    "Lead",
    "Definition",
    "Anwendungen",
    "Herausforderungen",
    "Dateipfad"
]

ws.append(spalten)

for daten in auswertungen:

    ws.append([
        daten.get(spalte, "")
        for spalte in spalten
    ])


# ------------------------------------------------------------
# Tabellenblatt 2: Kontrolle der Zwischentitel
# ------------------------------------------------------------

ws2 = wb.create_sheet("Kontrolle_Titel")

ws2.append([
    "Dateiname",
    "Erkannte fett formatierte Überschrift",
    "Anzahl"
])

for eintrag in kontrolle:

    ws2.append([
        eintrag["Dateiname"],
        eintrag["Überschrift"],
        eintrag["Anzahl"]
    ])


# ============================================================
# 6. EXCEL FORMATIEREN
# ============================================================

for sheet in wb.worksheets:

    # Kopfzeile formatieren
    for cell in sheet[1]:

        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.fill = PatternFill(
            start_color="4472C4",
            end_color="4472C4",
            fill_type="solid"
        )

        cell.alignment = Alignment(
            vertical="center",
            wrap_text=True
        )

    # Spaltenbreiten anpassen
    for column in sheet.columns:

        max_length = 0

        column_letter = get_column_letter(
            column[0].column
        )

        for cell in column:

            if cell.value:

                max_length = max(
                    max_length,
                    len(str(cell.value))
                )

        sheet.column_dimensions[column_letter].width = min(
            max(max_length + 3, 15),
            60
        )

    # Filter aktivieren
    sheet.auto_filter.ref = sheet.dimensions

    # Erste Zeile fixieren
    sheet.freeze_panes = "A2"

    # Textumbruch für alle Zellen
    for row in sheet.iter_rows(min_row=2):

        for cell in row:

            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True
            )


# ============================================================
# 7. EXCEL SPEICHERN
# ============================================================

wb.save(EXCEL_DATEI)

print("\n======================================")
print("AUSWERTUNG ABGESCHLOSSEN")
print("======================================")
print(f"Gefundene Word-Dateien: {len(word_dateien)}")
print(f"Erfolgreich ausgewertet: {len(auswertungen)}")
print(f"Erkannte Überschriften: {len(kontrolle)}")
print(f"Excel-Datei: {EXCEL_DATEI}")
print("======================================")
