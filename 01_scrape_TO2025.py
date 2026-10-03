#!/usr/bin/env python3
"""Scrape SATW Technology Outlook technology pages into a CSV.

Install:
    pip install requests beautifulsoup4

Run:
    python scrape_satw_technologies_v2.py

Output:
    satw_technologien.csv
"""

import csv
import re
import time
from pathlib import Path
from collections import OrderedDict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

START_URL = "https://technology-outlook.satw.ch/de/technologien-im-fokus"
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "01_scraping"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = RESULTS_DIR / "satw_technologien.csv"
TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 0.5

# Gewünschte Anzahl H2-Spalten je Content-Section
FIRST_SECTION_H2_COUNT = 6
SECOND_SECTION_H2_COUNT = 4

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    )
}


def clean_text(value: str) -> str:
    """Whitespace vereinheitlichen, Text aber in einer einzigen CSV-Zelle lassen."""
    return re.sub(r"\s+", " ", value or "").strip()


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_technology_links(session: requests.Session) -> list[str]:
    """Links aus div.teaser-item > a.full-link sammeln."""
    soup = get_soup(session, START_URL)
    links = []

    for teaser in soup.select("div.teaser-item"):
        link = teaser.select_one("a.full-link[href]")
        if not link:
            continue

        absolute_url = urljoin(START_URL, link["href"])
        if absolute_url not in links:
            links.append(absolute_url)

    return links


def find_content_sections(soup: BeautifulSoup) -> list[Tag]:
    """Die ersten zwei Content-Sections mit IDs wie c12345 finden.

    Bevorzugt werden Sections, die eine .text-holder enthalten. Falls die
    Website-Struktur leicht variiert, werden als Fallback alle passenden
    section[id=c#####] verwendet.
    """
    id_pattern = re.compile(r"^c\d{5}$")
    candidates = [
        section
        for section in soup.find_all("section", id=id_pattern)
        if isinstance(section, Tag)
    ]

    with_text_holder = [s for s in candidates if s.select_one(".text-holder")]
    if len(with_text_holder) >= 2:
        return with_text_holder[:2]

    return candidates[:2]


def extract_h2_paragraph_sections(container: Tag | None) -> OrderedDict[str, str]:
    """Jede H2 plus alle folgenden <p> bis zur nächsten H2 extrahieren.

    Nur <p>-Inhalte werden übernommen, wie gewünscht. Semikolons und andere
    Satzzeichen bleiben Teil des Textes und damit derselben CSV-Zelle.
    """
    result: OrderedDict[str, list[str]] = OrderedDict()
    current_heading = None

    if container is None:
        return OrderedDict()

    # h2 und p in DOM-Reihenfolge durchlaufen.
    for element in container.find_all(["h2", "p"]):
        if element.name == "h2":
            heading = clean_text(element.get_text(" ", strip=True))
            if heading:
                current_heading = heading
                result.setdefault(current_heading, [])
            continue

        if element.name == "p" and current_heading:
            paragraph = clean_text(element.get_text(" ", strip=True))
            if paragraph:
                result[current_heading].append(paragraph)

    return OrderedDict(
        (heading, " ".join(paragraphs))
        for heading, paragraphs in result.items()
    )


def scrape_technology(session: requests.Session, url: str) -> dict:
    soup = get_soup(session, url)

    h1 = soup.find("h1")
    lead = soup.select_one(".intro-paragraph")

    row = {
        "title": clean_text(h1.get_text(" ", strip=True)) if h1 else "",
        # ALLES aus intro-paragraph bleibt ein String/eine Zelle.
        "lead": clean_text(lead.get_text(" ", strip=True)) if lead else "",
        "url": url,
    }

    sections = find_content_sections(soup)

    # Erste Section: bis zu 6 H2-Spalten
    if len(sections) >= 1:
        first_holder = sections[0].select_one(".text-holder") or sections[0]
        first_data = extract_h2_paragraph_sections(first_holder)
        for heading, text in list(first_data.items())[:FIRST_SECTION_H2_COUNT]:
            row[heading] = text

    # Zweite Section: bis zu 4 H2-Spalten
    if len(sections) >= 2:
        second_holder = sections[1].select_one(".text-holder") or sections[1]
        second_data = extract_h2_paragraph_sections(second_holder)
        for heading, text in list(second_data.items())[:SECOND_SECTION_H2_COUNT]:
            # Falls zufällig dieselbe H2 in beiden Sections vorkommt, nichts
            # überschreiben, sondern eindeutig benennen.
            key = heading if heading not in row else f"{heading} (2)"
            row[key] = text

    return row


def build_fieldnames(rows: list[dict]) -> list[str]:
    """title + lead + 6 H2 aus Section 1 + 4 H2 aus Section 2 + url.

    Die echten H2-Texte werden als Spaltennamen verwendet. Wir sammeln die
    Reihenfolge anhand der Seiten. Da die SATW-Artikel grundsätzlich dieselbe
    Struktur verwenden, ergeben sich dadurch die gewünschten Spalten.
    """
    fixed = ["title", "lead"]
    dynamic = []

    # Maximal 10 Inhalts-Spalten insgesamt.
    for row in rows:
        for key in row:
            if key in {"title", "lead", "url"}:
                continue
            if key not in dynamic:
                dynamic.append(key)
            if len(dynamic) >= FIRST_SECTION_H2_COUNT + SECOND_SECTION_H2_COUNT:
                break
        if len(dynamic) >= FIRST_SECTION_H2_COUNT + SECOND_SECTION_H2_COUNT:
            break

    return fixed + dynamic + ["url"]


def main() -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"Lese Übersichtsseite: {START_URL}")
    links = get_technology_links(session)
    print(f"Gefundene Technologie-Links: {len(links)}")

    if not links:
        raise RuntimeError(
            "Keine Technologie-Links gefunden. Prüfe div.teaser-item und a.full-link."
        )

    rows = []
    for index, url in enumerate(links, start=1):
        print(f"[{index}/{len(links)}] {url}")
        try:
            row = scrape_technology(session, url)
            rows.append(row)
            content_cols = [k for k in row if k not in {"title", "lead", "url"}]
            print(f"    -> {len(content_cols)} H2-Bereiche gefunden")
        except requests.RequestException as exc:
            print(f"    FEHLER: {exc}")

        time.sleep(REQUEST_DELAY_SECONDS)

    fieldnames = build_fieldnames(rows)

    # Semikolon als CSV-Trennzeichen ist für deutschsprachiges Excel praktisch.
    # QUOTE_ALL ist entscheidend: Ein Semikolon IM TEXT bleibt sicher innerhalb
    # derselben Zelle, z.B. "Teil 1; Teil 2" -> "Teil 1; Teil 2".
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            delimiter=";",
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    print(f"\nFertig: {OUTPUT_CSV}")
    print(f"Spalten ({len(fieldnames)}):")
    for field in fieldnames:
        print(f"  - {field}")


if __name__ == "__main__":
    main()
