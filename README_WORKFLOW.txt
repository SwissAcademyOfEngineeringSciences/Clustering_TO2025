SATW FINAL WORKFLOW
===================

Ordnerstruktur nach einem vollständigen Lauf:

.
├── 01_scrape_satw.py
├── 02_semantisches_clustering.py
├── 03_lexikalische_analyse_und_diagnostik.py
└── results/
    ├── 01_scraping/
    ├── 02_semantisches_clustering/
    └── 03_lexikalische_analyse/

Reihenfolge:
1. 01_scrape_satw.py
2. 02_semantisches_clustering.py
3. 03_lexikalische_analyse_und_diagnostik.py

Für Aufsetzen einer neuen virtuellen Python Umgebung, Installation aller Pakete über konsole:
python -m pip install requests beautifulsoup4 pandas numpy matplotlib scipy scikit-learn sentence-transformers umap-learn hdbscan spacy stopwordsiso openpyxl wordcloud
python -m spacy download de_core_news_sm

Test, ob Umgebung bereit ist, über Konsole:
python -c "import pandas, numpy, spacy, hdbscan, umap, sentence_transformers; print('Umgebung OK')"


Methodische Logik:
- Skript 02 reproduziert den Textfeldvergleich A-E und kürzt die Entscheidung
  für B_CHANCEN_HERAUSF nicht ab.
- HDBSCAN erzeugt Mikrocluster ohne vorgegebenes k.
- Die Makrocluster werden semantisch aus Mikrocluster-Zentroiden konstruiert.
- k=4...10 wird vollständig verglichen; k=9 wird nicht automatisch "optimiert",
  sondern als anschliessend interpretierte Lösung exportiert.
- Skript 03 verändert keine Clusterstruktur mehr.
