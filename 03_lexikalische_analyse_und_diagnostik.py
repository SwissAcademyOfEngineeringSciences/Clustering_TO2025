# -*- coding: utf-8 -*-
"""
03_lexikalische_analyse_und_diagnostik.py
=========================================

Inhaltliche Charakterisierung und Ergebnisdiagnostik der in Skript 02
mathematisch bestimmten k=9-Makrocluster.

Dieses Skript verändert KEINE Embeddings, UMAP-Koordinaten, HDBSCAN-Zuordnungen
oder Makrocluster. Es arbeitet ausschliesslich auf den Resultaten aus Skript 02.

Ablauf
------
1. k=9-Segment-/Mikro-/Makrozuordnung laden.
2. Lexikalische Aufbereitung nur für die Interpretation:
   NFC -> Gender-Suffix-Normalisierung -> spaCy -> NOUN/PROPN/ADJ ->
   Lemmatisierung -> deutsche stopwordsiso-Stopwörter.
3. Eigenes c-TF-IDF je Makrocluster.
4. Repräsentative Originalsegmente über Cosine Similarity zum Makrozentroid.
5. Technologie x Makrocluster: Counts, Anteil unter geclusterten Segmenten,
   Anteil unter ALLEN Segmenten, Expected Counts, Lift, Pearson-Residuals.
6. Heatmap Technologie x Makrocluster.
7. Top-Term-Balkendiagramme und Wordclouds.
8. Coverage-/Noise-/Entropie-/Dominanzdiagnostik aus dem früheren Skript 30.

Installation
------------
pip install pandas numpy matplotlib scikit-learn spacy stopwordsiso openpyxl wordcloud
python -m spacy download de_core_news_sm
"""

from pathlib import Path
from collections import Counter
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import spacy
from stopwordsiso import stopwords
from sklearn.preprocessing import normalize

# =====================================================================
# 0. PARAMETER
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CLUSTER_DIR = RESULTS_DIR / "02_semantisches_clustering" / "hauptanalyse"
OUT = RESULTS_DIR / "03_lexikalische_analyse"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_K = 9
SPACY_MODEL = "de_core_news_sm"
LEXICAL_POS = {"NOUN", "PROPN", "ADJ"}
USE_LEMMATIZATION = True
MAX_DF = 0.60
TOP_N_TERMS = 20
TOP_N_BAR = 10
TOP_N_WORDCLOUD = 20
TOP_N_REPRESENTATIVE_SEGMENTS = 12

BAR_COLOR = "#02AFA9"
HEATMAP_LOW_COLOR = "#02AFA9"
HEATMAP_HIGH_COLOR = "#163E91"
TECH_MACRO_CMAP = LinearSegmentedColormap.from_list(
    "satw_tech_macro", [HEATMAP_LOW_COLOR, HEATMAP_HIGH_COLOR]
)
DPI = 220
CSV_SEP = ";"
CSV_ENCODING = "utf-8-sig"

SEGMENT_FILE = CLUSTER_DIR / f"13_segment_mikro_makro_k{TARGET_K}.csv"
EMBEDDING_FILE = CLUSTER_DIR / "02_segment_embeddings_original.npy"

# =====================================================================
# 1. HILFSFUNKTIONEN
# =====================================================================
def nfc(x):
    if pd.isna(x):
        return ""
    return unicodedata.normalize("NFC", str(x)).strip()


def gender_normalize(text):
    return re.sub(r":(?:innen|in)\b", "", nfc(text), flags=re.IGNORECASE)


def lexical_tokens(text, nlp, sw):
    doc = nlp(gender_normalize(text))
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct or tok.pos_ not in LEXICAL_POS:
            continue
        term = tok.lemma_ if USE_LEMMATIZATION else tok.text
        term = nfc(term).strip().lower()
        if not term or len(term) < 2 or term in sw:
            continue
        if not any(ch.isalpha() for ch in term):
            continue
        out.append(term)
    return out


def own_ctfidf(class_token_docs, max_df=0.60, top_n=20):
    class_ids = sorted(class_token_docs)
    n_classes = len(class_ids)
    counters = {cid: Counter(class_token_docs[cid]) for cid in class_ids}
    lengths = {cid: sum(counters[cid].values()) for cid in class_ids}
    mean_len = float(np.mean([max(1, lengths[cid]) for cid in class_ids]))

    class_df = Counter()
    for cid in class_ids:
        for term in counters[cid]:
            class_df[term] += 1

    vocab_rows, allowed = [], set()
    for term, dfreq in sorted(class_df.items()):
        share = dfreq / n_classes
        keep = share <= max_df
        vocab_rows.append({
            "term": term, "macro_df": dfreq,
            "macro_df_share": share, "keep_max_df": keep
        })
        if keep:
            allowed.add(term)

    rows = []
    for cid in class_ids:
        L = max(1, lengths[cid])
        vals = []
        for term, count in counters[cid].items():
            if term not in allowed:
                continue
            tf = count / L
            idf = math.log(1.0 + mean_len / class_df[term])
            score = tf * idf
            vals.append((term, count, class_df[term], tf, idf, score))
        vals.sort(key=lambda z: (-z[5], -z[1], z[0]))
        for rank, (term, count, dfreq, tf, idf, score) in enumerate(vals[:top_n], 1):
            rows.append({
                "macro_cluster": cid, "rank": rank, "term": term,
                "count_in_macro": count, "macro_df": dfreq,
                "tf": tf, "idf": idf, "ctfidf": score
            })
    return pd.DataFrame(rows), pd.DataFrame(vocab_rows)


def normalized_entropy(vals):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a) & (a > 0)]
    if len(a) <= 1:
        return 0.0
    p = a / a.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def corr(a, b, method):
    z = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b, method=method))


def macro_number(x):
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else 999999


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def scatter(df, x, y, xlab, ylab, title, filename):
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.scatter(df[x], df[y], s=55, alpha=.8, color=BAR_COLOR)
    for r in df.nsmallest(5, "coverage").itertuples():
        xv, yv = getattr(r, x), getattr(r, y)
        if pd.notna(xv) and pd.notna(yv):
            ax.annotate(r.technology, (xv, yv), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=.2)
    if x in ["coverage", "noise_share"]:
        ax.set_xlim(-.03, 1.03)
    if y in ["max_macro_share_clustered", "macro_entropy_clustered"]:
        ax.set_ylim(-.03, 1.03)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 2. INPUT
# =====================================================================
print("[1/7] Lade mathematisch bestimmte Clusterstruktur ...")
if not SEGMENT_FILE.exists() or not EMBEDDING_FILE.exists():
    raise FileNotFoundError(
        "Outputs aus 02_semantisches_clustering.py fehlen. "
        "Bitte zuerst Skript 02 ausführen."
    )

seg = pd.read_csv(SEGMENT_FILE, sep=CSV_SEP, encoding=CSV_ENCODING)
emb = np.load(EMBEDDING_FILE).astype(float)
emb = normalize(emb, norm="l2")

if len(seg) != len(emb):
    raise RuntimeError("Segmentdatei und Embedding-Matrix haben unterschiedliche Länge.")

clustered_mask = seg["cluster"].to_numpy() >= 0
clustered = seg.loc[clustered_mask].copy().reset_index(drop=True)
clustered_emb = emb[clustered_mask]

if clustered["macro_cluster"].isna().any():
    raise RuntimeError("Geclusterte Segmente ohne Makroclusterzuordnung gefunden.")

macros = sorted(clustered["macro_cluster"].unique(), key=macro_number)
technologies = sorted(seg["technology"].dropna().unique())

# =====================================================================
# 3. c-TF-IDF
# =====================================================================
print("[2/7] Lexikalische Charakterisierung mit c-TF-IDF ...")
try:
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
except OSError as e:
    raise RuntimeError(
        f"spaCy-Modell {SPACY_MODEL!r} fehlt. "
        f"Installieren: python -m spacy download {SPACY_MODEL}"
    ) from e
nlp.max_length = max(nlp.max_length, 3_000_000)
sw = set(stopwords("de"))

token_docs = {}
for macro in macros:
    joined = "\n".join(
        clustered.loc[clustered["macro_cluster"] == macro, "segment_text"].astype(str)
    )
    token_docs[macro] = lexical_tokens(joined, nlp, sw)

terms, vocab = own_ctfidf(token_docs, max_df=MAX_DF, top_n=TOP_N_TERMS)
terms.to_csv(OUT / "01_macro_top_terms_ctfidf.csv",
             index=False, sep=CSV_SEP, encoding=CSV_ENCODING)
vocab.to_csv(OUT / "02_macro_vokabular_qc.csv",
             index=False, sep=CSV_SEP, encoding=CSV_ENCODING)

# =====================================================================
# 4. REPRÄSENTATIVE ORIGINALSEGMENTE
# =====================================================================
print("[3/7] Bestimme repräsentative Originalsegmente ...")
rep_rows, macro_diag_rows = [], []
for macro in macros:
    mask = clustered["macro_cluster"].to_numpy() == macro
    X = clustered_emb[mask]
    g = clustered.loc[mask].reset_index(drop=True)
    c = X.mean(axis=0)
    c /= max(np.linalg.norm(c), 1e-12)
    sims = X @ c
    order = np.argsort(-sims)

    for rank, pos in enumerate(order[:TOP_N_REPRESENTATIVE_SEGMENTS], 1):
        r = g.iloc[pos]
        rep_rows.append({
            "macro_cluster": macro,
            "rank": rank,
            "cosine_to_macro_centroid": float(sims[pos]),
            "micro_cluster": f"C{int(r['cluster'])}",
            "technology": r["technology"],
            "field": r["field"],
            "segment_id": r["segment_id"],
            "segment_text": r["segment_text"],
        })

    tv = g["technology"].value_counts()
    macro_diag_rows.append({
        "macro_cluster": macro,
        "n_segments": len(g),
        "n_technologies": tv.size,
        "largest_technology": str(tv.index[0]),
        "largest_technology_dominance": float(tv.iloc[0] / len(g)),
        "technology_entropy": normalized_entropy(tv.values),
        "mean_cosine_to_macro_centroid": float(np.mean(sims)),
        "member_microclusters": " | ".join(
            f"C{x}" for x in sorted(g["cluster"].astype(int).unique())
        ),
    })

reps = pd.DataFrame(rep_rows)
macro_diag = pd.DataFrame(macro_diag_rows)
reps.to_csv(OUT / "03_repraesentative_segmente.csv",
            index=False, sep=CSV_SEP, encoding=CSV_ENCODING)
macro_diag.to_csv(OUT / "04_makrocluster_diagnostik.csv",
                  index=False, sep=CSV_SEP, encoding=CSV_ENCODING)

# =====================================================================
# 5. TECHNOLOGIE x MAKROCLUSTER + ASSOZIATIONEN
# =====================================================================
print("[4/7] Berechne Technologie-Makrocluster-Profile und Assoziationen ...")
counts = pd.crosstab(
    clustered["technology"], clustered["macro_cluster"]
).reindex(index=technologies, columns=macros, fill_value=0)

shares_clustered = counts.div(
    counts.sum(axis=1).replace(0, np.nan), axis=0
).fillna(0)

total_all = seg["technology"].value_counts().reindex(technologies, fill_value=0)
shares_all = counts.div(total_all.replace(0, np.nan), axis=0).fillna(0)

row_totals = counts.sum(axis=1).astype(float)
col_totals = counts.sum(axis=0).astype(float)
grand_total = float(counts.to_numpy().sum())
expected = pd.DataFrame(
    np.outer(row_totals.to_numpy(), col_totals.to_numpy()) / grand_total,
    index=counts.index, columns=counts.columns
)
macro_base_rate = col_totals / grand_total
lift = shares_clustered.div(
    macro_base_rate.replace(0, np.nan), axis=1
).replace([np.inf, -np.inf], np.nan).fillna(0)
pearson_residual = (
    (counts.astype(float) - expected) /
    np.sqrt(expected.replace(0, np.nan))
).replace([np.inf, -np.inf], np.nan).fillna(0)

counts.to_csv(OUT / "05_technologie_macro_counts.csv",
              sep=CSV_SEP, encoding=CSV_ENCODING)
shares_clustered.to_csv(OUT / "06_technologie_macro_anteil_geclustert.csv",
                        sep=CSV_SEP, encoding=CSV_ENCODING)
shares_all.to_csv(OUT / "07_technologie_macro_anteil_alle_segmente.csv",
                  sep=CSV_SEP, encoding=CSV_ENCODING)
expected.to_csv(OUT / "08_erwartete_counts_unabhaengigkeit.csv",
                sep=CSV_SEP, encoding=CSV_ENCODING)
lift.to_csv(OUT / "09_lift_technologie_macro.csv",
            sep=CSV_SEP, encoding=CSV_ENCODING)
pearson_residual.to_csv(OUT / "10_pearson_residuals_technologie_macro.csv",
                        sep=CSV_SEP, encoding=CSV_ENCODING)

with pd.ExcelWriter(OUT / "11_technologie_macro_kennzahlen.xlsx",
                    engine="openpyxl") as writer:
    counts.to_excel(writer, sheet_name="Counts")
    shares_clustered.to_excel(writer, sheet_name="Anteil_geclustert")
    shares_all.to_excel(writer, sheet_name="Anteil_alle")
    expected.to_excel(writer, sheet_name="Erwartete_Counts")
    lift.to_excel(writer, sheet_name="Lift")
    pearson_residual.to_excel(writer, sheet_name="Pearson_Residual")

# Primäre Heatmap: Anteil an geclusterten Segmenten, wie in Skript 27.
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(
    shares_clustered.to_numpy(), aspect="auto", vmin=0, vmax=1,
    cmap=TECH_MACRO_CMAP
)
ax.set_xticks(range(len(macros)))
ax.set_xticklabels(macros)
ax.set_yticks(range(len(technologies)))
ax.set_yticklabels(technologies)
ax.set_xlabel("Makrocluster")
ax.set_ylabel("Technologie")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Anteil")
fig.tight_layout()
fig.savefig(OUT / "12_heatmap_technologie_macro_geclustert.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)

# Konservativere Heatmap mit Noise im Nenner.
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(
    shares_all.to_numpy(), aspect="auto", vmin=0, vmax=1,
    cmap=TECH_MACRO_CMAP
)
ax.set_xticks(range(len(macros)))
ax.set_xticklabels(macros)
ax.set_yticks(range(len(technologies)))
ax.set_yticklabels(technologies)
ax.set_xlabel("Makrocluster")
ax.set_ylabel("Technologie")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Anteil an allen Segmenten")
fig.tight_layout()
fig.savefig(OUT / "13_heatmap_technologie_macro_alle_segmente.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 6. TOP-TERMS UND WORDCLOUDS
# =====================================================================
print("[5/7] Erzeuge Top-Term-Grafiken und Wordclouds ...")
plot_dir = OUT / "topterms_und_wordclouds"
plot_dir.mkdir(exist_ok=True)

for macro in macros:
    sub = terms[terms["macro_cluster"].astype(str) == str(macro)].copy()
    d = sub.nsmallest(TOP_N_BAR, "rank").sort_values("ctfidf", ascending=True)
    fig_h = max(5.0, 0.48 * len(d) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    ax.barh(d["term"], d["ctfidf"], color=BAR_COLOR)
    ax.set_xlabel("c-TF-IDF")
    ax.set_ylabel("Term")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"bar_{safe_name(macro)}_top{TOP_N_BAR}_ctfidf.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    try:
        from wordcloud import WordCloud
        dw = sub.nsmallest(TOP_N_WORDCLOUD, "rank")
        frequencies = {
            str(r.term): float(r.ctfidf)
            for r in dw.itertuples()
            if pd.notna(r.ctfidf) and float(r.ctfidf) > 0
        }
        if frequencies:
            wc = WordCloud(
                width=1600, height=900, background_color="white",
                collocations=False, prefer_horizontal=.9, random_state=42
            ).generate_from_frequencies(frequencies)
            fig, ax = plt.subplots(figsize=(12, 6.75))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(plot_dir / f"wordcloud_{safe_name(macro)}.png",
                        dpi=DPI, bbox_inches="tight")
            plt.close(fig)
    except ImportError:
        print("Hinweis: Paket 'wordcloud' fehlt; Wordclouds werden übersprungen.")

# Kombinierte Top-Term-Plots mit gemeinsamer x-Skala.
top_all = terms[terms["rank"] <= TOP_N_BAR].copy()
global_max = float(top_all["ctfidf"].max()) if len(top_all) else 1.0
global_xlim = max(global_max * 1.12, 1e-12)

def combined_plot(cluster_list, nrows, ncols, filename):
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(16, 6.5*nrows),
        sharex=True, sharey=False
    )
    axes = list(np.atleast_1d(axes).flat)
    for ax, macro in zip(axes, cluster_list):
        d = terms[
            (terms["macro_cluster"].astype(str) == str(macro)) &
            (terms["rank"] <= TOP_N_BAR)
        ].copy().sort_values("ctfidf", ascending=True)
        ax.barh(d["term"], d["ctfidf"], color=BAR_COLOR)
        ax.set_title(f"{macro}: Top {len(d)}", fontsize=13, fontweight="bold")
        ax.set_xlim(0, global_xlim)
        ax.set_xlabel("c-TF-IDF")
        ax.set_ylabel("Term")
        ax.grid(axis="x", alpha=.2)
        ax.tick_params(axis="y", labelleft=True, labelsize=9)
    for ax in axes[len(cluster_list):]:
        ax.axis("off")
    fig.tight_layout(h_pad=2.2, w_pad=5.0)
    fig.savefig(plot_dir / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

combined_plot(macros[:6], 3, 2, "combined_M0_M5_topterms.png")
if len(macros) > 6:
    combined_plot(macros[6:9], 2, 2, "combined_M6_M8_topterms.png")

# =====================================================================
# 7. COVERAGE- UND THEMENPROFIL-DIAGNOSTIK
# =====================================================================
print("[6/7] Erzeuge Coverage-/Noise-/Entropie-Diagnostik ...")
total = seg.groupby("technology").size().rename("n_segments_total")
cl = seg[seg["cluster"] >= 0].groupby("technology").size().rename("n_segments_clustered")
no = seg[seg["cluster"] < 0].groupby("technology").size().rename("n_segments_noise")
td = pd.concat([total, cl, no], axis=1).fillna(0).astype(int)
td["coverage"] = td.n_segments_clustered / td.n_segments_total
td["noise_share"] = td.n_segments_noise / td.n_segments_total
td["n_macrothemes_represented"] = (counts > 0).sum(axis=1)
td["dominant_macrotheme"] = shares_clustered.idxmax(axis=1)
td["max_macro_share_clustered"] = shares_clustered.max(axis=1)
td["macro_entropy_clustered"] = shares_clustered.apply(
    lambda r: normalized_entropy(r.values), axis=1
)
zero = td.n_segments_clustered == 0
td.loc[zero, ["dominant_macrotheme","max_macro_share_clustered",
             "macro_entropy_clustered"]] = [None, np.nan, np.nan]
td["dominant_macro_share_all_segments"] = [
    np.nan if pd.isna(td.loc[t, "dominant_macrotheme"])
    else shares_all.loc[t, td.loc[t, "dominant_macrotheme"]]
    for t in td.index
]
td = td.reset_index().rename(columns={"index": "technology"}).sort_values(
    ["coverage", "technology"]
)

pairs = [
    ("coverage", "max_macro_share_clustered"),
    ("coverage", "macro_entropy_clustered"),
    ("noise_share", "max_macro_share_clustered"),
    ("noise_share", "macro_entropy_clustered"),
]
cr = pd.DataFrame([
    {
        "x": x, "y": y,
        "pearson_r": corr(td[x], td[y], "pearson"),
        "spearman_rho": corr(td[x], td[y], "spearman"),
        "n": int((td[x].notna() & td[y].notna()).sum()),
    }
    for x, y in pairs
])

td.to_csv(OUT / "14_technologie_coverage_diagnostik.csv",
          index=False, sep=CSV_SEP, encoding=CSV_ENCODING)
cr.to_csv(OUT / "15_coverage_zusammenhangsdiagnostik.csv",
          index=False, sep=CSV_SEP, encoding=CSV_ENCODING)

scatter(
    td, "coverage", "max_macro_share_clustered",
    "Mikrocluster-Coverage",
    "Grösster Makrothemenanteil (geclusterte Segmente)",
    "Coverage und thematische Konzentration",
    "16_coverage_vs_max_themenanteil.png"
)
scatter(
    td, "coverage", "macro_entropy_clustered",
    "Mikrocluster-Coverage", "Normalisierte Makrothemenentropie",
    "Coverage und Diversität des Themenprofils",
    "17_coverage_vs_themenentropie.png"
)
scatter(
    td, "noise_share", "max_macro_share_clustered",
    "Noise-Anteil",
    "Grösster Makrothemenanteil (geclusterte Segmente)",
    "Noise und thematische Konzentration",
    "18_noise_vs_max_themenanteil.png"
)

p = td.sort_values("coverage")
fig, ax = plt.subplots(figsize=(10, max(8, .34*len(p)+2)))
ax.barh(p.technology, p.coverage, color=BAR_COLOR)
ax.set_xlim(0, 1)
ax.set_xlabel("Anteil geclusterter Segmente an allen Segmenten")
ax.set_ylabel("Technologie")
ax.grid(axis="x", alpha=.2)
fig.tight_layout()
fig.savefig(OUT / "19_coverage_je_technologie.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(
    macro_diag["largest_technology_dominance"],
    macro_diag["technology_entropy"],
    s=70, color=BAR_COLOR
)
for r in macro_diag.itertuples():
    ax.annotate(
        r.macro_cluster,
        (r.largest_technology_dominance, r.technology_entropy),
        xytext=(5,5), textcoords="offset points", fontsize=9
    )
ax.set_xlim(-.03, 1.03)
ax.set_ylim(-.03, 1.03)
ax.set_xlabel("Largest-Tech-Dominance")
ax.set_ylabel("Normalisierte Technologieentropie")
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT / "20_makrocluster_breite_dominanz.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)

with pd.ExcelWriter(OUT / "21_diagnostik_gesamt.xlsx", engine="openpyxl") as w:
    td.to_excel(w, sheet_name="Technologie_Diagnostik", index=False)
    macro_diag.to_excel(w, sheet_name="Makro_Diagnostik", index=False)
    counts.to_excel(w, sheet_name="Counts")
    shares_clustered.to_excel(w, sheet_name="Anteil_geclustert")
    shares_all.to_excel(w, sheet_name="Anteil_alle_Segmente")
    cr.to_excel(w, sheet_name="Zusammenhaenge", index=False)
    terms.to_excel(w, sheet_name="cTFIDF_TopTerms", index=False)
    reps.to_excel(w, sheet_name="Rep_Segmente", index=False)

params = {
    "target_k": TARGET_K,
    "lexical": {
        "spacy_model": SPACY_MODEL,
        "pos": sorted(LEXICAL_POS),
        "lemmatization": USE_LEMMATIZATION,
        "german_stopwords": "stopwordsiso",
        "gender_suffix_normalization": True,
        "max_df": MAX_DF,
        "top_n_terms": TOP_N_TERMS,
    },
    "ctfidf": "TF within aggregated macrocluster text * log(1 + mean_class_length / macro_df)",
    "important": "Lexical analysis characterizes existing clusters and does not alter clustering.",
}
with open(OUT / "00_parameter_lexikalische_analyse.json", "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False, indent=2)

print("[7/7] FERTIG.")
print(f"Resultate: {OUT}")
print("Die Clusterstruktur wurde in diesem Skript nicht verändert.")
