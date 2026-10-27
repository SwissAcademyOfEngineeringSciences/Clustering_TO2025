# -*- coding: utf-8 -*-
"""
02_semantisches_clustering.py
=============================

Reproduzierbarer semantischer Hauptworkflow nach dem Scraping.

Ablauf
------
1. SATW-Texte laden.
2. Definition, "Heutige Anwendungen und Chancen" und "Herausforderungen"
   satzweise segmentieren. Kein nicht-leeres Segment wird verworfen.
3. Tokenlimit des Embedding-Modells prüfen; bei Überschreitung Abbruch statt Truncation.
4. Alle Segmente EINMAL mit Alibaba-NLP/gte-multilingual-base einbetten.
5. Textfeldentscheidung transparent reproduzieren:
      A_CORE
      B_CHANCEN_HERAUSF
      C_CHANCEN
      D_HERAUSFORDERUNGEN
      E_DEFINITION
   Für jede Variante: identische UMAP- und HDBSCAN-Parameter + Diagnostik.
   Es wird KEINE Variante automatisch gewählt.
6. Die im Forschungsprozess begründete Hauptvariante
   B_CHANCEN_HERAUSF explizit als Hauptanalyse weiterführen.
7. HDBSCAN-Mikrocluster dieser Variante als Ausgangspunkt verwenden.
8. Mikrocluster-Zentroide im ORIGINALEN Embeddingraum berechnen.
9. Semantische Cosine-Distanzen zwischen Mikrocluster-Zentroiden berechnen.
10. Hierarchische agglomerative Konsolidierung mit Average Linkage.
11. k = 4 ... 10 systematisch vergleichen; KEIN automatisches "bestes k".
12. Die für die Ergebnisanalyse gewählte Lösung k=9 zusätzlich exportieren.

Wichtig
-------
- HDBSCAN bestimmt die Anzahl Mikrocluster ohne vorgegebenes k.
- Die Makroclusterzahl wird NICHT automatisch gewählt.
- k=4...10 bleibt als methodischer Vergleich erhalten.
- TARGET_K=9 ist die anschliessend inhaltlich interpretierte Lösung, nicht das
  Resultat einer automatischen Optimierung.
- Technologieprofile werden NICHT zur Konstruktion der Makrocluster verwendet.
  Die Makrostruktur basiert zu 100 % auf Semantik.
- Titel dienen nur als Identifikatoren und gehen nicht in die Embeddings ein.

Installation
------------
pip install pandas numpy matplotlib scipy scikit-learn sentence-transformers umap-learn hdbscan spacy
python -m spacy download de_core_news_sm
"""

from pathlib import Path
import json
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 0. PARAMETER
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
INPUT_CSV = RESULTS_DIR / "01_scraping" / "satw_technologien.csv"
OUT = RESULTS_DIR / "02_semantisches_clustering"
OUT.mkdir(parents=True, exist_ok=True)

FIELD_DEFINITION = "Definition"
FIELD_CHANCEN = "Heutige Anwendungen und Chancen"
FIELD_HERAUSF = "Herausforderungen"
ALL_FIELDS = [FIELD_DEFINITION, FIELD_CHANCEN, FIELD_HERAUSF]

VARIANTS = {
    "A_CORE": [FIELD_DEFINITION, FIELD_CHANCEN, FIELD_HERAUSF],
    "B_CHANCEN_HERAUSF": [FIELD_CHANCEN, FIELD_HERAUSF],
    "C_CHANCEN": [FIELD_CHANCEN],
    "D_HERAUSFORDERUNGEN": [FIELD_HERAUSF],
    "E_DEFINITION": [FIELD_DEFINITION],
}
MAIN_VARIANT = "B_CHANCEN_HERAUSF"

##########################
# Sprachmodell Parameter #
##########################
# Festlegen des Embedding Modells
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
# Festlegen des Deutsches spaCy-Sprachmodell zur Erkennung von Satzgrenzen.
SPACY_MODEL = "de_core_news_sm"
# Legt Zufallswert für reproduzierbare Ergebnisse fest → heikel
RANDOM_SEED = 42
# Dokmentiert, dass Texte nicht automatisch abgeschnitten werden sollen
ALLOW_TRUNCATION = False

##################
# UMAP-Parameter #
##################
# Anzahl Nachbaren die UMAP zur Erfassung der lokalen Struktur berücksichtigt → Fokus auf lokal vs global
UMAP_N_NEIGHBORS = 15
# Anzahl Dimensionen nach der Reduktion
UMAP_N_COMPONENTS = 5
# Mindestabstand, den UMAP bei der Einbettung ähnlichen Punkte anstrebt → je tiefer desto kompaktere Anordnung ähnlicher Punkte
UMAP_MIN_DIST = 0.0
# Verwendetes Distanzmass für die ursprünlichen Embeddings
UMAP_METRIC = "cosine"

#####################
# HDBSCAN Parameter #
#####################
# Mindestgrösse eines Clusters → Untergrenze an Anzahl Embeddings
HDBSCAN_MIN_CLUSTER_SIZE = 15
# Parameter für die Dichteschätzung (tiefere Werte = Mehr Punkte können Clustern zugeordnet werden)
HDBSCAN_MIN_SAMPLES = 2
# Verwendetes Distanzmass für HDBSCAN
HDBSCAN_METRIC = "euclidean"
# Verwendete Methode zur Auswahl stabiler Cluster
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

LINKAGE_METHOD = "average"
K_VALUES = list(range(4, 11))
TARGET_K = 9

BAR_COLOR = "#02AFA9"
DPI = 220
CSV_SEP = ";"
CSV_ENCODING = "utf-8-sig"

# =====================================================================
# 1. HILFSFUNKTIONEN
# =====================================================================
def nfc(x):
    if pd.isna(x):
        return ""
    return unicodedata.normalize("NFC", str(x)).strip()


def get_max_seq_length(model):
    vals = []
    try:
        vals.append(int(model.max_seq_length))
    except Exception:
        pass
    try:
        vals.append(int(model.tokenizer.model_max_length))
    except Exception:
        pass
    vals = [v for v in vals if 0 < v < 1_000_000]
    if not vals:
        raise RuntimeError("Kein sinnvolles Tokenlimit des Modells gefunden.")
    return min(vals)


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def build_segments(df, nlp, model):
    tokenizer = model.tokenizer
    max_len = get_max_seq_length(model)
    rows = []
    seg_id = 0
    for row_idx, row in df.iterrows():
        technology = nfc(row["title"])
        for field in ALL_FIELDS:
            text = nfc(row[field])
            if not text:
                continue
            doc = nlp(text)
            sentence_no = 0
            for sent in doc.sents:
                raw = nfc(sent.text)
                if not raw:
                    continue
                sentence_no += 1
                seg_id += 1
                tc = token_count(tokenizer, raw)
                rows.append({
                    "segment_id": f"S{seg_id:05d}",
                    "technology": technology,
                    "source_row": row_idx,
                    "field": field,
                    "sentence_no": sentence_no,
                    "n_chars": len(raw),
                    "n_words": len(re.findall(r"\b\w+\b", raw, flags=re.UNICODE)),
                    "n_model_tokens": tc,
                    "model_max_seq_length": max_len,
                    "exceeds_model_limit": tc > max_len,
                    "segment_text": raw,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Keine Segmente gefunden.")
    too_long = out[out["exceeds_model_limit"]]
    if len(too_long):
        raise RuntimeError(
            f"ABBRUCH: {len(too_long)} Segment(e) über Modelllimit {max_len}. "
            "Es wird nicht trunciert oder automatisch geteilt.\n" +
            too_long[["segment_id","technology","field","n_model_tokens"]]
            .head(20).to_string(index=False)
        )
    return out


def normalized_entropy(counts):
    x = np.asarray(counts, dtype=float)
    x = x[x > 0]
    if len(x) <= 1:
        return 0.0
    p = x / x.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def centroid_cohesion(X, labels):
    vals = {}
    for cid in sorted(set(labels)):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        Xi = X[idx]
        c = Xi.mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-12)
        vals[cid] = float(np.mean(Xi @ c))
    return vals


def run_variant(name, fields, all_segments, all_embeddings, n_technologies):
    outdir = OUT / "textfeldvergleich" / name
    outdir.mkdir(parents=True, exist_ok=True)

    mask = all_segments["field"].isin(fields).to_numpy()
    seg = all_segments.loc[mask].copy().reset_index(drop=True)
    emb = all_embeddings[mask]

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED,
        transform_seed=RANDOM_SEED,
    )
    Z = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(Z)
    probs = clusterer.probabilities_

    seg["cluster"] = labels
    seg["hdbscan_probability"] = probs
    for d in range(Z.shape[1]):
        seg[f"umap_{d+1}"] = Z[:, d]
    seg.to_csv(outdir / "01_segment_clusterzuordnung.csv",
               index=False, sep=CSV_SEP, encoding=CSV_ENCODING)
    np.save(outdir / "02_umap_coordinates.npy", Z)

    valid = labels >= 0
    n_clusters = int(len(set(labels[valid]))) if valid.any() else 0
    noise_n = int((labels == -1).sum())
    cohesion = centroid_cohesion(emb, labels)

    diag_rows = []
    for cid in sorted(set(labels)):
        if cid < 0:
            continue
        g = seg[seg["cluster"] == cid]
        tv = g["technology"].value_counts()
        diag_rows.append({
            "variant": name,
            "cluster": cid,
            "n_segments": len(g),
            "n_technologies": tv.size,
            "technology_coverage_share": tv.size / n_technologies,
            "largest_single_technology_share": float(tv.iloc[0] / len(g)),
            "technology_entropy_normalized": normalized_entropy(tv.values),
            "mean_hdbscan_probability": float(g["hdbscan_probability"].mean()),
            "embedding_centroid_cohesion": cohesion[cid],
        })
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(outdir / "03_cluster_diagnostik.csv",
                index=False, sep=CSV_SEP, encoding=CSV_ENCODING)

    sil_orig = np.nan
    if valid.sum() >= 3 and n_clusters >= 2:
        sil_orig = float(silhouette_score(
            emb[valid], labels[valid], metric="cosine"
        ))

    def wavg(col):
        if diag.empty:
            return np.nan
        return float(np.average(diag[col], weights=diag["n_segments"]))

    return {
        "variant": name,
        "fields": " | ".join(fields),
        "n_segments": len(seg),
        "n_clusters": n_clusters,
        "n_noise": noise_n,
        "noise_share": noise_n / len(seg),
        "assignment_coverage": 1 - noise_n / len(seg),
        "segment_weighted_mean_technology_coverage": wavg("technology_coverage_share"),
        "segment_weighted_mean_technology_entropy": wavg("technology_entropy_normalized"),
        "segment_weighted_mean_single_tech_dominance": wavg("largest_single_technology_share"),
        "segment_weighted_mean_hdbscan_probability": wavg("mean_hdbscan_probability"),
        "segment_weighted_mean_embedding_cohesion": wavg("embedding_centroid_cohesion"),
        "silhouette_original_embedding_cosine_nonnoise": sil_orig,
    }, diag


def pairwise_cosine_distance(X):
    S = cosine_similarity(X)
    S = np.clip(S, -1.0, 1.0)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return S, D


def labels_from_linkage(Z, k):
    raw = fcluster(Z, t=k, criterion="maxclust")
    unique = sorted(np.unique(raw))
    remap = {old: i for i, old in enumerate(unique)}
    return np.array([remap[x] for x in raw], dtype=int)


def safe_silhouette(D, labels):
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(labels):
        return np.nan
    return float(silhouette_score(D, labels, metric="precomputed"))


# =====================================================================
# 2. INPUT, SEGMENTIERUNG, EMBEDDINGS
# =====================================================================
print("[1/8] Lade SATW-Daten ...")
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"{INPUT_CSV} fehlt. Zuerst 01_scrape_satw.py ausführen."
    )
df = pd.read_csv(INPUT_CSV, sep=CSV_SEP, encoding=CSV_ENCODING)
required = ["title"] + ALL_FIELDS
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Fehlende Spalten: {missing}")

print("[2/8] Lade spaCy und SentenceTransformer ...")
try:
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
except OSError as e:
    raise RuntimeError(
        f"spaCy-Modell {SPACY_MODEL!r} fehlt. "
        f"Installieren: python -m spacy download {SPACY_MODEL}"
    ) from e
nlp.max_length = max(nlp.max_length, 3_000_000)
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
max_seq_length = get_max_seq_length(model)

print("[3/8] Segmentiere alle drei Textfelder und prüfe Tokenlimits ...")
segments = build_segments(df, nlp, model)
segments.to_csv(OUT / "01_alle_segmente.csv",
                index=False, sep=CSV_SEP, encoding=CSV_ENCODING)

field_counts = segments.groupby("field").size().rename("n_segments").reset_index()
field_counts.to_csv(OUT / "02_segmentanzahl_nach_textfeld.csv",
                    index=False, sep=CSV_SEP, encoding=CSV_ENCODING)

print("[4/8] Berechne Segment-Embeddings einmalig ...")
embeddings = model.encode(
    segments["segment_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
embeddings = np.asarray(embeddings, dtype=np.float32)
np.save(OUT / "03_alle_segment_embeddings.npy", embeddings)

# =====================================================================
# 3. TEXTFELDVERGLEICH – methodischer Entscheidungsschritt bleibt sichtbar
# =====================================================================
print("[5/8] Reproduziere Textfeldvergleich A-E ...")
summary_rows, all_diag = [], []
for name, fields in VARIANTS.items():
    row, diag = run_variant(
        name, fields, segments, embeddings, df["title"].nunique()
    )
    summary_rows.append(row)
    all_diag.append(diag)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "04_textfeldvergleich_diagnostik.csv",
               index=False, sep=CSV_SEP, encoding=CSV_ENCODING)

plot_specs = [
    ("n_clusters", "Anzahl HDBSCAN-Cluster (ohne Noise)",
     "04a_textfeldvergleich_clusteranzahl.png"),
    ("noise_share", "Noise-Anteil", "04b_textfeldvergleich_noise.png"),
    ("segment_weighted_mean_technology_coverage",
     "Gewichtete Technologie-Coverage", "04c_textfeldvergleich_coverage.png"),
    ("segment_weighted_mean_technology_entropy",
     "Gewichtete Technologie-Entropie", "04d_textfeldvergleich_entropie.png"),
    ("segment_weighted_mean_single_tech_dominance",
     "Gewichtete Einzeldominanz", "04e_textfeldvergleich_dominanz.png"),
    ("segment_weighted_mean_embedding_cohesion",
     "Gewichtete Embedding-Kohäsion", "04f_textfeldvergleich_kohaesion.png"),
]
for col, ylabel, filename in plot_specs:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["variant"], summary[col], color=BAR_COLOR)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# =====================================================================
# 4. HAUPTVARIANTE B: MIKROCLUSTER
# =====================================================================
print(f"[6/8] Führe begründete Hauptvariante {MAIN_VARIANT} weiter ...")
main_dir = OUT / "hauptanalyse"
main_dir.mkdir(exist_ok=True)

main_fields = VARIANTS[MAIN_VARIANT]
main_mask = segments["field"].isin(main_fields).to_numpy()
main_segments = segments.loc[main_mask].copy().reset_index(drop=True)
main_embeddings = embeddings[main_mask]

# Exakte Hauptzuordnung aus dem Textfeldvergleich kopieren, statt neu zu fitten.
main_assignment_path = (
    OUT / "textfeldvergleich" / MAIN_VARIANT / "01_segment_clusterzuordnung.csv"
)
main_assignment = pd.read_csv(
    main_assignment_path, sep=CSV_SEP, encoding=CSV_ENCODING
)
main_assignment.to_csv(
    main_dir / "01_segment_mikrocluster.csv",
    index=False, sep=CSV_SEP, encoding=CSV_ENCODING
)
np.save(main_dir / "02_segment_embeddings_original.npy", main_embeddings)

valid = main_assignment["cluster"].to_numpy() >= 0
valid_seg = main_assignment.loc[valid].copy().reset_index(drop=True)
valid_emb = normalize(main_embeddings[valid], norm="l2")
cluster_ids = sorted(valid_seg["cluster"].astype(int).unique())
micro_labels = [f"C{x}" for x in cluster_ids]

centroids = []
micro_rows = []
technologies = sorted(main_assignment["technology"].dropna().unique())

for cid in cluster_ids:
    mask = valid_seg["cluster"].astype(int).to_numpy() == cid
    X = valid_emb[mask]
    g = valid_seg.loc[mask]
    c = X.mean(axis=0)
    c /= max(np.linalg.norm(c), 1e-12)
    centroids.append(c)
    tv = g["technology"].value_counts()
    micro_rows.append({
        "micro_cluster": f"C{cid}",
        "n_segments": len(g),
        "n_technologies": tv.size,
        "largest_technology_share": float(tv.iloc[0] / len(g)),
        "technology_entropy_normalized": normalized_entropy(tv.values),
        "mean_hdbscan_probability": float(g["hdbscan_probability"].mean()),
    })

centroids = np.vstack(centroids)
np.save(main_dir / "03_mikrocluster_centroids.npy", centroids)
pd.DataFrame(micro_rows).to_csv(
    main_dir / "04_mikrocluster_diagnostik.csv",
    index=False, sep=CSV_SEP, encoding=CSV_ENCODING
)

# =====================================================================
# 5. MAKROCLUSTER: 100 % SEMANTISCHE HIERARCHIE
# =====================================================================
print("[7/8] Berechne semantische Mikrocluster-Hierarchie ...")
S_sem, D_sem = pairwise_cosine_distance(centroids)
pd.DataFrame(S_sem, index=micro_labels, columns=micro_labels).to_csv(
    main_dir / "05_mikrocluster_semantik_similarity.csv",
    sep=CSV_SEP, encoding=CSV_ENCODING
)
pd.DataFrame(D_sem, index=micro_labels, columns=micro_labels).to_csv(
    main_dir / "06_mikrocluster_semantik_distanz.csv",
    sep=CSV_SEP, encoding=CSV_ENCODING
)

Z_main = linkage(squareform(D_sem, checks=False), method=LINKAGE_METHOD)
np.save(main_dir / "07_linkage_matrix.npy", Z_main)

# Dendrogramm mit Anzahl verbleibender Cluster an den Zusammenzügen.
fig, ax = plt.subplots(figsize=(14, 8))
ddata = dendrogram(Z_main, labels=micro_labels, leaf_rotation=0, ax=ax)
merge_distances = Z_main[:, 2]
for icoord, dcoord in zip(ddata["icoord"], ddata["dcoord"]):
    y = float(dcoord[1])
    x = float((icoord[1] + icoord[2]) / 2.0)
    merge_idx = int(np.argmin(np.abs(merge_distances - y)))
    k_after_merge = len(cluster_ids) - (merge_idx + 1)
    if k_after_merge >= 2:
        ax.annotate(
            f"k={k_after_merge}", xy=(x, y), xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom", fontsize=9
        )
ax.set_ylabel("Semantische Distanz")
fig.tight_layout()
fig.savefig(main_dir / "08_dendrogramm_mikrocluster.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)

# k=4...10 bewusst vollständig vergleichen.
solution_rows, mapping_rows, labels_by_k = [], [], {}
for k in K_VALUES:
    labels = labels_from_linkage(Z_main, k)
    labels_by_k[k] = labels
    sizes = pd.Series(labels).value_counts()

    macro_dom, macro_ent, macro_ntech, macro_nseg = [], [], [], []
    for m in sorted(np.unique(labels)):
        member_cids = [cluster_ids[i] for i in np.where(labels == m)[0]]
        gm = valid_seg[valid_seg["cluster"].astype(int).isin(member_cids)]
        tv = gm["technology"].value_counts()
        macro_nseg.append(len(gm))
        macro_ntech.append(tv.size)
        macro_dom.append(float(tv.iloc[0] / len(gm)))
        macro_ent.append(normalized_entropy(tv.values))

    solution_rows.append({
        "k_requested": k,
        "k_obtained": len(np.unique(labels)),
        "silhouette_semantic_microcluster_distance": safe_silhouette(D_sem, labels),
        "smallest_macro_n_microclusters": int(sizes.min()),
        "largest_macro_n_microclusters": int(sizes.max()),
        "smallest_macro_n_segments": int(min(macro_nseg)),
        "largest_macro_n_segments": int(max(macro_nseg)),
        "mean_macro_n_technologies": float(np.mean(macro_ntech)),
        "mean_largest_single_technology_share": float(np.mean(macro_dom)),
        "mean_technology_entropy_normalized": float(np.mean(macro_ent)),
    })
    for i, cid in enumerate(cluster_ids):
        mapping_rows.append({
            "k": k,
            "micro_cluster": f"C{cid}",
            "macro_cluster": f"M{labels[i]}",
        })

solution_df = pd.DataFrame(solution_rows)
mapping_df = pd.DataFrame(mapping_rows)
solution_df.to_csv(main_dir / "09_k4_10_diagnostik.csv",
                   index=False, sep=CSV_SEP, encoding=CSV_ENCODING)
mapping_df.to_csv(main_dir / "10_mikro_zu_makro_k4_10.csv",
                  index=False, sep=CSV_SEP, encoding=CSV_ENCODING)

for col, filename, ylabel in [
    ("silhouette_semantic_microcluster_distance",
     "11a_k_vergleich_silhouette.png", "Silhouette"),
    ("mean_largest_single_technology_share",
     "11b_k_vergleich_technologiedominanz.png",
     "mittlere Dominanz der grössten Technologie"),
    ("mean_technology_entropy_normalized",
     "11c_k_vergleich_technologieentropie.png",
     "mittlere normalisierte Technologie-Entropie"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(solution_df["k_requested"], solution_df[col],
            marker="o", color=BAR_COLOR)
    ax.set_xlabel("Anzahl Makrocluster k")
    ax.set_ylabel(ylabel)
    ax.set_xticks(K_VALUES)
    fig.tight_layout()
    fig.savefig(main_dir / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# Expliziter Export der nachgelagert interpretierten k=9-Lösung.
target_mapping = mapping_df[mapping_df["k"] == TARGET_K].copy()
target_mapping.to_csv(
    main_dir / f"12_mikro_zu_makro_k{TARGET_K}.csv",
    index=False, sep=CSV_SEP, encoding=CSV_ENCODING
)

# Segment -> Makro für k=9; Noise bleibt -1 / ohne Makrocluster.
micro_to_macro = {
    int(re.search(r"\d+", r.micro_cluster).group()):
    r.macro_cluster
    for r in target_mapping.itertuples()
}
target_segments = main_assignment.copy()
target_segments["macro_cluster"] = target_segments["cluster"].map(
    lambda x: micro_to_macro.get(int(x), np.nan) if int(x) >= 0 else np.nan
)
target_segments.to_csv(
    main_dir / f"13_segment_mikro_makro_k{TARGET_K}.csv",
    index=False, sep=CSV_SEP, encoding=CSV_ENCODING
)

# =====================================================================
# 6. PARAMETER / PROVENIENZ
# =====================================================================
print("[8/8] Speichere Parameter und Provenienz ...")
params = {
    "input_csv": str(INPUT_CSV.relative_to(BASE_DIR)),
    "model_name": MODEL_NAME,
    "model_max_seq_length": max_seq_length,
    "random_seed": RANDOM_SEED,
    "segmentation": {
        "unit": "spaCy sentence",
        "fields_segmented": ALL_FIELDS,
        "automatic_split": False,
        "truncation_allowed": ALLOW_TRUNCATION,
        "title_used_as_embedding_input": False,
    },
    "text_field_comparison": VARIANTS,
    "main_variant_after_methodological_comparison": MAIN_VARIANT,
    "main_variant_fields": main_fields,
    "umap": {
        "n_neighbors": UMAP_N_NEIGHBORS,
        "n_components": UMAP_N_COMPONENTS,
        "min_dist": UMAP_MIN_DIST,
        "metric": UMAP_METRIC,
    },
    "hdbscan": {
        "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
        "min_samples": HDBSCAN_MIN_SAMPLES,
        "metric": HDBSCAN_METRIC,
        "cluster_selection_method": HDBSCAN_CLUSTER_SELECTION_METHOD,
    },
    "macroclustering": {
        "representation": "normalized centroids in original segment embedding space",
        "distance": "1 - cosine similarity",
        "linkage": LINKAGE_METHOD,
        "semantic_weight": 1.0,
        "k_values_compared": K_VALUES,
        "automatic_k_selection": False,
        "target_k_for_subsequent_interpretation": TARGET_K,
    },
}
with open(OUT / "00_parameter_und_provenienz.json", "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False, indent=2)

print("\nFERTIG.")
print(f"Resultate: {OUT}")
print(f"Hauptvariante: {MAIN_VARIANT}")
print(f"Makro-k verglichen: {K_VALUES}; nachgelagerte Interpretation: k={TARGET_K}")
