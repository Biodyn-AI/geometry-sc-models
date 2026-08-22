"""AUDIT MaxToki's Genecorpus-175M CONSTITUENT STUDIES FOR TUMOUR / CELL-LINE CONTENT.

Supplementary Table S1 of Gomez Ortega et al. 2026 (PMC13060336, media-1.xlsx) lists every constituent dataset
as (source DOI, n_cells_passing_filtering, organ). The `organ` column is ANATOMICAL, so a breast-tumour study is
labelled "breast" -- absence of cancer words there proves nothing. The study TITLE does discriminate, so we
resolve every unique DOI against the CrossRef API and classify by title (plus journal/subject where available).

WHY THIS MATTERS. The paper states that "malignant cells and immortalized cell lines were excluded". That is an
authorial claim about CELL-level filtering. This audit measures something different and complementary: how much
of the corpus comes from STUDIES THAT SAMPLED TUMOURS AT ALL. The two are consistent -- a tumour study can
contribute only its non-malignant compartment (stroma, immune infiltrate, endothelium), all karyotypically
normal -- so a nonzero cancer-study share does NOT contradict the exclusion. What it bounds is EXPOSURE: the
maximum fraction of the corpus where malignant cells could possibly have leaked through an imperfect filter.

READING IT
  cancer-study share ~0%     -> the copy-number route had almost no opportunity; CNV account is dead
  small (a few %)            -> exposure is bounded and small; leakage would have to be near-total to matter
  large (>20%)               -> the exclusion filter is doing heavy lifting and its unstated method matters a lot

DELIBERATELY CONSERVATIVE: the keyword list is broad and matches substrings, so it OVER-calls cancer (e.g. a
study of "tumor-adjacent normal tissue" counts as cancer-exposed). Over-calling is the safe direction here --
it inflates the exposure bound rather than hiding it.

Out: results/corpus_s1_audit.json  (+ unresolved DOIs listed for manual follow-up)
"""
import json, os, re, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = sys.argv[1] if len(sys.argv) > 1 else str(_DATA / "media-1.xlsx")   # see docs/DATA.md
MAILTO = _os.environ.get("CROSSREF_MAILTO", "")   # Crossref polite-pool contact          # CrossRef "polite pool" -- identifies the caller, gets better service
UA = f"biomi-automation-corpus-audit/1.0 (mailto:{MAILTO})"

# Broad, substring-matched. Over-calling is intentional (see docstring).
CANCER = ["tumor", "tumour", "cancer", "carcinoma", "malign", "neoplas", "leukemi", "leukaemi", "lymphoma",
          "melanoma", "glioma", "glioblast", "sarcoma", "myeloma", "metasta", "adenoma", "blastoma",
          "oncogen", "oncolog", "\bcll\b", "\baml\b", "\ball\b", "myelodysplas", "astrocytoma", "mesotheli",
          "cholangiocarcin", "hepatocellular", "squamous cell", "premalignant", "dysplasia", "polyp"]
CELLLINE = ["cell line", "cell-line", "k562", "hela", "hek293", "jurkat", "immortalized", "immortalised",
            "perturb-seq", "crispri screen"]


def load_rows(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
    out = []
    for r in list(ws.iter_rows(values_only=True))[1:]:
        src = str(r[0]).strip() if r[0] else ""
        if not src:
            continue
        out.append((src, r[1] if isinstance(r[1], (int, float)) else 0,
                    str(r[2]).strip().lower() if r[2] else ""))
    return out


def norm_doi(s):
    s = s.strip()
    m = re.search(r"10\.\d{4,9}/\S+", s)
    return m.group(0).rstrip(").,;") if m else None


def fetch(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as r:
                m = json.load(r)["message"]
                return dict(doi=doi, title=" ".join(m.get("title") or []),
                            journal=" ".join(m.get("container-title") or []),
                            subject="; ".join(m.get("subject") or []),
                            year=(m.get("issued", {}).get("date-parts") or [[None]])[0][0])
        except Exception as e:
            if attempt == 2:
                return dict(doi=doi, error=repr(e)[:80])
            time.sleep(1.5 * (attempt + 1))


def classify(rec):
    blob = " ".join(str(rec.get(k, "")) for k in ("title", "journal", "subject")).lower()
    if not blob.strip():
        return "unresolved"
    if any(re.search(k, blob) for k in CELLLINE):
        return "cell_line"
    if any(re.search(k, blob) for k in CANCER):
        return "cancer"
    return "other"


def main():
    rows = load_rows(XLSX)
    total_cells = sum(n for _, n, _ in rows)
    dois = {}
    unparsed = []
    for src, n, organ in rows:
        d = norm_doi(src)
        if d is None:
            unparsed.append((src, n)); continue
        dois.setdefault(d, 0)
        dois[d] += n
    print(f"[setup] {len(rows)} rows | {total_cells:,} cells | {len(dois)} unique DOIs | "
          f"{len(unparsed)} rows with unparseable source ({sum(n for _, n in unparsed):,} cells)", flush=True)

    recs, done = {}, [0]
    def work(d):
        r = fetch(d); recs[d] = r; done[0] += 1
        if done[0] % 100 == 0:
            print(f"   resolved {done[0]}/{len(dois)}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, dois))

    agg, per = {}, []
    for d, cells in dois.items():
        c = classify(recs.get(d, {}))
        agg[c] = agg.get(c, 0) + cells
        per.append(dict(doi=d, cells=cells, cls=c, title=recs.get(d, {}).get("title", "")[:160]))

    print(f"\n--- corpus exposure by study class (denominator = {total_cells:,} cells) ---")
    for c in ("cancer", "cell_line", "other", "unresolved"):
        n = agg.get(c, 0)
        print(f"  {c:<12} {n:>13,}  {100*n/total_cells:6.2f}%")
    exposure = agg.get("cancer", 0) + agg.get("cell_line", 0)
    unres = agg.get("unresolved", 0)
    print(f"\n  CANCER-EXPOSED (cancer + cell_line): {exposure:,} cells = {100*exposure/total_cells:.2f}%")
    print(f"  worst case if EVERY unresolved DOI were cancer: "
          f"{100*(exposure+unres)/total_cells:.2f}%")

    print("\n--- 15 largest cancer/cell-line studies by cell count ---")
    for r in sorted([p for p in per if p["cls"] in ("cancer", "cell_line")], key=lambda p: -p["cells"])[:15]:
        print(f"  {r['cells']:>9,}  [{r['cls']}]  {r['title'][:100]}")

    bad = [p for p in per if p["cls"] == "unresolved"]
    if bad:
        print(f"\n--- {len(bad)} unresolved DOIs ({sum(p['cells'] for p in bad):,} cells), top 10 ---")
        for r in sorted(bad, key=lambda p: -p["cells"])[:10]:
            print(f"  {r['cells']:>9,}  {r['doi']}  {recs.get(r['doi'],{}).get('error','')}")

    out = dict(total_cells=total_cells, n_rows=len(rows), n_dois=len(dois),
               unparsed_rows=len(unparsed), unparsed_cells=sum(n for _, n in unparsed),
               by_class={k: v for k, v in agg.items()},
               cancer_exposed_cells=exposure, cancer_exposed_frac=exposure / total_cells,
               worst_case_frac=(exposure + unres) / total_cells, per_doi=per)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "corpus_s1_audit.json"), "w"), indent=1)
    print("\n[done] -> results/corpus_s1_audit.json")


if __name__ == "__main__":
    main()
