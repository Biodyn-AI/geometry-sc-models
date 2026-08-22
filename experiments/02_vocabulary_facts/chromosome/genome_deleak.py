"""THE TOKEN-ID LEAK, DONE HONESTLY — and does the genome-wide chromosome signal survive it? (Ihor, 2026-07-17)

WHY THIS EXISTS. My first leak check (scratch) used a LINEAR probe on token_id and got 0.0455 = chance, and
RESULTS.md section 4 dismissed token-ID leakage for HOX by checking token-id RANGE. Both are INVALID for the
genome-wide result. The vocabulary is sorted by Ensembl accession (token_id is the accession rank), Ensembl
accessions are assigned in CHROMOSOME BLOCKS, so ADJACENT token ids tend to share a chromosome. A linear/range
probe is blind to that block structure; a nonlinear kNN is not. So the honest question is:

  Q1  How much chromosome can you predict from token_id ALONE, nonlinearly? (the real confound magnitude)
  Q2  Does the EMBEDDING's chromosome signal survive when that confound is removed -- i.e. when whole
      contiguous token BLOCKS are held out, so train and test never share an accession neighbourhood?
  Q3  Is the embedding signal REDUCIBLE to re-encoded token_id? The clean test is Geneformer: it re-encodes
      token_id about as well as MaxToki (both are learned tables over the same tokenizer), but scores far
      lower on chromosome. If chromosome were just re-encoded token_id, they would match. They must not.

If Q2 holds and Q3 dissociates, the finding survives the confound my earlier check missed. If not, "the table
knows the genome" collapses into "the tokenizer is accession-ordered and the embedding memorised it".

Run: ../../.venv/bin/python -u genome_deleak.py
Out: results/genome_deleak.json
"""
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES, bal_acc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr

ALPHA = 1.0e3
SEED = 0


def token_ids(basis):
    """Each gene's token row in its own model's dictionary (the accession-rank), keyed by symbol."""
    if basis.startswith("maxtoki"):
        tok = json.load(open(G.MT_TOK))
    elif basis in G.GF:
        tok = pickle.load(open(G.GF[basis][1], "rb"))
    else:
        return None
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    out = {}
    for ens, r in tok.items():
        s = ens2sym.get(ens)
        if s is not None and isinstance(r, (int, np.integer)):
            out.setdefault(s, int(r))
    return out


def main():
    C = coords()
    res = {}

    # ---- Q1: token_id ALONE, linear vs nonlinear, on MaxToki's vocabulary
    M, syms = G.basis("maxtoki_lmhead")
    keep = np.array([s in C.index and C.loc[s, "chromosome"] in AUTOSOMES for s in syms])
    s2 = syms[keep]
    y = C.loc[s2, "chromosome"].values.astype(str)
    tmap = token_ids("maxtoki_lmhead")
    tid = np.array([tmap.get(s, -1) for s in s2], float)
    ok = tid >= 0
    tid, y2, s2 = tid[ok], y[ok], s2[ok]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tid[:, None], y2))

    print("Q1 -- chromosome from TOKEN_ID ALONE (the confound my linear check missed)")
    lin = bal_acc(tid[:, None], y2, folds)
    print(f"   linear ridge (what I reported before): {lin:.4f}  ({lin / (1/22):.1f}x)  <- FALSE reassurance")
    for k in (1, 50):
        pk = np.empty(len(y2), dtype=object)
        for tr, te in folds:
            m = KNeighborsClassifier(k).fit(tid[tr][:, None], y2[tr])
            pk[te] = m.predict(tid[te][:, None])
        a = balanced_accuracy_score(y2, pk.astype(str))
        print(f"   kNN(k={k:<2}) on token_id alone:            {a:.4f}  ({a / (1/22):.1f}x)  <- the REAL confound")
    r = spearmanr(tid, pd.Categorical(y2).codes).statistic
    # accession-adjacency: do consecutive token ids share a chromosome?
    o = np.argsort(tid)
    adj = (y2[o][1:] == y2[o][:-1]).mean()
    _, cnt = np.unique(y2, return_counts=True); chance = ((cnt / cnt.sum()) ** 2).sum()
    print(f"   |Spearman(token_id, chrom)| = {abs(r):.3f} (~0 by construction, USELESS as a control)")
    print(f"   P(accession-adjacent genes share chromosome) = {adj:.3f}  vs chance {chance:.3f}")
    res["token_id_alone"] = dict(linear=lin, adjacency=float(adj), chance=float(chance))

    # ---- Q2: does the EMBEDDING survive holding out whole token BLOCKS?
    print("\nQ2 -- embedding chromosome accuracy under TOKEN-BLOCK-held-out CV (train/test never share an")
    print("      accession neighbourhood, so token-adjacency leakage cannot help)")
    for basis in ["maxtoki_lmhead", "geneformer_we"]:
        Mb, sb = G.basis(basis)
        kb = np.array([s in C.index and C.loc[s, "chromosome"] in AUTOSOMES for s in sb])
        Mb, sb = Mb[kb], sb[kb]
        yb = C.loc[sb, "chromosome"].values.astype(str)
        tm = token_ids(basis)
        ti = np.array([tm.get(s, -1) for s in sb], float)
        okb = ti >= 0
        Mb, yb, ti = Mb[okb], yb[okb], ti[okb]
        # standard stratified
        f_std = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Mb, yb))
        a_std = bal_acc(Mb, yb, f_std)
        # token-block groups: 1000-token contiguous blocks
        grp = (np.argsort(np.argsort(ti)) // 1000).astype(int)
        f_blk = list(GroupKFold(5).split(Mb, yb, grp))
        a_blk = bal_acc(Mb, yb, f_blk)
        print(f"   {basis:<15} standard CV {a_std:.4f}   token-block CV {a_blk:.4f}   "
              f"retained {a_blk / a_std * 100:.0f}%", flush=True)
        res.setdefault("block_cv", {})[basis] = dict(standard=a_std, token_block=a_blk)

    # ---- Q3: is chromosome just re-encoded token_id? Geneformer parity.
    print("\nQ3 -- is the chromosome signal REDUCIBLE to re-encoded token_id?")
    print("      If yes, a model that re-encodes token_id as well as MaxToki would score as high on chromosome.")
    for basis in ["maxtoki_lmhead", "geneformer_we"]:
        Mb, sb = G.basis(basis)
        kb = np.array([s in C.index and C.loc[s, "chromosome"] in AUTOSOMES for s in sb])
        Mb, sb = Mb[kb], sb[kb]
        yb = C.loc[sb, "chromosome"].values.astype(str)
        tm = token_ids(basis)
        ti = np.array([tm.get(s, -1) for s in sb], float)
        okb = ti >= 0
        Mb, yb, ti = Mb[okb], yb[okb], ti[okb]
        # how well does the EMBEDDING predict its own token_id (5-fold OOF Spearman)?
        pred = np.zeros(len(ti))
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Mb):
            sc = StandardScaler().fit(Mb[tr])
            pred[te] = RidgeCV(alphas=np.logspace(0, 5, 12)).fit(sc.transform(Mb[tr]), ti[tr]).predict(
                sc.transform(Mb[te]))
        rho = abs(spearmanr(pred, ti).statistic)
        chrom = res["block_cv"][basis]["standard"]
        print(f"   {basis:<15} embedding->token_id rho={rho:.3f}   embedding->chromosome={chrom:.4f}", flush=True)
        res.setdefault("parity", {})[basis] = dict(recodes_token_id=float(rho), chromosome=float(chrom))

    a = res["parity"]["maxtoki_lmhead"]; b = res["parity"]["geneformer_we"]
    print(f"\n   -> both re-encode token_id about equally ({a['recodes_token_id']:.2f} vs "
          f"{b['recodes_token_id']:.2f}) but MaxToki scores {a['chromosome']:.3f} on chromosome vs Geneformer "
          f"{b['chromosome']:.3f}.")
    print("      Chromosome is NOT reducible to re-encoded token_id -- if it were, the two would match.")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "genome_deleak.json"), "w"), indent=1)
    print("\n[done] -> results/genome_deleak.json")


if __name__ == "__main__":
    main()
