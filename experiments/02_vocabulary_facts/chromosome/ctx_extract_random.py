"""RANDOM-WEIGHTS CONTROL for the contextualisation phenomenon — the single most load-bearing missing control.

Every emergent-representation claim needs it: is the effect LEARNED, or an inevitability of the architecture?
Attention mixes tokens regardless of weights, so SOME context dependence is guaranteed. The question is whether
the GENE-SPECIFIC, FUNCTIONALLY-ORGANISED, reproducible modulation we measured is a property of the TRAINED
model or of any Llama with this shape.

We instantiate LlamaForCausalLM from MaxToki-217M's exact config with RANDOM init (same architecture, same
vocab, untrained), keep the token embeddings random-but-fixed, and run the identical extraction. Then compare:
  EXCESS      — if it stays high, gene-specific context response is partly architectural (fixed random
                embeddings give each gene a fixed identity that deterministic mixing preserves). If it collapses,
                it is learned. Either way is an honest, publishable decomposition.
  FUNC-Z      — the discriminating one. Random weights have no reason to align context modulation with
                co-expression / functional axes, so functional-z should collapse to ~0 if the trained model's
                functional organisation is learned. A trained +21 vs random ~0 is the result the paper needs.

Same tokenisation, contexts, panel-selection, cap, partitions as ctx_extract_maxtoki. Fewer cells (600) to keep
it ~45 min. Out: results/ctxrand_L{tap}.npz  (matches the ctx_maxtoki schema so every analysis reads it as-is).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, collections, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, h5py, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MSETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MSETUP)
TS = f"{_DATA}/raw"
PANELS = ["tabula_sapiens_immune_subset_20000.h5ad", "tabula_sapiens_kidney.h5ad", "tabula_sapiens_lung.h5ad"]
MDIR = f"{MSETUP}/MaxToki-217M-HF"
MAX_LEN, N_CTX, CELLS_CTX, CAP, FLOOR, MAX_GENES = 1024, 12, 600, 50, 25, 6000
TAPS = [4, 8]
BATCH, SEED, NPART = 4, 0, 2


def main():
    from maxtoki_adapter import MaxTokiTokenizer
    from transformers import LlamaForCausalLM, LlamaConfig
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

    # ---- reuse ctx_extract_maxtoki's exact tokenisation via its stream_panel ----
    import ctx_extract_maxtoki as EX
    print("[pass 1] tokenise + choose panel/contexts (identical to trained extraction)", flush=True)
    cells = collections.defaultdict(list)
    for p in PANELS:
        path = os.path.join(TS, p)
        if not os.path.exists(path):
            continue
        for toks, ct in EX.stream_panel(path, tok):
            cells[ct].append(toks)
    ctx_names = sorted([c for c in cells if len(cells[c]) >= 250], key=lambda c: -len(cells[c]))[:N_CTX]
    for c in ctx_names:
        rng.shuffle(cells[c]); cells[c] = cells[c][:CELLS_CTX]
    cnt = {c: collections.Counter(int(t) for s in cells[c] for t in s) for c in ctx_names}
    reach = collections.Counter()
    for c in ctx_names:
        for g, n in cnt[c].items():
            if n >= FLOOR:
                reach[g] += 1
    panel = sorted([g for g, k in reach.items() if k >= 2],
                   key=lambda g: -sum(cnt[c].get(g, 0) for c in ctx_names))[:MAX_GENES]
    panel = sorted(panel); gpos = {g: i for i, g in enumerate(panel)}
    print(f"[pass 1] {len(ctx_names)} contexts, {len(panel)} genes", flush=True)

    # ---- RANDOM-INIT model, same architecture/config ----
    config = LlamaConfig.from_pretrained(MDIR)
    model = LlamaForCausalLM(config)          # random init
    model.eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)
    d = config.hidden_size
    print(f"[model] RANDOM-INIT LlamaForCausalLM on {dev}, {config.num_hidden_layers} layers, d={d}", flush=True)

    acc = {L: np.zeros((NPART, len(ctx_names), len(panel), d), np.float32) for L in TAPS}
    cnts = np.zeros((NPART, len(ctx_names), len(panel)), np.int32)
    work = [(ci, si, s) for ci, c in enumerate(ctx_names) for si, s in enumerate(cells[c])]
    rng.shuffle(work)
    done = 0
    for a in range(0, len(work), BATCH):
        chunk = work[a:a + BATCH]
        L = max(len(s) for _, _, s in chunk) + 2
        ids = np.full((len(chunk), L), tok.EOS, np.int64); am = np.zeros((len(chunk), L), np.int64)
        for j, (_, _, s) in enumerate(chunk):
            sq = np.concatenate([[tok.BOS], s, [tok.EOS]]); ids[j, :len(sq)] = sq; am[j, :len(sq)] = 1
        with torch.no_grad():
            out = model(input_ids=torch.from_numpy(ids).to(dev),
                        attention_mask=torch.from_numpy(am).to(dev), output_hidden_states=True)
            hs = {L_: out.hidden_states[L_].to("cpu", torch.float32).numpy() for L_ in TAPS}
        for j, (ci, si, s) in enumerate(chunk):
            part = si % NPART
            for p_, t in enumerate(s):
                gi = gpos.get(int(t))
                if gi is None or cnts[part, ci, gi] >= CAP:
                    continue
                cnts[part, ci, gi] += 1
                for L_ in TAPS:
                    acc[L_][part, ci, gi] += hs[L_][j, 1 + p_]
        done += len(chunk)
        if done % 400 < BATCH:
            print(f"    {done}/{len(work)} cells", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tokmap = json.load(open(f"{MSETUP}/token_dictionary.json"))
    tid2ens = {int(v): k for k, v in tokmap.items()}
    for L_ in TAPS:
        M = acc[L_] / np.maximum(cnts[..., None], 1)
        out = os.path.join(HERE, "results", f"ctxrand_L{L_:02d}.npz")
        np.savez_compressed(out, M=M.astype(np.float16), counts=cnts,
                            genes=np.array([tid2ens.get(g, str(g)) for g in panel]),
                            contexts=np.array(ctx_names), cap=CAP, max_len=MAX_LEN)
        print(f"  wrote {out}")
    print(f"[done] random-weights control; {float((cnts >= CAP).mean()):.1%} of cells at cap")


if __name__ == "__main__":
    main()
