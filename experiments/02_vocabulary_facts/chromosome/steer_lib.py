"""steer_lib — a general activation-STEERING harness for MaxToki (Ihor, 2026-07-17).

WHY THIS EXISTS. route_genemanifold has established, via DECODABILITY (probes on the gene table) and one
OBSERVATIONAL causal test (§6B: regress the model's own logits on a context annotation, controlling for gene
identity), that the table encodes genomic locus and the forward pass conditions on it. The missing leg is
INTERVENTIONAL: perturb a feature inside the model and watch the effect propagate. This is that tool, built to
be reused across many hypotheses (chromosome, cell-cycle phase, pathway, tissue program, ...).

THE ONE VALIDITY RULE (learned the hard way in §5 and §6B, do not violate it):
    A steering result is evidence ONLY if the READOUT is causally downstream of, and independent from, the
    intervention. §5 steered a centroid direction and read a classifier fit on those same centroids -> scored
    1.000 for EVERY basis (co-expression, ESM2, everything) because it was arithmetic, not causation. §6B's
    naive ablation removed an lm_head-defined subspace and read lm_head -> circular by construction.
So the PRIMARY readout here is the MODEL'S OWN next-gene logits (no fitted probe -> nothing to be circular
with). A trained classifier is a SECOND, optional readout whose value is SPECIFICITY across properties, added
later. And every steering run is meaningless without its CONTROLS:
    - random   : a norm-matched random direction must produce ~0 effect (else the effect is generic energy).
    - dose      : sweep alpha; a real causal channel is monotone, not a step.
    - specificity: (with the multi-head classifier) steering feature F moves F's head, not unrelated heads.

TWO INTERVENTION SITES (both supported, per Ihor 2026-07-17):
    site="embed"  -> add alpha*d to embed_tokens output at chosen positions (perturb a gene's INPUT embedding;
                     this is where route_genemanifold's findings physically live, so it is the most direct test).
    site=L (int)  -> add alpha*d to the residual stream AFTER decoder layer L (feature as re-represented mid-
                     computation). Classic activation-steering.

Directions live in HIDDEN space (dim = config.hidden_size). Build them from the gene-embedding table with
`centroid_direction` (categorical axis) or pass any probe weight vector. TAG each direction with the basis it
came from so a caller can refuse to read it in a circular basis.

Model is run via projects/maxtoki/setup/maxtoki_adapter.MaxTokiAttentionExtractor (LlamaForCausalLM, HF
safetensors, 217M). Needs the transformers venv: ../../.venv_state/bin/python.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np
import torch

MAXTOKI_SETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MAXTOKI_SETUP)
from maxtoki_adapter import MaxTokiAttentionExtractor, MaxTokiTokenizer  # noqa: E402

MODEL_217M = f"{MAXTOKI_SETUP}/MaxToki-217M-HF"     # hidden 1232, 11 layers  -- all work through 2026-07-19
MODEL_1B = f"{MAXTOKI_SETUP}/MaxToki-1B-HF"        # hidden 2304, 20 layers  -- ~4x the compute per forward
MODELS = {"217m": MODEL_217M, "1b": MODEL_1B}
TOKMAP = f"{MAXTOKI_SETUP}/token_dictionary.json"


# ---------------------------------------------------------------- direction library
@dataclass
class Direction:
    """A hidden-space steering direction, tagged with provenance so it is never read in a circular basis."""
    vec: np.ndarray               # (hidden,), stored UNIT-normalised
    name: str                     # e.g. "cluster:A->B"
    basis: str                    # where it came from, e.g. "embed_tokens" / "lm_head" / "probe:cellcycle"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        v = np.asarray(self.vec, dtype=np.float64).ravel()
        n = np.linalg.norm(v)
        self.vec = (v / n) if n > 0 else v


def _embed_matrix(xt, which="embed"):
    """Return (M, hidden) the model's OWN gene-embedding table in hidden space, plus a token_id->row map.
    which='embed' -> input embed_tokens.weight ; 'lmhead' -> lm_head.weight (untied output)."""
    if which == "embed":
        W = xt.model.model.embed_tokens.weight
    else:
        W = xt.model.lm_head.weight
    return W.detach().cpu().to(torch.float64).numpy()      # CPU first: MPS has no float64


def centroid_direction(xt, token_ids_a, token_ids_b, name, which="embed"):
    """Direction = mean(rows of group B) - mean(rows of group A), in the model's OWN embed/lmhead space.
    This is a hidden-space vector you can ADD to the residual: 'make this position look more like group B'."""
    M = _embed_matrix(xt, which)
    a = M[np.asarray(token_ids_a)].mean(0)
    b = M[np.asarray(token_ids_b)].mean(0)
    return Direction(vec=(b - a), name=name, basis=f"{which}_tokens" if which == "embed" else "lm_head",
                     meta=dict(kind="centroid", n_a=len(token_ids_a), n_b=len(token_ids_b)))


def random_direction(xt, seed, name="random", like=None):
    """A norm-matched random control direction. If `like` is given, matches its (already-unit) geometry trivially
    (both unit); the point of the control is DIRECTION, magnitude is carried by alpha at steer time."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(xt.hidden_size)
    return Direction(vec=v, name=name, basis="random", meta=dict(seed=seed))


# ---------------------------------------------------------------- the steerer
class Steerer:
    """Wraps a MaxTokiAttentionExtractor and injects steering vectors via forward hooks.

    Usage:
        st = Steerer()
        base = st.logits(ids)                                  # (seq, vocab), no steering
        with st.steering(direction, alpha=6.0, positions=mask, site="embed"):
            steered = st.logits(ids)
    Hooks are removed on context exit; nesting is not supported (assert guards it).
    """

    def __init__(self, model_dir=MODEL_217M, dtype=torch.float32):
        self.xt = MaxTokiAttentionExtractor(model_dir=model_dir, dtype=dtype)
        self.model = self.xt.model
        self.device = self.xt.device
        self.hidden_size = self.xt.hidden_size
        self.n_layers = self.xt.n_layers
        self.dtype = dtype
        self.tok = MaxTokiTokenizer(model_input_size=4096)
        self._active = None

    # -- the injection hook -------------------------------------------------
    def _pos_mask(self, positions, batch, seq):
        """Normalise `positions` to a (batch, seq) float mask on-device. Accepts: a 1-D index list / 1-D bool
        (applied to every row), or a (batch, seq) bool array (per-row — needed for the split-half design)."""
        p = np.asarray(positions)
        if p.ndim == 2:
            m = p.astype(bool)
        else:
            row = p.astype(bool) if p.dtype == bool else np.zeros(seq, bool)
            if p.dtype != bool:
                row[p.astype(int)] = True
            m = np.broadcast_to(row, (batch, seq))
        return torch.as_tensor(np.ascontiguousarray(m), dtype=self.dtype, device=self.device)

    @contextmanager
    def steering(self, direction: "Direction", alpha: float, positions, site="embed", seq=None, batch=1,
                 per_token=None, input_ids=None):
        """Add alpha * direction to the residual at `positions`, at the embedding output (site='embed') or
        after decoder layer `site` (int). `positions` may be a per-row (batch, seq) bool mask.

        SOURCE-RELATIVE STEERING (`per_token`). A single global vector implements "become more C-ish" only if
        the feature is an ORDERED axis with parallel offsets. For chromosome it is NOT: §5 established a
        CLUSTERING (unordered blobs) whose offsets are only ~0.2 aligned, so one global d_C means different
        things depending on where a gene starts. Pass `per_token` as a (vocab, hidden) matrix of PER-GENE
        displacements (e.g. centroid(target) - centroid(the gene's own chromosome)) plus `input_ids`, and each
        pushed position gets the displacement appropriate to the gene actually sitting there.
        """
        assert self._active is None, "nested steering is not supported"
        mask_cache = {"m": None if seq is None else self._pos_mask(positions, batch, seq)}

        if per_token is not None:
            assert input_ids is not None, "per_token steering needs the input_ids to look genes up"
            # per_token is (codebook (K,hidden), token_row (vocab,)) -- a COMPACT lookup, not a full
            # (vocab, hidden) matrix. Materialising the dense form costs ~200 MB per direction and blew past
            # the machine's memory budget when sweeping many targets (see memory: x6-machine-memory-budget).
            cb, row = per_token
            CB = torch.as_tensor(np.asarray(cb), dtype=self.dtype, device=self.device)
            RW = torch.as_tensor(np.asarray(row), dtype=torch.long, device=self.device)
            ids_t = torch.as_tensor(np.asarray(input_ids), dtype=torch.long,
                                    device=self.device).reshape(1, -1)
            add_pt = alpha * CB[RW[ids_t]]                                       # (1, seq, hidden)
            add = None
        else:
            d = torch.tensor(direction.vec, dtype=self.dtype, device=self.device)    # unit vector
            add = alpha * d                                                          # (hidden,)
            add_pt = None

        def _edit(hidden):
            b, s, _ = hidden.shape
            m = mask_cache["m"]
            if m is None or m.shape != (b, s):
                m = self._pos_mask(positions, b, s); mask_cache["m"] = m
            if add_pt is not None:
                return hidden + m.unsqueeze(-1) * add_pt[:, :s, :].to(hidden.dtype)
            return hidden + m.unsqueeze(-1) * add.to(hidden.dtype)

        if site == "embed":
            module = self.model.model.embed_tokens

            def hook(mod, inp, out):
                return _edit(out)
        else:
            module = self.model.model.layers[int(site)]

            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    return (_edit(out[0]),) + tuple(out[1:])
                return _edit(out)

        handle = module.register_forward_hook(hook)
        self._active = (site, direction.name if direction is not None else "per_token", float(alpha))
        try:
            yield
        finally:
            handle.remove()
            self._active = None

    # -- forward readout ----------------------------------------------------
    @torch.no_grad()
    def logits(self, input_ids, attention_mask=None):
        """Native next-token logits. 1-D ids -> (seq, vocab); 2-D ids -> (batch, seq, vocab). float32 on CPU.
        THE primary, non-circular readout (no fitted probe -> nothing to be circular with)."""
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        single = ids.ndim == 1
        if single:
            ids = ids.reshape(1, -1)
        kw = {}
        if attention_mask is not None:
            kw["attention_mask"] = torch.as_tensor(attention_mask, dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, **kw)
        lg = out.logits.detach().to("cpu", dtype=torch.float32)
        return lg[0] if single else lg

    @torch.no_grad()
    def hidden(self, input_ids, attention_mask=None, layer=-1):
        """Hidden states at `layer` (-1 = final). 1-D ids -> (seq, hidden); 2-D -> (batch, seq, hidden).
        This is what the SECOND readout (the shared multi-head classifier) reads. Steering at the input
        embedding or an early layer and reading the FINAL hidden state means the readout is separated from the
        intervention by real transformer computation -- the condition that makes it non-circular."""
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        single = ids.ndim == 1
        if single:
            ids = ids.reshape(1, -1)
        kw = {}
        if attention_mask is not None:
            kw["attention_mask"] = torch.as_tensor(attention_mask, dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, output_hidden_states=True, **kw)
        h = out.hidden_states[layer].detach().to("cpu", dtype=torch.float32)
        return h[0] if single else h

    @torch.no_grad()
    def forward_both(self, input_ids, layer=-1):
        """(hidden_at_layer, logits) from ONE forward pass — the classifier readout and the native readout
        together. Halves the cost of any experiment that wants both (e.g. the specificity test)."""
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device).reshape(1, -1)
        out = self.model(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[layer][0].detach().to("cpu", dtype=torch.float32)
        lg = out.logits[0].detach().to("cpu", dtype=torch.float32)
        return h, lg

    @staticmethod
    def pool(hidden_seq, positions):
        """Mean-pool a (seq, hidden) hidden state over `positions` -> (hidden,). The standard per-cell vector."""
        p = np.asarray(positions)
        idx = np.nonzero(p)[0] if p.dtype == bool else p.astype(int)
        return hidden_seq[idx].mean(0).numpy()

    # -- convenience --------------------------------------------------------
    def token_of(self, ensembl_or_symbol_map, sym):
        """Look up a gene's token id via a provided symbol->token dict."""
        return ensembl_or_symbol_map.get(sym.upper())


# ---------------------------------------------------------------- readout helpers
def mean_logit(logits_row, token_ids):
    """Mean logit over a set of gene tokens at one position (a readout summary)."""
    t = np.asarray(token_ids, dtype=np.int64)
    t = t[t < logits_row.shape[-1]]
    return float(logits_row[t].mean()) if len(t) else np.nan


def dose_response(steerer, input_ids, direction, positions, read_positions, target_tokens, control_tokens,
                  alphas, site="embed"):
    """Sweep alpha; at each `read_positions` report mean logit over target vs control token sets, relative to
    the alpha=0 baseline. A real causal channel: target rises with alpha, control ~flat, random direction flat.
    Returns a list of dicts (one per alpha)."""
    base = steerer.logits(input_ids)
    b_t = np.mean([mean_logit(base[p], target_tokens) for p in read_positions])
    b_c = np.mean([mean_logit(base[p], control_tokens) for p in read_positions])
    rows = []
    for a in alphas:
        if a == 0:
            dt, dc = 0.0, 0.0
        else:
            with steerer.steering(direction, alpha=a, positions=positions, site=site):
                lg = steerer.logits(input_ids)
            dt = float(np.mean([mean_logit(lg[p], target_tokens) for p in read_positions]) - b_t)
            dc = float(np.mean([mean_logit(lg[p], control_tokens) for p in read_positions]) - b_c)
        rows.append(dict(alpha=float(a), d_target=dt, d_control=dc, specificity=dt - dc))
    return rows


if __name__ == "__main__":
    # minimal self-check: load, one baseline forward, one steered forward, confirm the hook changes logits.
    st = Steerer()
    ids = np.array([2, 100, 200, 300, 3], dtype=np.int64)      # bos, 3 arbitrary genes, eos
    base = st.logits(ids)
    d = random_direction(st.xt, seed=0)
    with st.steering(d, alpha=10.0, positions=[1, 2, 3], site="embed"):
        steered = st.logits(ids)
    delta = (steered - base).abs().mean().item()
    print(f"[smoke] hidden={st.hidden_size} n_layers={st.n_layers} vocab={base.shape[-1]}")
    print(f"[smoke] mean|Δlogit| under a random embed-steer (alpha=10): {delta:.4f}  "
          f"({'hook works' if delta > 0 else 'HOOK DID NOTHING'})")
    with st.steering(d, alpha=10.0, positions=[1, 2, 3], site=2):
        steered_L2 = st.logits(ids)
    dL2 = (steered_L2 - base).abs().mean().item()
    print(f"[smoke] mean|Δlogit| under a random layer-2 steer (alpha=10): {dL2:.4f}  "
          f"({'hook works' if dL2 > 0 else 'HOOK DID NOTHING'})")
