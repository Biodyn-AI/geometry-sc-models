"""Gate stack (handoff §2, pipeline §2 / §6 Phase 2) applied to each operator's LET embedding.

  trustworthiness >= 0.80   : sklearn trustworthiness( operator-feature , LET-latent )
  geodesic<->ruler rho      : Spearman( k-NN-graph geodesic distances in latent , ruler distances )
  blocked-permutation null  : shuffle ruler labels WITHIN blocks (tissue for scGPT; cc-phase for Geneformer),
                              recompute geodesic<->ruler rho -> null margin + empirical p
  matched-null branch       : a biologically-nonsensical target (random cell-type relabeling) that must FAIL
"""
import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

from let import ruler_targets


def _geodesic(z, n_neighbors=12):
    """Isomap-style geodesic distances on a k-NN graph over latent z (n x d)."""
    n = len(z)
    k = min(n_neighbors, n - 1)
    G = kneighbors_graph(z, n_neighbors=k, mode="distance", include_self=False)
    G = G.maximum(G.T)                                    # symmetrise
    Dg = shortest_path(G, method="D", directed=False)
    # patch unreachable pairs with the finite max (disconnected components) so Spearman is defined
    finite = np.isfinite(Dg)
    if not finite.all():
        Dg[~finite] = Dg[finite].max() * 1.5
    return Dg


def geodesic_ruler_rho(z, ruler, n_pairs=30000, seed=0, n_neighbors=12):
    Dg = _geodesic(z, n_neighbors)
    n = len(z)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    m = i != j; i, j = i[m], j[m]
    # ruler distance per pair (reuse LET's ruler target, computed pairwise on the fly)
    if ruler["kind"] == "celltype":
        ct, ti = ruler["cell_type"], ruler["tissue"]
        rd = np.where(ct[i] == ct[j], 0.0, np.where(ti[i] == ti[j], 1.0, 2.0))
    else:
        s, g = ruler["s_score"], ruler["g2m_score"]
        rd = np.sqrt((s[i] - s[j]) ** 2 + (g[i] - g[j]) ** 2)
    gd = Dg[i, j]
    return float(spearmanr(gd, rd).statistic)


def _blocks(ruler):
    if ruler["kind"] == "celltype":
        return ruler["tissue"]
    return ruler["cc_phase"]


def blocked_permutation(z, ruler, n_perm=200, seed=0, n_neighbors=12):
    """Shuffle ruler labels WITHIN blocks; recompute geodesic<->ruler rho. Return real rho, null mean/std, p."""
    real = geodesic_ruler_rho(z, ruler, seed=seed, n_neighbors=n_neighbors)
    Dg = _geodesic(z, n_neighbors)                        # geodesic fixed; permute the ruler
    n = len(z)
    rng = np.random.default_rng(seed)
    ip = rng.integers(0, n, 30000); jp = rng.integers(0, n, 30000)
    mm = ip != jp; ip, jp = ip[mm], jp[mm]
    gd = Dg[ip, jp]
    blocks = _blocks(ruler)
    uniq = np.unique(blocks)
    nulls = []
    for p in range(n_perm):
        perm = np.arange(n)
        for bval in uniq:
            bidx = np.where(blocks == bval)[0]
            perm[bidx] = rng.permutation(bidx)
        if ruler["kind"] == "celltype":
            ct, ti = ruler["cell_type"][perm], ruler["tissue"][perm]
            rd = np.where(ct[ip] == ct[jp], 0.0, np.where(ti[ip] == ti[jp], 1.0, 2.0))
        else:
            s, g = ruler["s_score"][perm], ruler["g2m_score"][perm]
            rd = np.sqrt((s[ip] - s[jp]) ** 2 + (g[ip] - g[jp]) ** 2)
        nulls.append(spearmanr(gd, rd).statistic)
    nulls = np.array(nulls)
    p = float((1 + np.sum(np.abs(nulls) >= abs(real))) / (n_perm + 1))
    return dict(real=real, null_mean=float(nulls.mean()), null_std=float(nulls.std()),
                margin=float(real - nulls.mean()), p=p, n_perm=n_perm)


def trust(feature, z, n_neighbors=15):
    from common import zscore
    n = len(z)
    return float(trustworthiness(zscore(feature), zscore(z), n_neighbors=min(n_neighbors, n - 1)))


def matched_null_ruler(ruler, seed=0):
    """A biologically-nonsensical target: random relabeling that destroys the real structure.
    celltype -> random permutation of cell_type labels across all cells (breaks lineage).
    cellcycle -> shuffled scores. The pipeline requires this branch to FAIL the gates."""
    rng = np.random.default_rng(seed + 777)
    if ruler["kind"] == "celltype":
        n = len(ruler["cell_type"])
        return dict(kind="celltype", cell_type=ruler["cell_type"][rng.permutation(n)],
                    tissue=ruler["tissue"][rng.permutation(n)])
    n = len(ruler["s_score"])
    pr = rng.permutation(n)
    return dict(kind="cellcycle", s_score=ruler["s_score"][pr], g2m_score=ruler["g2m_score"][pr],
                cc_phase=ruler["cc_phase"][pr])
