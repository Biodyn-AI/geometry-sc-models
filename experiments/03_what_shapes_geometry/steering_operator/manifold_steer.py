"""manifold_steer — on-manifold steering of single-cell trajectories via local-tangent projection.

Standalone (numpy + scikit-learn only; no foundation model, no project internals). Packages the method
validated in ../RESULTS.md: to move a cell along a chosen direction (a supervised pseudotime-ascent direction,
or toward a target population) while staying on the manifold of real cells, project every integration step onto
the LOCAL TANGENT — the top-m principal directions of the cell's nearest real neighbours. The ablations in
`benchmark_setty.py` show the projection, not the direction, is what makes steering stay on-manifold; Gate 0 in
../RESULTS.md shows it works at least as well in raw counts as in any foundation-model embedding — hence
"model-free".

Scope (stated up front; see ../RESULTS.md H-A2/H-A3/H-B1):
  - It INTERPOLATES along observed geometry. It does not GENERATE states off the data (it cannot cross a gap
    where a population was removed), and it does not predict which perturbation would cause a move.
  - It needs a populated manifold between source and target; steering into empty regions drifts.

Typical uses: in-silico fate biasing ("show a progenitor moving toward fate X on-manifold"), plausible
intermediate-state interpolation, denoising a direction to its realizable on-manifold part.

Quick start:
    ms = ManifoldSteer(n_pcs=20).fit(X)                 # X: cells x genes (log-normalized), or any embedding
    d  = ms.ascent_direction(pseudotime)                # or ms.fate_direction(src_mask, target_mask)
    traj = ms.steer(ms.transform(X[src_mask]), d, n_steps=8)   # (n_steps+1, n_src, n_pcs), on-manifold
    expr = ms.to_ambient(traj[-1])                      # decode endpoint back to gene space (inverse PCA)
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge


def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


class ManifoldSteer:
    """Fit a whitened-PCA manifold from cells, then steer source cells on it via local-tangent projection.

    Parameters
    ----------
    n_pcs : int         dimensionality of the working (whitened PCA) space.
    k_tangent : int     neighbours used to estimate the local tangent at a point.
    m_tangent : int     local tangent dimension (top-m local PCs the step is projected onto).
    step_frac : float   integration step length as a fraction of d0 (median nearest-neighbour distance).
    random_state : int
    """

    def __init__(self, n_pcs=20, k_tangent=100, m_tangent=10, step_frac=0.25, random_state=0):
        self.n_pcs = n_pcs
        self.k_tangent = k_tangent
        self.m_tangent = m_tangent
        self.step_frac = step_frac
        self.random_state = random_state

    # ------------------------------------------------------------------ fit / spaces
    def fit(self, X):
        """X: (n_cells, n_features). Builds PCA+whiten, the reference cloud, and the step scale d0."""
        X = np.asarray(X, np.float64)
        self.mean_ = X.mean(0)
        d = min(self.n_pcs, X.shape[1], X.shape[0] - 1)
        self.pca_ = PCA(d, random_state=self.random_state).fit(X - self.mean_)
        self.scaler_ = StandardScaler().fit(self.pca_.transform(X - self.mean_))
        self.Z_ = self.transform(X)                                   # reference cells in working space
        self.dim_ = self.Z_.shape[1]
        self._nn = NearestNeighbors(n_neighbors=self.k_tangent).fit(self.Z_)
        self._nn1 = NearestNeighbors(n_neighbors=1).fit(self.Z_)
        dd = NearestNeighbors(n_neighbors=2).fit(self.Z_).kneighbors(self.Z_)[0][:, 1]
        self.d0_ = float(np.median(dd))
        self.step_ = self.step_frac * self.d0_
        return self

    def transform(self, X):
        """Feature space -> whitened working space."""
        return self.scaler_.transform(self.pca_.transform(np.asarray(X, np.float64) - self.mean_))

    def to_ambient(self, Z):
        """Whitened working space -> original feature space (inverse whiten + inverse PCA). The model-free decode."""
        return self.pca_.inverse_transform(self.scaler_.inverse_transform(np.asarray(Z, np.float64))) + self.mean_

    # ------------------------------------------------------------------ geometry
    def tangent_bases(self, Zq, ref_Z=None):
        """Top-m local PCs (the tangent basis) at each query point, from its k nearest reference cells."""
        ref = self.Z_ if ref_Z is None else ref_Z
        nn = self._nn if ref_Z is None else NearestNeighbors(n_neighbors=self.k_tangent).fit(ref)
        _, idx = nn.kneighbors(np.atleast_2d(Zq))
        out = []
        for i in range(len(idx)):
            C = ref[idx[i]] - ref[idx[i]].mean(0)
            out.append(np.linalg.svd(C, full_matrices=False)[2][: self.m_tangent])
        return out

    def project(self, Zq, vecs, ref_Z=None):
        """Project each vector onto the local tangent at its query point (the core on-manifold move)."""
        Zq = np.atleast_2d(Zq); vecs = np.atleast_2d(vecs)
        out = np.zeros_like(vecs)
        for i, V in enumerate(self.tangent_bases(Zq, ref_Z=ref_Z)):
            out[i] = V.T @ (V @ vecs[i])
        return out

    def off_manifold_ratio(self, Z):
        """Mean distance to the nearest reference cell, in units of d0 (1.0 ~ a typical real cell; >1 = off-manifold)."""
        return float(self._nn1.kneighbors(np.atleast_2d(Z))[0][:, 0].mean() / self.d0_)

    # ------------------------------------------------------------------ directions
    def ascent_direction(self, y, alpha=10.0):
        """Constant linear ascent direction of a scalar (e.g. pseudotime): unit ridge coef of y ~ Z."""
        y = np.asarray(y, np.float64)
        y = (y - y.mean()) / (y.std() + 1e-9)
        return _unit(Ridge(alpha=alpha).fit(self.Z_, y).coef_)

    def fate_direction(self, source, target):
        """Unit direction from the source population's centroid to the target population's centroid.
        `source`/`target` are boolean masks or index arrays over the fitted reference cells."""
        return _unit(self.Z_[target].mean(0) - self.Z_[source].mean(0))

    # ------------------------------------------------------------------ steering
    def steer(self, Z0, direction, n_steps, project=True, ref_Z=None):
        """Integrate source points Z0 (n_src, dim) along `direction` for n_steps equal-length unit steps.

        direction : (dim,) constant vector, (n_src, dim) per-cell, or callable Z -> (n_src, dim) field.
        project   : if True, project each step onto the local tangent (on-manifold). If False, plain Euler
                    (the ablation that drifts off-manifold).
        Returns   : trajectory array (n_steps + 1, n_src, dim).
        """
        Z0 = np.atleast_2d(np.asarray(Z0, np.float64))
        traj = [Z0.copy()]
        x = Z0.copy()
        for _ in range(n_steps):
            D = direction(x) if callable(direction) else np.broadcast_to(np.atleast_2d(direction), x.shape).copy()
            if project:
                D = self.project(x, D, ref_Z=ref_Z)
            D = _unit(D)
            x = x + self.step_ * D
            traj.append(x.copy())
        return np.stack(traj)

    def steer_to_fate(self, source, target, n_steps=8, project=True):
        """Convenience: steer the source cells toward the target population's centroid, on-manifold.
        Returns (trajectory, endpoint_ambient)."""
        d = self.fate_direction(source, target)
        traj = self.steer(self.Z_[source], d, n_steps, project=project)
        return traj, self.to_ambient(traj[-1])

