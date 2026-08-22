"""Stage-2 LET adaptor (handoff §2, pipeline §5 / §6 Phase 6).

Latent Embedding Transfer: a small linear head g_theta: f(cell) -> z in R^d_latent whose pairwise
arc-cosine distances match the biological-ruler distances, plus a reconstruction regulariser:

    z = W_enc (x - b)
    d_hat_ij = beta * arccos(cos(z_i, z_j))
    L = || d_hat - d_target ||^2  +  alpha * || W_enc^T z + b - x ||^2

Trained on the INTERNAL panel only. The frozen head (input scaler + W_enc + b + beta) transfers to the
external holdout with ZERO retraining (transform()).

Rulers (target distances):
  celltype (scGPT):   tiered lineage proxy  0 same cell_type / 1 same tissue / 2 different tissue  (route_geo v1)
  cellcycle (Genef.): continuous  d = || [s_i,g2m_i] - [s_j,g2m_j] ||  (Tirosh S/G2M score space)
"""
import numpy as np
import torch
import torch.nn as nn


def ruler_targets(ruler, idx):
    """(B,B) target-distance matrix for a batch of cell indices, per ruler kind. Scaled to ~[0,1]."""
    if ruler["kind"] == "celltype":
        ct = ruler["cell_type"][idx]
        ti = ruler["tissue"][idx]
        same_t = ct[:, None] == ct[None, :]
        same_s = ti[:, None] == ti[None, :]
        D = np.where(same_t, 0.0, np.where(same_s, 1.0, 2.0)) / 2.0
        return D.astype(np.float32)
    s = ruler["s_score"][idx]
    g = ruler["g2m_score"][idx]
    coord = np.stack([s, g], 1)
    D = np.linalg.norm(coord[:, None, :] - coord[None, :, :], axis=2)
    D = D / (D.max() + 1e-8)
    return D.astype(np.float32)


class LETHead:
    def __init__(self, d_latent=10, alpha=0.05, epochs=400, batch=256, lr=5e-3, seed=0, device="cpu"):
        self.d_latent = d_latent
        self.alpha = alpha
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.seed = seed
        self.device = device

    def fit(self, X, ruler, verbose=False):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.nan_to_num(np.asarray(X, dtype=np.float32))
        self.mu = X.mean(0, keepdims=True)
        self.sd = X.std(0, keepdims=True) + 1e-6
        Xz = (X - self.mu) / self.sd
        n, D = Xz.shape
        Xt = torch.tensor(Xz, device=self.device)
        self.W = nn.Parameter(torch.empty(self.d_latent, D, device=self.device))
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(D, device=self.device))
        self.beta = nn.Parameter(torch.tensor(1.0, device=self.device))
        opt = torch.optim.Adam([self.W, self.b, self.beta], lr=self.lr)
        for ep in range(self.epochs):
            idx = rng.choice(n, size=min(self.batch, n), replace=False)
            xb = Xt[idx]
            z = (xb - self.b) @ self.W.T                         # (B, d_latent)
            zc = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            cos = (zc @ zc.T).clamp(-1 + 1e-6, 1 - 1e-6)
            dhat = self.beta.abs() * torch.arccos(cos)
            dtar = torch.tensor(ruler_targets(ruler, idx), device=self.device)
            fit = ((dhat - dtar) ** 2).mean()
            recon = ((z @ self.W + self.b) - xb).pow(2).mean()   # W_enc^T z + b
            loss = fit + self.alpha * recon
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and (ep % 100 == 0 or ep == self.epochs - 1):
                print(f"      LET ep{ep:4d} loss={loss.item():.4f} fit={fit.item():.4f} recon={recon.item():.4f}",
                      flush=True)
        self.W_ = self.W.detach().cpu().numpy()
        self.b_ = self.b.detach().cpu().numpy()
        return self

    def transform(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float32))
        Xz = (X - self.mu) / self.sd
        return (Xz - self.b_) @ self.W_.T
