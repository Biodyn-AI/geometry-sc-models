# manifold_steer

On-manifold steering of single-cell trajectories via **local-tangent projection** — the packaged, model-free
form of the steering method validated in [`../RESULTS.md`](../RESULTS.md).

You have a cell state and a direction you want to push it (toward a fate, up a pseudotime gradient, along a
perturbation axis). Moving along that direction in gene-expression space walks straight off the manifold of real
cells into biologically meaningless states. `manifold_steer` fixes this with one move: **at every integration
step, project the direction onto the local tangent of the data** — the top-*m* principal directions of the
cell's nearest real neighbours. The step keeps its on-manifold advance and drops its off-manifold component, so
the path follows the curved trajectory of real cells instead of cutting across it.

The method is **model-free**: it needs only a cell-by-gene matrix (or any embedding). Per Gate 0 in
`../RESULTS.md`, it works at least as well in raw log-normalized counts as in any single-cell foundation-model
embedding — the useful structure is the data manifold, not the model.

## Install

Pure `numpy` + `scikit-learn`. No scanpy/anndata/torch. Drop `manifold_steer.py` into your project and import it.

```python
from manifold_steer import ManifoldSteer
```

## Quick start

```python
ms = ManifoldSteer(n_pcs=20).fit(X)                 # X: cells x genes (log-normalized), or any embedding

# steer along a pseudotime gradient:
d    = ms.ascent_direction(pseudotime)              # constant linear ascent direction
traj = ms.steer(ms.transform(X[early_mask]), d, n_steps=8)   # (9, n_early, n_pcs), stays on-manifold
expr = ms.to_ambient(traj[-1])                      # decode the endpoint back to gene space (inverse PCA)

# or steer toward a target population (in-silico fate biasing):
traj, endpoint_expr = ms.steer_to_fate(source=early_mask, target=(clusters == "Erythroid"), n_steps=12)
```

Run `python example.py` for a self-contained synthetic demo (no data needed).

## API

| method | purpose |
|---|---|
| `fit(X)` | build the whitened-PCA manifold, the reference cloud, and the step scale `d0` |
| `transform(X)` / `to_ambient(Z)` | map gene space ↔ working (whitened-PC) space; `to_ambient` is the decode |
| `ascent_direction(y)` | constant linear ascent direction of a scalar (e.g. pseudotime) |
| `fate_direction(source, target)` | unit direction from a source population's centroid to a target's |
| `steer(Z0, direction, n_steps, project=True)` | integrate source points along `direction`; `project=True` is the method, `False` the ablation. `direction` may be a constant vector, per-cell array, or callable field |
| `steer_to_fate(source, target, n_steps)` | convenience: fate_direction + on-manifold steer + decode |
| `tangent_bases(Zq)` / `project(Zq, vecs)` | the local tangent basis and the projection operator, exposed for custom use |
| `off_manifold_ratio(Z)` | mean distance to the nearest real cell, in units of `d0` (≈1 on-manifold, >1 off) |

Parameters: `n_pcs` (working dim, 20), `k_tangent` (neighbours for the tangent, 100), `m_tangent` (tangent
dimension, 10), `step_frac` (step = `step_frac`·`d0`, 0.25).

## Scope — what it does and does not do

Read this before using it. Established honestly in `../RESULTS.md`:

- ✅ **Interpolates along observed geometry.** It routes cells to real fates through real intermediate states
  (LARRY benchmark: reaches all 6 terminal fates, with biologically realistic intermediate mixtures).
- ✅ **Model-free.** Counts work as well as or better than a foundation model's embedding (Gate 0).
- ❌ **Does not generate off-data states.** If a population is deleted from the manifold, steering cannot
  reconstruct it — it interpolates, it does not extrapolate (H-A3).
- ❌ **Does not predict interventions.** It moves a cell *toward* a fate; it does not tell you which gene/TF
  perturbation would cause that move (H-A2).
- ⚠️ **Long-range constant-direction steering drifts.** A single fixed direction over many steps eventually
  strays even when projected; for long trajectories use a *local* field (re-estimate the direction each step —
  pass a `callable` to `steer`) rather than a constant vector.

## Benchmarks

Both use the project's cached data via `_data.py` (the class itself has no data dependency).

- **`benchmark_setty.py`** — ablation ladder on Setty hematopoiesis, model-free counts. Shows local-tangent
  projection holds the off-manifold ratio flat (0.84→0.89 over the path) while plain-linear drifts
  (0.92→1.59), and beats a graph-retraction corrector. `results/benchmark_setty.json`.
- **`benchmark_larry.py`** — cross-dataset real-fate validation on Weinreb LARRY (a dataset the method never
  saw; fates are real clonal terminal populations, not cluster labels). Steering undifferentiated cells toward
  each fate reaches the correct real fate **6/6**, through a path that stays closer to real cells than
  plain-linear (mean path off-manifold ratio 1.54 vs 1.91) and yields realistic intermediate mixtures.
  `results/benchmark_larry.json`.

```sh
../../../.venv/bin/python benchmark_setty.py
../../../.venv/bin/python benchmark_larry.py
python example.py                                    # standalone, no data
```

## Provenance

Packaged from `route_geometry_hypotheses` (see `../RESULTS.md`, `../HYPOTHESES.md`). The method and its
model-free character were established there and in `../../route_steering/LOCAL_STEERING_RESULTS.md` and
`../../route_steering2/RESULTS.md`. This directory is a clean, standalone reimplementation — it does not import
any project code, so it can be lifted out as-is.

