"""One training run, from a TrainConfig to a run folder. Every model in the project goes through here."""
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from flybrain.data.subcircuit import load_subcircuit
from flybrain.models.baselines import ChebGRUBaseline, MeanPredictor, MLPBaseline
from flybrain.models.moe import SpectralMoE, moe_full, moe_linear, moe_single
from flybrain.paths import PROCESSED, RUNS
from flybrain.runlog import make_run_dir, write_metrics
from flybrain.spectral.bands import equal_count_bands
from flybrain.spectral.basis import load_basis
from flybrain.spectral.laplacian import signed_symmetric_laplacian
from flybrain.train.dataset import (active_mask, condition_splits, ever_active_mask, load_bundle,
                                    neuron_splits, noise_ceiling, trial_mean_rates)
from flybrain.train.metrics import summarise

MOE_PRESETS = {"moe_full": moe_full, "moe_linear": moe_linear, "moe_single": moe_single}
N_EXPERTS = 4


@dataclass(frozen=True)
class TrainConfig:
    model: str
    basis: str = "magnetic"
    graph: str = "real"
    loss_mask: str = "ever_active"
    seed: int = 0
    lr: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 40
    patience: int = 8
    weight_decay: float = 1e-4
    n_conditions: int | None = None
    device: str = "cuda"
    tag: str = ""


@dataclass
class Tensors:
    stim_train: np.ndarray
    target_train: np.ndarray
    stim_val: np.ndarray
    target_val: np.ndarray
    stim_test: np.ndarray
    target_test: np.ndarray
    loss_mask: np.ndarray
    metric_masks: dict
    ceiling: np.ndarray
    mn9_idx: list
    target_std: float
    n_bins: int
    stim_bins: int
    pred_idx: np.ndarray
    fill_trace: np.ndarray | None = None      # (T, N) training-mean trace for columns a model does not predict


def _mask_to_bool(idx_or_mask, n) -> np.ndarray:
    arr = np.asarray(idx_or_mask)
    if arr.dtype == bool:
        return arr
    out = np.zeros(n, dtype=bool)
    out[arr] = True
    return out


def effective_loss_mask(policy: str, observed: np.ndarray, ever_active: np.ndarray,
                        n_neurons: int) -> np.ndarray:
    obs = _mask_to_bool(observed, n_neurons)
    if policy == "all_observed":
        return obs
    if policy == "ever_active":
        return obs & _mask_to_bool(ever_active, n_neurons)
    raise ValueError(f"unknown loss_mask policy: {policy!r}")


def assert_basis_matches(basis_nodes: np.ndarray, subcircuit_nodes: np.ndarray) -> None:
    if basis_nodes is None or not np.array_equal(np.asarray(basis_nodes), np.asarray(subcircuit_nodes)):
        raise ValueError("basis does not belong to this subcircuit: stored nodes differ")


def basis_fingerprint(model) -> str:
    """Identify the basis and wiring a checkpoint was trained against.

    The MoE's basis buffers are non-persistent, so a state dict cannot detect being loaded against the
    wrong eigenbasis. This hashes the parts that must match.
    """
    if not hasattr(model, "mu"):
        return "n/a"
    h = hashlib.sha256()
    for name in ("mu", "band_of_mode", "input_idx", "group_of_input"):
        h.update(np.ascontiguousarray(getattr(model, name).detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def train_model(model, tensors: Tensors, cfg: TrainConfig) -> dict:
    if isinstance(model, MeanPredictor):
        return {"train_loss": [], "val_loss": [], "best_epoch": 0, "stopped_early": False}
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.max_epochs, 1))
    mask = torch.tensor(tensors.loss_mask[tensors.pred_idx], device=device)
    rng = np.random.default_rng(cfg.seed)
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0, "stopped_early": False}
    best = {"loss": float("inf"), "state": None, "epoch": 0}

    def batch_loss(stim_np, target_np, idx):
        stim = torch.tensor(stim_np[idx], device=device)
        target = torch.tensor(target_np[idx][:, :, tensors.pred_idx], device=device)
        pred = _forward(model, stim, tensors)
        diff = (pred.permute(0, 2, 1) - target / tensors.target_std)[:, :, mask]
        return (diff ** 2).mean()

    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(tensors.stim_train))
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(tensors.stim_train, tensors.target_train, idx)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        model.eval()
        with torch.no_grad():
            val = [float(batch_loss(tensors.stim_val, tensors.target_val,
                                    np.arange(s, min(s + cfg.batch_size, len(tensors.stim_val)))))
                   for s in range(0, len(tensors.stim_val), cfg.batch_size)]
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(float(np.mean(val)))
        if history["val_loss"][-1] < best["loss"] - 1e-6:
            best = {"loss": history["val_loss"][-1],
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch + 1}
        elif epoch + 1 - best["epoch"] >= cfg.patience:
            history["stopped_early"] = True
            break
    if best["state"] is None and history["val_loss"]:
        raise RuntimeError(f"training diverged: validation loss was never finite "
                           f"(val_loss={history['val_loss']}); lower the learning rate")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    history["best_epoch"] = best["epoch"]
    return history


def _forward(model, stim: torch.Tensor, tensors: Tensors) -> torch.Tensor:
    if isinstance(model, MLPBaseline):
        return model(stim)
    return model(stim, tensors.n_bins, tensors.stim_bins)


def predict(model, stim: np.ndarray, tensors: Tensors, cfg: TrainConfig) -> np.ndarray:
    n_neurons = len(tensors.loss_mask)
    out = np.full((len(stim), tensors.n_bins, n_neurons), np.nan, dtype=np.float32)
    if tensors.fill_trace is not None:
        # columns this model does not predict get the training-mean trace, so every model is scored on
        # identical neurons instead of silently on an easier subset
        out[:] = tensors.fill_trace.astype(np.float32)[None]
    if isinstance(model, MeanPredictor):
        out[:, :, :] = model.predict(len(stim))
        return out
    device = torch.device(cfg.device)
    model.to(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        for start in range(0, len(stim), cfg.batch_size):
            chunk = torch.tensor(stim[start:start + cfg.batch_size], device=device)
            pred = _forward(model, chunk, tensors).permute(0, 2, 1).cpu().numpy()
            out[start:start + len(chunk)][:, :, tensors.pred_idx] = pred * tensors.target_std
    return out


def prepare(cfg: TrainConfig) -> tuple[Tensors, dict]:
    bundle = load_bundle()
    sub = load_subcircuit(PROCESSED / "subcircuit.npz")
    n = bundle.n_neurons
    tr, va, te = condition_splits(bundle.n_conditions, seed=0)
    if cfg.n_conditions:
        tr, va, te = tr[:cfg.n_conditions], va[:max(cfg.n_conditions // 4, 2)], te[:max(cfg.n_conditions // 4, 2)]
    protected = np.concatenate([np.concatenate(list(sub.input_local.values())), sub.mn9_local])
    observed_idx, heldout_idx = neuron_splits(sub.d_in, protected, seed=0)
    observed = _mask_to_bool(observed_idx, n)
    heldout = _mask_to_bool(heldout_idx, n)
    ever = ever_active_mask(bundle, tr)
    loss_mask = effective_loss_mask(cfg.loss_mask, observed, ever, n)

    target_train = trial_mean_rates(bundle, tr)
    target_val = trial_mean_rates(bundle, va)
    target_test = trial_mean_rates(bundle, te)
    act_test = active_mask(target_test)
    masks = {"observed_active": observed & act_test, "heldout_active": heldout & act_test}
    tensors = Tensors(
        stim_train=bundle.conditions[tr], target_train=target_train,
        stim_val=bundle.conditions[va], target_val=target_val,
        stim_test=bundle.conditions[te], target_test=target_test,
        loss_mask=loss_mask, metric_masks=masks, ceiling=noise_ceiling(bundle, te),
        mn9_idx=[int(i) for i in sub.mn9_local], n_bins=bundle.n_bins,
        stim_bins=int(round(bundle.meta["stimulus"]["t_on_ms"] / bundle.meta["bin_ms"])),
        target_std=float(target_train[:, :, loss_mask].std()), pred_idx=np.arange(n),
        fill_trace=target_train.mean(axis=0))
    ctx = {"subcircuit": sub, "bundle_meta": bundle.meta}
    return tensors, ctx


def build_model(cfg: TrainConfig, tensors: Tensors, ctx: dict):
    sub = ctx["subcircuit"]
    meta = ctx["bundle_meta"]
    input_local = np.concatenate([np.asarray(sub.input_local[g]) for g in meta["group_names"]])
    group_of_input = np.concatenate([np.full(len(sub.input_local[g]), i)
                                    for i, g in enumerate(meta["group_names"])])
    w = ctx.get("W", sub.W)
    if cfg.model == "mean":
        return MeanPredictor.fit(tensors.target_train)
    if cfg.model == "mlp":
        idx = np.flatnonzero(tensors.loss_mask)
        tensors.pred_idx = idx
        return MLPBaseline(n_stim=tensors.stim_train.shape[1], n_out=len(idx), n_bins=tensors.n_bins)
    if cfg.model == "chebgru":
        lap = signed_symmetric_laplacian(sp.csr_matrix(w).toarray())
        return ChebGRUBaseline(sp.csr_matrix(lap), input_local, group_of_input)
    if cfg.model in MOE_PRESETS:
        basis = ctx["basis"]
        assert_basis_matches(basis.nodes, sub.nodes)
        bands = equal_count_bands(len(basis.eigenvalues), N_EXPERTS)
        return SpectralMoE(basis.eigenvalues, basis.eigenvectors, bands, input_local,
                           group_of_input, MOE_PRESETS[cfg.model]())
    raise ValueError(f"unknown model: {cfg.model!r}")


def train_one(cfg: TrainConfig, run_root: Path = RUNS) -> dict:
    if cfg.graph == "rewired" and cfg.model in ("mlp", "mean"):
        raise ValueError(f"graph='rewired' is meaningless for model {cfg.model!r}: it ignores the graph, "
                         "so the run would be a mislabelled copy of the real-graph one")
    # the seed must govern model initialisation, not just shuffling: results are paired by seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tensors, ctx = prepare(cfg)
    if cfg.graph == "rewired":
        ctx["W"] = sp.load_npz(PROCESSED / f"rewired/W_seed{cfg.seed}.npz")
    if cfg.model in MOE_PRESETS:
        if cfg.graph == "rewired":
            ctx["basis"] = load_basis(PROCESSED / f"rewired/basis_{cfg.basis}_seed{cfg.seed}.npz")
        else:
            ctx["basis"] = load_basis(PROCESSED / "bases" / f"{cfg.basis}.npz")
    name = f"{cfg.model}_{cfg.basis}_{cfg.graph}_{cfg.loss_mask}_seed{cfg.seed}{cfg.tag}"
    run_dir = make_run_dir(f"p2_{name}", asdict(cfg) | {"tf32": True, "model_dtype": "float32"}, runs_root=run_root)
    model = build_model(cfg, tensors, ctx)
    history = train_model(model, tensors, cfg)
    pred = predict(model, tensors.stim_test, tensors, cfg)
    scored = summarise(pred, tensors.target_test, tensors.metric_masks, tensors.ceiling, tensors.mn9_idx)
    scored_cols = np.zeros(len(tensors.loss_mask), dtype=bool)
    for m in tensors.metric_masks.values():
        scored_cols |= m
    unpredicted = int((scored_cols & ~np.isin(np.arange(len(scored_cols)), tensors.pred_idx)).sum())
    # a per-neuron mean has a free parameter per neuron, and the MLP has an output per neuron, so
    # neither can generalise to neurons it never saw: their held-out scores are null, not oracle scores
    if cfg.model in ("mlp", "mean"):
        for key in list(scored):
            if "heldout" in key and key.startswith(("r2_", "pooled_", "normalised_")):
                scored[key] = None
    params = 0 if isinstance(model, MeanPredictor) else sum(p.numel() for p in model.parameters())
    metrics = {**scored, "history": history, "n_parameters": params,
               "n_loss_mask": int(tensors.loss_mask.sum()), "n_scored_without_prediction": unpredicted, "target_std_hz": tensors.target_std,
               "config": asdict(cfg)}
    write_metrics(run_dir, metrics)
    if not isinstance(model, MeanPredictor):
        torch.save({"state_dict": model.state_dict(), "fingerprint": basis_fingerprint(model)},
                   run_dir / "model.pt")
    metrics["run_dir"] = str(run_dir)
    return metrics


def load_run_model(run_dir: Path):
    """Rebuild a trained model from a run folder: its resolved config plus model.pt."""
    run_dir = Path(run_dir)
    cfg_blob = json.loads((run_dir / "config.json").read_text())["config"]
    # the folder's config also records run-environment keys (tf32, model_dtype) that are not TrainConfig fields
    cfg = TrainConfig(**{k: v for k, v in cfg_blob.items() if k in TrainConfig.__dataclass_fields__})
    if cfg.graph == "rewired" and cfg.model not in MOE_PRESETS:
        # train_one hands the rewired W to every model; rebuilding a rewired baseline here on the real graph
        # would be silent (its fingerprint is "n/a"), and nothing needs to reload one, so refuse
        raise ValueError(f"cannot faithfully rebuild model {cfg.model!r} trained on a rewired graph")
    tensors, ctx = prepare(cfg)
    if cfg.model in MOE_PRESETS:
        if cfg.graph == "rewired":
            ctx["W"] = sp.load_npz(PROCESSED / f"rewired/W_seed{cfg.seed}.npz")
            ctx["basis"] = load_basis(PROCESSED / f"rewired/basis_{cfg.basis}_seed{cfg.seed}.npz")
        else:
            ctx["basis"] = load_basis(PROCESSED / "bases" / f"{cfg.basis}.npz")
    model = build_model(cfg, tensors, ctx)
    blob = torch.load(run_dir / "model.pt", map_location="cpu")
    if "fingerprint" not in blob:
        raise RuntimeError(f"{run_dir}: checkpoint predates the basis-fingerprint guard, so it cannot be "
                           "verified against the eigenbasis")
    # weights loaded onto a different eigenbasis would give plausible-looking garbage, so refuse
    if blob["fingerprint"] != basis_fingerprint(model):
        raise RuntimeError(f"{run_dir}: checkpoint was trained against basis fingerprint {blob['fingerprint']}, "
                           f"but the rebuilt model has {basis_fingerprint(model)}")
    model.load_state_dict(blob["state_dict"])
    return model, tensors, cfg
