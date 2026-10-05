import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flybrain.models.moe import SpectralMoE, moe_full
from flybrain.paths import PROCESSED, RUNS
from flybrain.spectral.bands import equal_count_bands
from flybrain.models.baselines import MeanPredictor
from flybrain.models.baselines import ChebGRUBaseline, MLPBaseline
from flybrain.train.loop import (TrainConfig, Tensors, assert_basis_matches, basis_fingerprint,
                                 build_model, effective_loss_mask, load_run_model, predict, train_model, train_one)

N, M, T = 12, 12, 6


def synthetic(seed=0, n_train=16, n_val=8):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, M, generator=g) + 1j * torch.randn(N, M, generator=g)
    u, _ = torch.linalg.qr(a)
    mu = torch.linspace(0.3, 1.7, M)
    torch.manual_seed(1000 + seed)                             # teacher and student inits reproducible
    model = SpectralMoE(mu, u.to(torch.complex64), equal_count_bands(M, 4),
                        np.arange(3), np.array([0, 1, 4]), moe_full())
    rng = np.random.default_rng(seed)
    stim_tr = (rng.random((n_train, 5)) * 200).astype(np.float32)
    stim_va = (rng.random((n_val, 5)) * 200).astype(np.float32)
    with torch.no_grad():
        tgt_tr = model(torch.tensor(stim_tr), T, T // 2).permute(0, 2, 1).numpy()
        tgt_va = model(torch.tensor(stim_va), T, T // 2).permute(0, 2, 1).numpy()
    mask = np.ones(N, dtype=bool)
    tensors = Tensors(stim_train=stim_tr, target_train=tgt_tr, stim_val=stim_va, target_val=tgt_va,
                      stim_test=stim_va, target_test=tgt_va, loss_mask=mask,
                      metric_masks={"observed_active": mask, "heldout_active": np.zeros(N, bool)},
                      ceiling=np.ones(N), mn9_idx=[0], target_std=1.0, n_bins=T, stim_bins=T // 2,
                      pred_idx=np.arange(N))
    torch.manual_seed(seed)
    fresh = SpectralMoE(mu, u.to(torch.complex64), equal_count_bands(M, 4),
                        np.arange(3), np.array([0, 1, 4]), moe_full())
    return fresh, tensors


def test_effective_loss_mask_never_includes_heldout_neurons():
    # review focus 5: a neuron can be both held out and ever-active; held out must always win
    observed = np.array([True, True, False, False])
    ever = np.array([True, False, True, True])
    ea = effective_loss_mask("ever_active", observed, ever, 4)
    allobs = effective_loss_mask("all_observed", observed, ever, 4)
    assert ea.tolist() == [True, False, False, False]
    assert allobs.tolist() == [True, True, False, False]
    assert not ea[2] and not allobs[2]


def test_unknown_loss_mask_policy_raises():
    with pytest.raises(ValueError, match="loss_mask"):
        effective_loss_mask("everything", np.ones(2, bool), np.ones(2, bool), 2)


def test_basis_node_mismatch_is_refused():
    # review focus 4: a basis from a different subcircuit would silently mis-pair rows with columns
    assert_basis_matches(np.arange(5), np.arange(5))
    with pytest.raises(ValueError, match="does not belong"):
        assert_basis_matches(np.arange(5), np.array([0, 1, 2, 3, 9]))


def test_training_reduces_loss_on_a_learnable_target():
    model, tensors = synthetic()
    cfg = TrainConfig(model="moe_full", seed=0, lr=3e-3, batch_size=8, max_epochs=6, patience=6,
                      device="cpu")
    hist = train_model(model, tensors, cfg)
    assert len(hist["train_loss"]) <= 6
    assert hist["train_loss"][-1] < 0.6 * hist["train_loss"][0]


def test_training_is_deterministic_per_seed():
    a_model, a_tensors = synthetic()
    b_model, b_tensors = synthetic()
    cfg = TrainConfig(model="moe_full", seed=3, lr=1e-3, batch_size=8, max_epochs=3, patience=3,
                      device="cpu")
    ha = train_model(a_model, a_tensors, cfg)
    hb = train_model(b_model, b_tensors, cfg)
    assert ha["train_loss"] == hb["train_loss"]
    hc = train_model(*synthetic(), dataclasses.replace(cfg, seed=4))
    assert hc["train_loss"] != ha["train_loss"]


def test_early_stopping_halts_and_restores_best_weights():
    model, tensors = synthetic()
    tensors.target_val = np.zeros_like(tensors.target_val)     # val loss cannot improve meaningfully
    cfg = TrainConfig(model="moe_full", seed=0, lr=5e-2, batch_size=8, max_epochs=20, patience=2,
                      device="cpu")
    hist = train_model(model, tensors, cfg)
    assert hist["stopped_early"]
    assert len(hist["val_loss"]) < 20
    assert hist["best_epoch"] <= len(hist["val_loss"])
    # best weights were restored: re-evaluating validation loss reproduces the minimum of the history
    assert hist["best_epoch"] == int(np.argmin(hist["val_loss"])) + 1
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(tensors.stim_val), T, T // 2).permute(0, 2, 1).numpy()
    assert float(((pred - tensors.target_val) ** 2).mean()) == pytest.approx(min(hist["val_loss"]), rel=1e-4)


def test_loss_ignores_neurons_outside_the_mask():
    model, tensors = synthetic()
    tensors.loss_mask = np.zeros(N, dtype=bool)
    tensors.loss_mask[:3] = True
    cfg = TrainConfig(model="moe_full", seed=0, lr=1e-3, batch_size=8, max_epochs=1, patience=1,
                      device="cpu")
    baseline = train_model(model, tensors, cfg)["train_loss"][0]
    model2, tensors2 = synthetic()
    tensors2.loss_mask = tensors.loss_mask.copy()
    tensors2.target_train = tensors2.target_train.copy()
    tensors2.target_train[:, :, 5:] += 1000.0                  # garbage outside the mask
    changed = train_model(model2, tensors2, cfg)["train_loss"][0]
    assert abs(baseline - changed) < 1e-6


def test_basis_fingerprint_identifies_basis_and_wiring():
    a, _ = synthetic()
    b, _ = synthetic()
    assert basis_fingerprint(a) == basis_fingerprint(b)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, M, generator=g) + 1j * torch.randn(N, M, generator=g)
    u, _ = torch.linalg.qr(x)
    other_wiring = SpectralMoE(torch.linspace(0.3, 1.7, M), u.to(torch.complex64),
                               equal_count_bands(M, 4), np.arange(3), np.array([0, 1, 4]), moe_full())
    other_wiring.input_idx = torch.tensor([0, 1, 5])
    assert basis_fingerprint(other_wiring) != basis_fingerprint(a)
    permuted = SpectralMoE(torch.linspace(0.3, 1.7, M).flip(0), u.to(torch.complex64),
                           equal_count_bands(M, 4), np.arange(3), np.array([0, 1, 4]), moe_full())
    assert basis_fingerprint(permuted) != basis_fingerprint(a)
    assert basis_fingerprint(MeanPredictor(np.zeros((T, N)))) == "n/a"


def test_train_model_raises_on_diverged_validation_loss():
    model, tensors = synthetic()
    tensors.target_val = np.full_like(tensors.target_val, np.nan)
    cfg = TrainConfig(model="moe_full", max_epochs=2, patience=2, batch_size=8, device="cpu")
    with pytest.raises(RuntimeError, match="diverged"):
        train_model(model, tensors, cfg)


def test_predict_rescales_by_target_std():
    model, tensors = synthetic()
    tensors.target_std = 7.5
    cfg = TrainConfig(model="moe_full", batch_size=4, device="cpu")
    out = predict(model, tensors.stim_val, tensors, cfg)
    model.eval()
    with torch.no_grad():
        ref = model(torch.tensor(tensors.stim_val), T, T // 2).permute(0, 2, 1).numpy() * 7.5
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_mlp_predict_aligns_columns_and_fills_the_rest_with_the_training_mean():
    _, tensors = synthetic()
    tensors.pred_idx = np.array([2, 5, 9])
    tensors.fill_trace = np.arange(T * N, dtype=np.float32).reshape(T, N)
    tensors.target_std = 3.0
    torch.manual_seed(0)
    mlp = MLPBaseline(n_stim=5, n_out=3, n_bins=T)
    cfg = TrainConfig(model="mlp", batch_size=4, device="cpu")
    out = predict(mlp, tensors.stim_val, tensors, cfg)
    with torch.no_grad():
        direct = mlp.eval()(torch.tensor(tensors.stim_val)).permute(0, 2, 1).numpy() * 3.0
    np.testing.assert_allclose(out[:, :, [2, 5, 9]], direct, rtol=1e-5, atol=1e-6)
    other = [i for i in range(N) if i not in (2, 5, 9)]
    for i in range(len(out)):
        np.testing.assert_array_equal(out[i][:, other], tensors.fill_trace[:, other])
    assert np.isfinite(out).all()


def test_build_model_chebgru_uses_the_supplied_graph():
    import scipy.sparse as sp
    from flybrain.train.loop import signed_symmetric_laplacian
    rng = np.random.default_rng(0)
    real = sp.csr_matrix(rng.random((N, N)) * (rng.random((N, N)) < 0.3))
    rewired = sp.csr_matrix(rng.random((N, N)) * (rng.random((N, N)) < 0.3))

    class Sub:
        W = real
        input_local = {"a": np.array([0, 1]), "b": np.array([4])}

    ctx = {"subcircuit": Sub(), "bundle_meta": {"group_names": ["a", "b"]}, "W": rewired}
    _, tensors = synthetic()
    cfg = TrainConfig(model="chebgru", graph="rewired", device="cpu")
    with_ctx = build_model(cfg, tensors, ctx)
    plain = build_model(cfg, tensors, {k: v for k, v in ctx.items() if k != "W"})
    assert isinstance(with_ctx, ChebGRUBaseline)
    import scipy.sparse as sp2
    lap_rew = sp2.csr_matrix(signed_symmetric_laplacian(rewired.toarray())) - sp2.eye(N)
    lap_real = sp2.csr_matrix(signed_symmetric_laplacian(real.toarray())) - sp2.eye(N)
    assert np.abs((lap_rew - lap_real).toarray()).max() > 1e-3
    np.testing.assert_allclose(with_ctx.lap_sparse.to_dense().numpy(), lap_rew.toarray(), atol=1e-5)
    np.testing.assert_allclose(plain.lap_sparse.to_dense().numpy(), lap_real.toarray(), atol=1e-5)


def test_rewired_graph_is_refused_for_graph_blind_models(tmp_path):
    for name in ("mlp", "mean"):
        with pytest.raises(ValueError, match="rewired"):
            train_one(TrainConfig(model=name, graph="rewired", device="cpu"), run_root=tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real-data run at N=5000 is only practical on GPU")
def test_train_one_is_seed_deterministic(tmp_path):
    kw = dict(model="chebgru", n_conditions=24, max_epochs=2, patience=2, device="cuda")
    a = train_one(TrainConfig(seed=1, **kw), run_root=tmp_path / "a")
    b = train_one(TrainConfig(seed=1, **kw), run_root=tmp_path / "b")
    c = train_one(TrainConfig(seed=2, **kw), run_root=tmp_path / "c")
    assert a["history"] == b["history"]
    assert a["history"] != c["history"]


REAL_RUNS = sorted(Path(RUNS).glob("p2_moe_full_magnetic_real_all_observed_seed0/*/model.pt"))
needs_real_run = pytest.mark.skipif(not REAL_RUNS, reason="needs a real trained moe_full run folder on disk")
needs_dataset = pytest.mark.skipif(not (PROCESSED / "dataset_v1").exists(), reason="needs the processed dataset")


def _copy_real_config(tmp_path):
    cfg = json.loads((REAL_RUNS[-1].parent / "config.json").read_text())
    cfg["config"]["device"] = "cpu"
    (tmp_path / "config.json").write_text(json.dumps(cfg))


@needs_dataset
def test_load_run_model_rejects_a_config_it_cannot_rebuild(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"config": {"model": "no_such_model"}}))
    (tmp_path / "model.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="unknown model"):
        load_run_model(tmp_path)


def test_load_run_model_refuses_a_rewired_baseline(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"config": {"model": "chebgru", "graph": "rewired"}}))
    with pytest.raises(ValueError, match="chebgru"):
        load_run_model(tmp_path)


@needs_real_run
def test_load_run_model_refuses_a_checkpoint_from_a_different_basis(tmp_path):
    _copy_real_config(tmp_path)
    torch.save({"state_dict": {}, "fingerprint": "0123456789abcdef"}, tmp_path / "model.pt")
    with pytest.raises(RuntimeError, match="0123456789abcdef"):
        load_run_model(tmp_path)


@needs_real_run
def test_load_run_model_names_a_checkpoint_that_predates_the_fingerprint(tmp_path):
    _copy_real_config(tmp_path)
    torch.save({"state_dict": {}}, tmp_path / "model.pt")
    with pytest.raises(RuntimeError, match="predates"):
        load_run_model(tmp_path)


@needs_real_run
def test_load_run_model_round_trips_a_real_run():
    run_dir = REAL_RUNS[-1].parent
    cfg_file = json.loads((run_dir / "config.json").read_text())
    model, tensors, cfg = load_run_model(run_dir)
    assert isinstance(model, SpectralMoE)
    assert cfg.model == "moe_full"
    assert cfg_file["config"]["seed"] == cfg.seed
    pred = predict(model, tensors.stim_test[:2], tensors, dataclasses.replace(cfg, device="cpu"))
    assert np.isfinite(pred).all()
