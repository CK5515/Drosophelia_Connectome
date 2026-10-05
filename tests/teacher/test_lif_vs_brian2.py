import numpy as np
import pytest
import scipy.sparse as sp
import torch

from flybrain.teacher.lif import LIFNetwork, LIFParams


@pytest.mark.brian2
def test_matches_brian2_spike_for_spike():
    b2 = pytest.importorskip("brian2")
    from brian2 import (Network, NeuronGroup, SpikeGeneratorGroup, SpikeMonitor, Synapses, defaultclock, mV,
                        ms)
    b2.prefs.codegen.target = "numpy"

    p = LIFParams()
    n, n_in, n_steps = 50, 5, 3000
    rng = np.random.default_rng(0)
    mask = rng.random((n, n)) < 0.15
    np.fill_diagonal(mask, False)
    dense = (mask * rng.choice([-1, 1], size=(n, n), p=[0.3, 0.7]) * rng.integers(1, 30, size=(n, n)))
    W = sp.csr_matrix(dense.astype(np.float64))
    input_idx = np.arange(n_in)
    events = rng.random((n_steps, n_in)) < 0.02  # ~200 Hz

    # --- Brian2, written exactly like Shiu model.py, with PoissonInput replaced by fixed kicks ---
    defaultclock.dt = p.dt_ms * ms
    ns = dict(v_0=p.v0_mv * mV, v_rst=p.v_reset_mv * mV, v_th=p.v_th_mv * mV,
              t_mbr=p.tau_mbr_ms * ms, tau=p.tau_syn_ms * ms)
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    neu = NeuronGroup(n, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                      refractory="rfc", namespace=ns)
    neu.v = ns["v_0"]
    neu.g = 0 * mV
    neu.rfc = p.t_refrac_ms * ms
    neu.rfc[input_idx] = 0 * ms
    pre, post = W.nonzero()
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p.t_delay_ms * ms)
    syn.connect(i=pre, j=post)
    syn.w = np.asarray(W[pre, post]).ravel() * p.w_syn_mv * mV
    ev_step, ev_neuron = np.nonzero(events)
    gen = SpikeGeneratorGroup(n_in, ev_neuron, ev_step * p.dt_ms * ms)
    kick = Synapses(gen, neu, on_pre="v += kick", namespace=dict(kick=p.w_syn_mv * p.f_poisson * mV))
    kick.connect(i=np.arange(n_in), j=input_idx)
    mon = SpikeMonitor(neu)
    Network(neu, syn, gen, kick, mon).run(n_steps * p.dt_ms * ms)
    brian = sorted(zip(np.round(np.asarray(mon.t / ms) / p.dt_ms).astype(int).tolist(),
                       np.asarray(mon.i).tolist()))

    # --- our port ---
    net = LIFNetwork(W, p, device="cpu", dtype=torch.float64)
    out = net.run(input_idx, n_steps, np.arange(n), 1, input_events=events[:, None, :])
    steps, neurons = np.nonzero(out[0].numpy())
    ours = sorted(zip(steps.tolist(), neurons.tolist()))

    assert len(brian) > 100, "network too quiet to be a meaningful comparison"
    assert ours == brian
