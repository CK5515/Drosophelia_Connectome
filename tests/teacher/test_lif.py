import numpy as np
import scipy.sparse as sp
import torch

from flybrain.teacher.lif import LIFNetwork, LIFParams, propagator

P = LIFParams()


def spike_steps(counts, neuron):
    return np.flatnonzero(counts[0, :, neuron].numpy())


def test_step_constants():
    assert P.refrac_steps == 22
    assert P.delay_steps == 18


def test_propagator_matches_closed_form():
    M = propagator(P)
    dt, tm, ts = P.dt_ms, P.tau_mbr_ms, P.tau_syn_ms
    u0, g0 = 3.0, 7.0
    u1 = u0 * np.exp(-dt / tm) + g0 * ts / (ts - tm) * (np.exp(-dt / ts) - np.exp(-dt / tm))
    g1 = g0 * np.exp(-dt / ts)
    np.testing.assert_allclose(M @ [u0, g0], [u1, g1], rtol=1e-12)


def test_silent_network_never_spikes():
    rng = np.random.default_rng(0)
    W = sp.csr_matrix((rng.random((20, 20)) < 0.2) * 50.0)
    net = LIFNetwork(W, P, device="cpu")
    events = np.zeros((500, 2, 3), dtype=bool)
    out = net.run(np.arange(3), 500, np.arange(20), 1, input_events=events)
    assert out.sum() == 0


def two_neuron_net():
    W = sp.csr_matrix(np.array([[0.0, 10000.0], [0.0, 0.0]]))  # 0 -> 1, 2750 mV into g per spike
    return LIFNetwork(W, P, device="cpu", dtype=torch.float64)


def test_input_kick_spikes_next_step_and_delay_is_18_steps():
    events = np.zeros((100, 1, 1), dtype=bool)
    events[10, 0, 0] = True
    out = two_neuron_net().run([0], 100, [0, 1], 1, input_events=events)
    assert spike_steps(out, 0).tolist() == [11]
    # delivery at 11 + 18 = 29 (after threshold), membrane crosses on the next update
    assert spike_steps(out, 1).tolist() == [30]


def test_reset_wipes_same_step_input_and_refractory_drops_synaptic_input():
    events = np.zeros((200, 1, 1), dtype=bool)
    events[10:, 0, 0] = True  # kick every step
    out = two_neuron_net().run([0], 200, [0, 1], 1, input_events=events)
    # the kick arriving on a spike step is wiped by reset, so the input fires every other step
    assert spike_steps(out, 0).tolist() == list(range(11, 200, 2))
    # neuron 1 gets an arrival on every odd step from 29. After spiking at 30 it is refractory for steps 31-51,
    # and Brian2 DISCARDS synaptic input to refractory neurons (verified against Brian2 2.9.0). The first
    # arrival that counts is at 53, so the next spike is at 54: ISI = 24, not 22.
    assert spike_steps(out, 1).tolist()[:3] == [30, 54, 78]
    assert np.all(np.diff(spike_steps(out, 1)) == 24)


def test_binning_preserves_spike_counts():
    rng = np.random.default_rng(1)
    W = sp.csr_matrix((rng.random((30, 30)) < 0.2) * rng.choice([-40.0, 60.0], size=(30, 30)))
    net = LIFNetwork(W, P, device="cpu")
    rates = torch.full((3, 4), 150.0)
    fine = net.run(np.arange(4), 1000, np.arange(30), 1, rates_hz=rates, seed=5)
    coarse = net.run(np.arange(4), 1000, np.arange(30), 50, rates_hz=rates, seed=5)
    np.testing.assert_array_equal(fine.reshape(3, 20, 50, 30).sum(2).numpy(), coarse.numpy())


def test_poisson_drive_gives_requested_rate_and_stops_after_stim():
    net = LIFNetwork(sp.csr_matrix((1, 1), dtype=np.float32), P, device="cpu")
    rates = torch.full((64, 1), 100.0)
    # stim ends at 9990 so a kick on the last stimulus step still fires inside bin 0 (spikes lag kicks by one step)
    out = net.run([0], 20000, [0], 10000, rates_hz=rates, stim_steps=9990, seed=0)
    on_rate = out[:, 0, 0].double().mean().item()   # spikes in the first second
    assert 90.0 < on_rate < 105.0
    assert out[:, 1, 0].sum().item() == 0             # nothing after stimulus off


def test_batch_rows_are_independent():
    # real recurrent edges; two different input trains as one batch must equal each train run alone
    rng = np.random.default_rng(3)
    W = sp.csr_matrix((rng.random((30, 30)) < 0.3) * rng.choice([-2000.0, 6000.0], size=(30, 30)))
    net = LIFNetwork(W, P, device="cpu", dtype=torch.float64)
    ev = rng.random((600, 2, 4)) < np.array([0.03, 0.12])[None, :, None]
    both = net.run(np.arange(4), 600, np.arange(30), 1, input_events=ev)
    assert both.sum() > 100 and not torch.equal(both[0], both[1])  # spiking is real and rows differ
    for b in range(2):
        alone = net.run(np.arange(4), 600, np.arange(30), 1, input_events=ev[:, b:b + 1])
        torch.testing.assert_close(both[b], alone[0], rtol=0, atol=0)
