import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from flybrain.models.wave import impulse_response, wave_solve, zoh_propagator


def reference(mu, gamma, alpha, drive, dt=1.0):
    """Integrate the same ODE with scipy under a zero-order-hold drive."""
    x, v = 0.0, 0.0
    out = []
    for f in drive:
        sol = solve_ivp(lambda t, s: [s[1], f - 2 * gamma * s[1] - alpha * mu * s[0]],
                        (0.0, dt), [x, v], rtol=1e-11, atol=1e-13)
        x, v = sol.y[0, -1], sol.y[1, -1]
        out.append(x)
    return np.array(out)


@pytest.mark.parametrize("mu,gamma,alpha,label", [
    (1.0, 0.1, 4.0, "underdamped"),     # alpha*mu - gamma^2 > 0
    (1.0, 2.0, 4.0, "critical"),        # gamma^2 == alpha*mu
    (1.0, 3.0, 1.0, "overdamped"),      # gamma^2 > alpha*mu
    (0.0, 0.5, 1.0, "zero-eigenvalue"),  # review focus 3: mu = 0
])
def test_matches_scipy_in_every_damping_regime(mu, gamma, alpha, label):
    drive = np.array([1.0, 1.0, 0.0, -0.5, 0.0, 2.0, 0.0, 0.0])
    prop = zoh_propagator(torch.tensor([mu], dtype=torch.float64),
                          torch.tensor([[gamma]], dtype=torch.float64),
                          torch.tensor([[alpha]], dtype=torch.float64))
    got = wave_solve(torch.tensor(drive, dtype=torch.float64).reshape(1, 1, 1, -1), prop)
    np.testing.assert_allclose(got[0, 0, 0].numpy(), reference(mu, gamma, alpha, drive),
                               rtol=1e-7, atol=1e-9, err_msg=label)


def test_zero_drive_gives_zero_and_stays_finite():
    prop = zoh_propagator(torch.rand(20, dtype=torch.float64) * 2,
                          torch.rand(20, 4, dtype=torch.float64) * 3,
                          torch.rand(20, 4, dtype=torch.float64) * 3)
    out = wave_solve(torch.zeros(2, 20, 4, 15, dtype=torch.float64), prop)
    assert torch.all(out == 0) and torch.isfinite(out).all()


def test_solver_is_linear_in_the_drive():
    prop = zoh_propagator(torch.tensor([0.7, 1.3], dtype=torch.float64),
                          torch.full((2, 2), 0.4, dtype=torch.float64),
                          torch.full((2, 2), 1.1, dtype=torch.float64))
    f1 = torch.randn(1, 2, 2, 9, dtype=torch.float64)
    f2 = torch.randn(1, 2, 2, 9, dtype=torch.float64)
    lhs = wave_solve(2.5 * f1 - 1.5 * f2, prop)
    rhs = 2.5 * wave_solve(f1, prop) - 1.5 * wave_solve(f2, prop)
    torch.testing.assert_close(lhs, rhs)


def test_complex_drive_is_supported():
    prop = zoh_propagator(torch.tensor([1.0], dtype=torch.float64),
                          torch.tensor([[0.3]], dtype=torch.float64),
                          torch.tensor([[2.0]], dtype=torch.float64))
    f = torch.randn(1, 1, 1, 6, dtype=torch.complex128)
    out = wave_solve(f, prop)
    assert out.dtype == torch.complex128
    torch.testing.assert_close(out.real, wave_solve(f.real.contiguous(), prop))
    torch.testing.assert_close(out.imag, wave_solve(f.imag.contiguous(), prop))


def test_gradients_flow_to_gamma_and_alpha():
    mu = torch.tensor([0.5, 1.5], dtype=torch.float64)
    drive = torch.randn(1, 2, 1, 5, dtype=torch.float64)

    def f(gamma, alpha):
        return wave_solve(drive, zoh_propagator(mu, gamma, alpha)).sum()

    gamma = torch.full((2, 1), 0.4, dtype=torch.float64, requires_grad=True)
    alpha = torch.full((2, 1), 1.2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (gamma, alpha), eps=1e-6, atol=1e-6)


def test_impulse_response_decays_and_oscillates():
    under = impulse_response(mu=1.0, gamma=0.05, alpha=16.0, n_steps=60)
    over = impulse_response(mu=1.0, gamma=3.0, alpha=1.0, n_steps=60)
    assert np.sum(np.diff(np.sign(under)) != 0) >= 2      # oscillates
    assert np.all(np.sign(over[over != 0]) == np.sign(over[np.flatnonzero(over)[0]]))  # no sign change
    assert abs(under[-1]) < abs(under[:5]).max()          # decays
