"""Sample the GHZ-reuse model using physical quantum instruments.

Alice's remaining GHZ register is a logical qubit, so its joint state with
Bob is a 4 x 4 density matrix. The final Bell-state measurement (BSM) leaves
only Bob's 2 x 2 state. Array axes are ``(block, row, column)`` throughout.

The encoding contrast is fixed at delta = 0.2. Bob's decisions use sampled
readout outcomes; reset decisions use sampled heralds. Density matrices
and fidelities are diagnostic state, not information available to either
party. Depolarization acts on Bob once
per round. Retries reuse the same filter on the actual post-failure state.
The amplitude-updated retry comparison is an exact calculation, not part of
this sampler. No experiment here certifies a secret-key rate.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
XX = np.kron(X, X)
BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
BELL_DM = np.outer(BELL, BELL.conj())


def _integer(name, value, minimum):
    """Validate a scalar count without silently truncating a float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probability(name, value):
    """Validate a finite scalar in the closed unit interval."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not 0 <= value <= 1
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def apply(rho, operator):
    """Apply a Kraus operator to one state or a batch, without normalization."""
    return operator @ rho @ np.swapaxes(operator.conj(), -1, -2)


def trace(rho):
    """Return real traces along the last two axes."""
    return np.trace(rho, axis1=-2, axis2=-1).real


def sample(branches, rng):
    """Sample one outcome per block and return its normalized state.

    ``branches`` contains one array of unnormalized states per outcome.
    Input states must be normalized; branch traces must sum to one for each
    block. Tiny negative roundoff is clipped only after checking these
    conditions. The generator consumes exactly one uniform draw per block.
    """
    if not branches or len(branches[0]) == 0:
        raise ValueError("an instrument needs outcomes and a nonempty batch")
    probabilities = np.stack([trace(branch) for branch in branches], axis=1)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("measurement probabilities must be finite")
    if probabilities.min() < -1e-10:
        raise ValueError("negative measurement probability")
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-7, atol=1e-10):
        raise ValueError("measurement probabilities must sum to one")
    probabilities = np.maximum(probabilities, 0)
    outcomes = (
        rng.random(len(probabilities))[:, None]
        >= np.cumsum(probabilities, axis=1)[:, :-1]
    ).sum(axis=1)
    states = np.stack(branches, axis=1)[np.arange(len(outcomes)), outcomes]
    states /= probabilities[np.arange(len(outcomes)), outcomes, None, None]
    return outcomes, states


def encoding(bits):
    """Return positive amplitudes (a, b) for binary encoding at delta = 0.2."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be a one-dimensional binary array")
    return (
        np.sqrt(np.where(bits == 0, 0.6, 0.4)),
        np.sqrt(np.where(bits == 0, 0.4, 0.6)),
    )


def diagonal(values):
    """Construct a batch of diagonal operators from rows of coefficients."""
    operators = np.zeros((len(values), values.shape[1], values.shape[1]), complex)
    positions = np.arange(values.shape[1])
    operators[:, positions, positions] = values
    return operators


def intermediate_bsm(rho, bits, rng):
    """Measure the encoding ancilla and one GHZ qubit, retaining logical AB.

    The X correction acts on every remaining physical qubit; the Z
    correction acts on one Alice qubit. Corrected states do not depend on
    the uniform phase bit, but both BSM bits count as communication.
    Returns the population outcome, phase bit, and corrected state.
    """
    a, b = encoding(bits)
    d0 = diagonal(np.stack([a, a, b, b], axis=1))
    d1 = diagonal(np.stack([b, b, a, a], axis=1))
    outcomes, states = sample([apply(rho, d0), apply(rho, XX @ d1)], rng)
    phase_bits = rng.integers(0, 2, len(bits))
    return outcomes, phase_bits, states


def final_bsm(rho, bits, rng):
    """Teleport using Alice's last qubit; return a BSM outcome and Bob state.

    Four rectangular Kraus maps send AB to B. Outcomes are ordered
    ``(s, t) = (0, 0), (0, 1), (1, 0), (1, 1)``. Corrections use Z^s X^t.
    """
    a, b = encoding(bits)
    amplitudes = np.stack([a, b], axis=1)
    branches = []
    for s in (0, 1):
        for t in (0, 1):
            operator = np.zeros((len(bits), 2, 4), complex)
            for i in (0, 1):
                for j in (0, 1):
                    operator[:, j, 2 * i + j] = (
                        amplitudes[:, t ^ i]
                        * (-1) ** (s * (t ^ i))
                        / np.sqrt(2)
                    )
            correction = np.linalg.matrix_power(Z, s) @ np.linalg.matrix_power(X, t)
            branches.append(apply(rho, correction @ operator))
    return sample(branches, rng)


def depolarize(rho, p):
    """Apply Bob's replacement channel (1-p) rho + p rho_A tensor I/2."""
    p = _probability("noise", p)
    dimension = rho.shape[-1]
    operators = [np.kron(I, u) if dimension == 4 else u for u in (X, Y, Z)]
    return (1 - 3 * p / 4) * rho + p / 4 * sum(apply(rho, u) for u in operators)


def readout(rho, strength, rng):
    """Sample Bob's weak Z readout and retain its conditional state."""
    strength = _probability("strength", strength)
    k0 = np.diag(np.sqrt([(1 + strength) / 2, (1 - strength) / 2]))
    k1 = np.diag(np.sqrt([(1 - strength) / 2, (1 + strength) / 2]))
    if rho.shape[-1] == 4:
        k0, k1 = np.kron(I, k0), np.kron(I, k1)
    return sample([apply(rho, k0), apply(rho, k1)], rng)


def reset(rho, bits, mode, rng):
    """Sample Alice's reset herald; zero accepts and one denotes failure.

    Alice knows the encoding bit but not Bob's private readout. The coherent
    instrument uses F0 = diag(b, a) and, after correction/relabeling,
    F1 = diag(a, b). A diagonal mixed ancilla additionally dephases Alice.
    Both instruments preserve trace when their outcomes are summed.
    """
    if mode not in ("coherent", "mixed"):
        raise ValueError("reset mode must be 'coherent' or 'mixed'")
    a, b = encoding(bits)
    f0 = diagonal(np.stack([b, b, a, a], axis=1))
    f1 = diagonal(np.stack([a, a, b, b], axis=1))
    if mode == "mixed":
        rho = rho.copy()
        rho[:, :2, 2:] = 0
        rho[:, 2:, :2] = 0
    return sample([apply(rho, f0), apply(rho, f1)], rng)


def wilson(successes, trials):
    """Return a two-sided 95% Wilson interval, or [None, None] without trials."""
    successes = _integer("successes", successes, 0)
    trials = _integer("trials", trials, 0)
    if successes > trials:
        raise ValueError("successes must not exceed trials")
    if trials == 0:
        return [None, None]
    z = 1.959963984540054
    phat = successes / trials
    denominator = 1 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denominator
    half = (
        z
        * np.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return [float(max(0, center - half)), float(min(1, center + half))]


def experiment(
    length,
    strength,
    noise,
    attempts,
    seed,
    reset_mode="coherent",
    max_reset_attempts=1,
    test_fraction=0.0,
    link_success=1.0,
):
    """Simulate independent attempted blocks and return JSON-ready statistics.

    ``length`` is the candidate-bit count L. Each delivered data block has
    L readout rounds and L-1 reset stages; failure to reset within the limit
    discards that block. ``strength`` is g and ``noise`` is the per-round
    depolarizing parameter p. ``attempts`` and ``seed`` are integer values.
    ``reset_mode`` selects a coherent or diagonal mixed reset ancilla.
    ``max_reset_attempts`` caps fixed-filter retries in each reset stage.

    ``link_success`` and ``test_fraction`` independently select delivered
    blocks and reserve delivered blocks before data operations. Reservation
    is resource accounting only: no parameter-estimation test is simulated.
    Failed delivery, reservation, and aborted blocks still count as quantum
    transmissions and GHZ preparations. Authentication and test messages,
    time, and hardware errors beyond the specified channel are not modeled.

    The returned raw error uses Bob's outcome as the bit estimate. Its
    standard error uses independent accepted blocks, not individual bits.
    Empirical conditional quantities are None without accepted samples.
    Exact single-attempt conditional error is None when acceptance is zero;
    exact retry results are calculated separately from this sampler.
    """
    length = _integer("length", length, 1)
    attempts = _integer("attempts", attempts, 1)
    max_reset_attempts = _integer("max_reset_attempts", max_reset_attempts, 1)
    seed = _integer("seed", seed, 0)
    strength = _probability("strength", strength)
    noise = _probability("noise", noise)
    test_fraction = _probability("test_fraction", test_fraction)
    link_success = _probability("link_success", link_success)
    if reset_mode not in ("coherent", "mixed"):
        raise ValueError("reset_mode must be 'coherent' or 'mixed'")

    # Draw order and batch ordering determine reproducibility for a given seed.
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 2, (attempts, length))
    errors = np.full((attempts, length), -1, dtype=np.int8)
    delivered = rng.random(attempts) < link_success
    tests = delivered & (rng.random(attempts) < test_fraction)
    active = np.flatnonzero(delivered & ~tests)
    rho = np.broadcast_to(BELL_DM, (len(active), 4, 4)).copy()

    reached = []
    errors_reached = []
    fidelity = []
    minimum_eigenvalue = 0.0
    reset_attempts = 0
    reset_accepts = 0
    bsm_rounds = 0
    readouts = 0

    for round_index in range(length):
        count = len(active)
        reached.append(count)
        if count == 0:
            errors_reached.append(0)
            fidelity.append(None)
            continue

        bits = keys[active, round_index]
        bsm_rounds += count
        if round_index == length - 1:
            _, rho = final_bsm(rho, bits, rng)
        else:
            _, _, rho = intermediate_bsm(rho, bits, rng)
        rho = depolarize(rho, noise)
        y, rho = readout(rho, strength, rng)
        readouts += count
        errors[active, round_index] = y != bits
        errors_reached.append(int((y != bits).sum()))
        minimum_eigenvalue = min(
            minimum_eigenvalue, float(np.linalg.eigvalsh(rho).min())
        )
        if round_index == length - 1:
            fidelity.append(None)
            break

        # Retry only failures, on their updated states. Success freezes a
        # branch for the rest of this stage; it is never sampled again here.
        accepted = np.zeros(count, dtype=bool)
        for _ in range(max_reset_attempts):
            pending = np.flatnonzero(~accepted)
            if len(pending) == 0:
                break
            herald, updated = reset(rho[pending], bits[pending], reset_mode, rng)
            rho[pending] = updated
            accepted[pending[herald == 0]] = True
            reset_attempts += len(pending)
        reset_accepts += int(accepted.sum())
        active, rho = active[accepted], rho[accepted]
        fidelity.append(
            float(np.einsum("i,nij,j->n", BELL.conj(), rho, BELL).real.mean())
            if len(active)
            else None
        )

    accepted_blocks = len(active)
    accepted_errors = errors[active]
    if accepted_blocks and not np.all(accepted_errors >= 0):
        raise RuntimeError("an accepted block contains an unmeasured round")
    accepted_per_round = accepted_errors.sum(axis=0).astype(int).tolist()
    bit_errors = int(accepted_errors.sum())
    accepted_bits = length * accepted_blocks

    # One accepted block, rather than one bit, is the independent SE unit.
    block_error_fractions = (
        accepted_errors.mean(axis=1) if accepted_blocks else np.array([])
    )
    standard_error = (
        float(block_error_fractions.std(ddof=1) / np.sqrt(accepted_blocks))
        if accepted_blocks > 1
        else None
    )
    raw_rate = bit_errors / accepted_bits if accepted_bits else None
    exact_acceptance = (
        link_success * (1 - test_fraction) * 0.48 ** (length - 1)
        if max_reset_attempts == 1
        else None
    )
    exact_error = (
        0.5 - 0.1 * strength * (1 - noise) ** length / length
        if max_reset_attempts == 1 and link_success > 0 and test_fraction < 1
        else None
    )

    return {
        "length": length,
        "strength": strength,
        "noise": noise,
        "reset_mode": reset_mode,
        "max_reset_attempts": max_reset_attempts,
        "seed": seed,
        "attempted_blocks": attempts,
        "test_fraction": test_fraction,
        "link_success": link_success,
        "delivered_blocks": int(delivered.sum()),
        "reserved_test_blocks": int(tests.sum()),
        "discarded_data_blocks": int((delivered & ~tests).sum()) - accepted_blocks,
        "accepted_blocks": accepted_blocks,
        "accepted_bits": accepted_bits,
        "fully_correct_accepted_blocks": int((accepted_errors.sum(axis=1) == 0).sum()),
        "quantum_transmissions": attempts,
        "bsm_rounds": bsm_rounds,
        "readouts": readouts,
        "ghz_preparations": attempts,
        "ghz_qubits_initialized": (length + 1) * attempts,
        "encoding_ancillas": bsm_rounds,
        "reset_ancillas": reset_attempts,
        "peak_alice_qubits_serial_schedule": length + 1,
        "peak_bob_qubits": 1,
        "physical_storage_time_modeled": False,
        "reset_attempts": reset_attempts,
        "successful_reset_stages": reset_accepts,
        "classical_protocol_bits": 2 * bsm_rounds + reset_attempts,
        "test_and_authentication_bits_included": False,
        "block_acceptance": accepted_blocks / attempts,
        "block_acceptance_ci95": wilson(accepted_blocks, attempts),
        "raw_bits_per_transmission": accepted_bits / attempts,
        "accepted_raw_bit_errors": bit_errors,
        "accepted_raw_error_rate": raw_rate,
        "accepted_raw_error_standard_error_by_block": standard_error,
        "accepted_error_by_round": accepted_per_round,
        "accepted_error_ci95_by_round": [
            wilson(value, accepted_blocks) for value in accepted_per_round
        ],
        "reached_by_round": reached,
        "errors_at_readout_by_round": errors_reached,
        "mean_fidelity_after_accepted_reset": fidelity,
        "minimum_sampled_state_eigenvalue": minimum_eigenvalue,
        "secret_key_rate_certified": False,
        "exact_block_acceptance": exact_acceptance,
        "exact_accepted_raw_error_rate": exact_error,
    }
