"""Check retry formulas against independently constructed physical circuits."""
import unittest

import numpy as np

from qkd_reuse.analysis import amplitude_updated_predictions, fixed_filter_predictions


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def dm(vector):
    return np.outer(vector, vector.conj())


def conjugate(rho, operator):
    return operator @ rho @ operator.conj().T


def gate(operator, target, qubits):
    result = np.array([[1]], dtype=complex)
    for position in range(qubits):
        result = np.kron(result, operator if position == target else I)
    return result


def cnot(control, target, qubits):
    result = np.zeros((2**qubits, 2**qubits), dtype=complex)
    for column in range(2**qubits):
        row = column
        if (column >> (qubits - 1 - control)) & 1:
            row ^= 1 << (qubits - 1 - target)
        result[row, column] = 1
    return result


def amplitudes(bit, delta):
    return np.sqrt([(1 + (-1)**bit * delta)/2,
                    (1 - (-1)**bit * delta)/2])


def bsm(rho, bit, delta, final=False):
    """Explicit S,A1,A2,B (or final S,A,B) gates and projections."""
    if final:
        qubits = 3
        joint = np.kron(dm(amplitudes(bit, delta)), rho)
    else:
        qubits = 4
        # Physical repetition encoding |a,b> -> |a,a,b>.
        repetition = np.zeros((8, 4), dtype=complex)
        for alice in (0, 1):
            for bob in (0, 1):
                repetition[6*alice + bob, 2*alice + bob] = 1
        expanded = repetition @ rho @ repetition.conj().T
        joint = np.kron(dm(amplitudes(bit, delta)), expanded)
    joint = conjugate(joint, gate(H, 0, qubits) @ cnot(0, 1, qubits))
    kept_dimension = 2**(qubits - 2)
    branches = []
    for s in (0, 1):
        for t in (0, 1):
            indices = [kept_dimension*(2*s + t) + index
                       for index in range(kept_dimension)]
            branch = joint[np.ix_(indices, indices)]
            correction = np.eye(kept_dimension, dtype=complex)
            if t:
                correction = (X if final else np.kron(X, X)) @ correction
            if s:
                correction = (Z if final else np.kron(Z, I)) @ correction
            branches.append(conjugate(branch, correction))
    return branches


def depolarize(rho, noise):
    if rho.shape == (2, 2):
        replacement = np.trace(rho)*I/2
    else:
        alice = np.einsum('abcb->ac', rho.reshape(2, 2, 2, 2))
        replacement = np.kron(alice, I/2)
    return (1-noise)*rho + noise*replacement


def readout(rho, strength):
    branches = []
    for y in (0, 1):
        kraus = np.diag(np.sqrt([(1 + (-1)**y*strength)/2,
                                 (1 - (-1)**y*strength)/2]))
        branches.append(conjugate(rho, kraus if rho.shape == (2, 2)
                                  else np.kron(I, kraus)))
    return branches


def trace(state):
    return float(np.trace(state).real)


def reset_circuit(state, coefficients):
    """Prepare C=(b,a), apply CNOT C->A, project A, retain C as new A."""
    a, b = coefficients
    joint = conjugate(np.kron(dm(np.array([b, a])), state), cnot(0, 1, 3))
    branches = []
    for herald in (0, 1):
        indices = [4*c + 2*herald + bob for c in (0, 1) for bob in (0, 1)]
        branch = joint[np.ix_(indices, indices)]
        if herald:
            branch = conjugate(branch, np.kron(X, I))
        branches.append(branch)
    return branches


def stage_for_bit(state, bit, delta, strength, noise, attempts, updated):
    """All unnormalized (BSM result, readout, success index) branches."""
    sectors, attempt_cost, failed_mass = {}, 0., 0.
    for bsm_index, after_bsm in enumerate(bsm(state, bit, delta)):
        for y, after_readout in enumerate(readout(depolarize(after_bsm, noise), strength)):
            remaining = after_readout
            coefficients = amplitudes(bit, delta)
            for m in range(attempts):
                attempt_cost += trace(remaining)
                accepted, remaining = reset_circuit(remaining, coefficients)
                sectors[bsm_index, y, m] = accepted
                if updated:
                    coefficients = coefficients**2
                    coefficients /= np.linalg.norm(coefficients)
            failed_mass += trace(remaining)
    return sectors, attempt_cost, failed_mass


def averaged_stage(state, delta, strength, noise, attempts, updated):
    accepted = np.zeros((4, 4), complex)
    current_error = np.zeros((4, 4), complex)
    cost = 0.
    for bit in (0, 1):
        sectors, bit_cost, _ = stage_for_bit(
            state, bit, delta, strength, noise, attempts, updated)
        cost += bit_cost / 2
        for (_, y, _), branch in sectors.items():
            accepted += branch / 2
            if y != bit:
                current_error += branch / 2
    return accepted, current_error, cost


def final_round(state, delta, strength, noise):
    retained, errors = 0., 0.
    for bit in (0, 1):
        for after_bsm in bsm(state, bit, delta, final=True):
            for y, branch in enumerate(readout(depolarize(after_bsm, noise), strength)):
                weight = trace(branch) / 2
                retained += weight
                if y != bit:
                    errors += weight
    return retained, errors


def circuit_block(length, delta, strength, noise, attempts, updated):
    """Exact marginalization over all branches, including discarded blocks."""
    state = dm(BELL)
    error_states = []
    reset_cost, bsm_cost = 0., 0.
    for _ in range(length - 1):
        bsm_cost += trace(state)
        accepted, current_error, cost = averaged_stage(
            state, delta, strength, noise, attempts, updated)
        reset_cost += cost
        prior_errors = [averaged_stage(err, delta, strength, noise, attempts, updated)[0]
                        for err in error_states]
        state, error_states = accepted, prior_errors + [current_error]
    bsm_cost += trace(state)
    accepted_mass, last_error = final_round(state, delta, strength, noise)
    previous_errors = sum(final_round(err, delta, strength, noise)[0] for err in error_states)
    return {
        "block_acceptance": accepted_mass,
        "raw_error": None if accepted_mass == 0 else (previous_errors + last_error) / (length * accepted_mass),
        "mean_reset_attempts": reset_cost,
        "mean_bsm_rounds": bsm_cost,
        "mean_public_bits": 2*bsm_cost + reset_cost,
    }


class RetryAnalysisTests(unittest.TestCase):
    def test_exact_predictions_against_complete_physical_branch_propagation(self):
        cases = [
            (1, .2, .5, .0, 4), (1, 1., 1., .0, 3),
            (2, .2, .5, .0, 1), (2, .2, .5, .07, 4),
            (3, .67, .81, .19, 2), (4, .2, .5, .005, 4),
            (3, .0, .5, .0, 3), (2, 1., .4, .2, 4),
            (3, .8, .0, .03, 2), (3, .4, 1., .0, 3),
            (2, .7, .8, 1., 2),
        ]
        for updated, prediction in ((False, fixed_filter_predictions),
                                    (True, amplitude_updated_predictions)):
            for length, delta, strength, noise, attempts in cases:
                with self.subTest(updated=updated, case=(length, delta, strength, noise, attempts)):
                    actual = circuit_block(length, delta, strength, noise, attempts, updated)
                    expected = prediction(length, delta, strength, noise, attempts)
                    for name, value in actual.items():
                        if value is None:
                            self.assertIsNone(expected[name])
                        else:
                            self.assertAlmostEqual(value, expected[name], places=11, msg=name)

    def test_complete_stage_instrument_on_arbitrary_complex_mixed_inputs(self):
        rng = np.random.default_rng(20261023)
        for _ in range(3):
            matrix = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
            state = matrix @ matrix.conj().T
            state /= trace(state)
            for updated, prediction in ((False, fixed_filter_predictions),
                                        (True, amplitude_updated_predictions)):
                expected = prediction(2, .37, .63, .07, 3)
                for bit in (0, 1):
                    sectors, cost, failed = stage_for_bit(state, bit, .37, .63, .07, 3, updated)
                    accepted = sum(trace(branch) for branch in sectors.values())
                    self.assertAlmostEqual(accepted + failed, 1., places=12)
                    self.assertAlmostEqual(accepted, expected["stage_acceptance"], places=12)
                    self.assertAlmostEqual(cost, expected["mean_reset_attempts"], places=12)
                    for branch in sectors.values():
                        self.assertGreaterEqual(np.linalg.eigvalsh(branch).min(), -1e-12)

    def test_public_success_index_sectors_and_private_history_conditioning(self):
        rng = np.random.default_rng(139)
        matrix = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
        state = matrix @ matrix.conj().T
        state /= trace(state)
        first, _, _ = stage_for_bit(state, 0, .6, .4, .2, 3, True)
        # A specific prior public BSM result, private y, and success index.
        history_state = first[1, 1, 2]
        history_state /= trace(history_state)
        for incoming in (state, history_state):
            updated = [stage_for_bit(incoming, bit, .6, .4, .2, 4, True)[0]
                       for bit in (0, 1)]
            fixed = [stage_for_bit(incoming, bit, .6, .4, .2, 4, False)[0]
                     for bit in (0, 1)]
            for key in updated[0]:
                # Equality includes unnormalized probability, not just state.
                np.testing.assert_allclose(updated[0][key], updated[1][key], atol=1e-13)
                if key[2] == 0:
                    np.testing.assert_allclose(fixed[0][key], fixed[1][key], atol=1e-13)
            self.assertGreater(max(np.max(np.abs(fixed[0][key]-fixed[1][key]))
                                   for key in fixed[0] if key[2] > 0), 1e-5)

    def test_conditioning_requires_joint_weights_not_forced_reset_success(self):
        # Keeping the ordinary pre-reset y weights and forcing every reset
        # to succeed incorrectly preserves the original 55% accuracy.
        joint_correct, joint_accept, forced_correct = 0., 0., 0.
        for bit in (0, 1):
            for after_bsm in bsm(dm(BELL), bit, .2):
                for y, before_reset in enumerate(readout(after_bsm, .5)):
                    accepted = reset_circuit(before_reset, amplitudes(bit, .2))[0]
                    joint_accept += trace(accepted) / 2
                    if y == bit:
                        joint_correct += trace(accepted) / 2
                        forced_correct += trace(before_reset) / 2
        self.assertAlmostEqual(joint_correct / joint_accept, .5, places=12)
        self.assertAlmostEqual(forced_correct, .55, places=12)

    def test_degenerate_boundaries_and_large_retry_limits(self):
        for prediction in (fixed_filter_predictions, amplitude_updated_predictions):
            zero = prediction(3, 1., .7, max_attempts=1000)
            self.assertEqual(zero["block_acceptance"], 0.)
            self.assertIsNone(zero["raw_error"])
            self.assertEqual(zero["mean_reset_attempts"], 1000.)
            self.assertEqual(zero["mean_bsm_rounds"], 1.)
            single = prediction(1, 1., 1., max_attempts=1000)
            self.assertEqual(single["block_acceptance"], 1.)
            self.assertEqual(single["raw_error"], 0.)
            self.assertEqual(single["mean_reset_attempts"], 0.)
            self.assertAlmostEqual(prediction(4, 0., .9, max_attempts=4)["raw_error"], .5)
            self.assertAlmostEqual(prediction(4, .2, .9, noise=1., max_attempts=4)["raw_error"], .5)
        self.assertAlmostEqual(amplitude_updated_predictions(2, .2, .5, max_attempts=1000)
                               ["stage_acceptance"], .8, places=14)
        self.assertEqual(amplitude_updated_predictions(2, 0., .5, max_attempts=4)
                         ["stage_acceptance"], .9375)
        # A floating-point underflow is not a zero-probability premise.
        underflow = fixed_filter_predictions(10000, .9, .5)
        self.assertEqual(underflow["block_acceptance"], 0.)
        self.assertIsNotNone(underflow["raw_error"])

    def test_invalid_parameters_are_rejected(self):
        invalid = [{"length": 0}, {"length": 2.5}, {"length": True},
                   {"delta": -0.1}, {"delta": float("nan")},
                   {"strength": 1.1}, {"noise": float("inf")},
                   {"max_attempts": 0}, {"max_attempts": 1.5}]
        for prediction in (fixed_filter_predictions, amplitude_updated_predictions):
            for replacement in invalid:
                arguments = dict(length=2, delta=.2, strength=.5)
                arguments.update(replacement)
                with self.subTest(function=prediction.__name__, arguments=arguments):
                    with self.assertRaises(ValueError):
                        prediction(**arguments)


if __name__ == "__main__":
    unittest.main(verbosity=2)
