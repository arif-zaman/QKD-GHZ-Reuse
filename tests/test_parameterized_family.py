"""Exact circuit checks for the symmetric population-encoding family.

These checks construct small physical registers, apply gates, and project
measurement outcomes without importing the simulator. Closed-form expressions
appear only as independently checked expectations.
"""
import unittest

import numpy as np


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
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


def reset(rho, bit, delta, mixed=False):
    """Physical C,A,B circuit; C controls A, whose outcome is the herald."""
    a, b = amplitudes(bit, delta)
    ancilla = np.diag([b*b, a*a]) if mixed else dm(np.array([b, a]))
    joint = conjugate(np.kron(ancilla, rho), cnot(0, 1, 3))
    branches = []
    for herald in (0, 1):
        indices = [4*c + 2*herald + bob for c in (0, 1) for bob in (0, 1)]
        branch = joint[np.ix_(indices, indices)]
        if herald:
            branch = conjugate(branch, np.kron(X, I))
        branches.append(branch)
    return branches


def accepted_branches(rho, bit, delta, strength, noise, mixed=False):
    result = []
    for after_bsm in bsm(rho, bit, delta):
        for after_readout in readout(depolarize(after_bsm, noise), strength):
            result.append(reset(after_readout, bit, delta, mixed)[0])
    return result


def exact_block(length, delta, strength, noise, mixed=False):
    """Propagate density matrices and error-weighted matrices exactly.

    Linearity marginalizes past classical histories without discarding their
    influence on future acceptance.  The jth error matrix weights precisely
    those histories in which y_j differs from x_j.
    """
    rho = dm(BELL)
    error_states = []
    for _ in range(length - 1):
        accumulated = np.zeros((4, 4), dtype=complex)
        next_errors = [np.zeros((4, 4), dtype=complex)
                       for _ in range(len(error_states) + 1)]
        for bit in (0, 1):
            branches = accepted_branches(rho, bit, delta, strength, noise, mixed)
            for index, branch in enumerate(branches):
                accumulated += branch/2
                if index % 2 != bit:
                    next_errors[-1] += branch/2
            for j, error_state in enumerate(error_states):
                next_errors[j] += sum(accepted_branches(
                    error_state, bit, delta, strength, noise, mixed))/2
        rho, error_states = accumulated, next_errors

    probability = 0.
    errors = np.zeros(length)
    for bit in (0, 1):
        for after_bsm in bsm(rho, bit, delta, final=True):
            for y, branch in enumerate(readout(depolarize(after_bsm, noise), strength)):
                weight = np.trace(branch).real/2
                probability += weight
                if y != bit:
                    errors[-1] += weight
        for j, error_state in enumerate(error_states):
            for after_bsm in bsm(error_state, bit, delta, final=True):
                errors[j] += sum(np.trace(branch).real for branch in
                                 readout(depolarize(after_bsm, noise), strength))/2
    return probability, None if probability == 0. else errors/probability


class ParameterizedCircuitTests(unittest.TestCase):
    def assert_physical_instrument(self, branches, incoming):
        for branch in branches:
            np.testing.assert_allclose(branch, branch.conj().T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(branch).min(), -1e-12)
        self.assertAlmostEqual(sum(np.trace(branch).real for branch in branches),
                               np.trace(incoming).real, places=12)

    def test_first_bsm_branches_and_helstrom_discrimination(self):
        for delta in (0., .2, .55, .9, 1.):
            local, final = [], []
            for bit in (0, 1):
                branches = bsm(dm(BELL), bit, delta)
                self.assert_physical_instrument(branches, dm(BELL))
                a, b = amplitudes(bit, delta)
                expected = dm(np.array([a, 0, 0, b]))/4
                np.testing.assert_allclose(branches, [expected]*4, atol=1e-12)
                local.append(np.einsum('abad->bd', sum(branches).reshape(2, 2, 2, 2)))
                final.append(sum(bsm(dm(BELL), bit, delta, final=True)))
                np.testing.assert_allclose(final[-1], dm(amplitudes(bit, delta)), atol=1e-12)
            for pair in (local, final):
                trace_norm = np.abs(np.linalg.eigvalsh(pair[0]-pair[1])).sum()
                self.assertAlmostEqual(.5 + trace_norm/4, (1+delta)/2, places=12)
            self.assertAlmostEqual(abs(np.vdot(amplitudes(0, delta),
                                              amplitudes(1, delta))),
                                   np.sqrt(1-delta**2), places=12)

    def test_weak_measurement_accuracy_before_reset(self):
        for delta in (0., .2, .8, 1.):
            for strength in (0., .45, 1.):
                correct = 0.
                for bit in (0, 1):
                    rho = sum(bsm(dm(BELL), bit, delta))
                    correct += np.trace(readout(rho, strength)[bit]).real/2
                self.assertAlmostEqual(correct, .5+delta*strength/2, places=12)

    def test_coherent_and_mixed_reset_have_equal_rates_different_coherence(self):
        for delta in (0., .2, .8, 1.):
            r = (1-delta**2)/2
            for bit in (0, 1):
                rho = sum(bsm(dm(BELL), bit, delta))
                for mixed in (False, True):
                    branches = reset(rho, bit, delta, mixed)
                    self.assert_physical_instrument(branches, rho)
                    expected = (np.diag([.5, 0, 0, .5]) if mixed else dm(BELL))*r
                    np.testing.assert_allclose(branches[0], expected, atol=1e-12)
                    self.assertAlmostEqual(np.trace(branches[0]).real, r, places=12)

    def test_accepted_record_and_state_erase_bit_for_arbitrary_inputs(self):
        rng = np.random.default_rng(20261001)
        matrix = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
        arbitrary = matrix @ matrix.conj().T
        arbitrary /= np.trace(arbitrary)
        for rho in (dm(BELL), arbitrary, np.diag([1., 0, 0, 0])):
            for delta in (0., .2, .7, 1.):
                for strength, noise in ((0., 0.), (.6, .2), (1., 1.)):
                    for mixed in (False, True):
                        branches0 = accepted_branches(rho, 0, delta, strength, noise, mixed)
                        branches1 = accepted_branches(rho, 1, delta, strength, noise, mixed)
                        np.testing.assert_allclose(branches0, branches1, atol=1e-12)
                        self.assertAlmostEqual(sum(np.trace(branch).real for branch in branches0),
                                               (1-delta**2)/2, places=12)

    def test_first_accepted_joint_readout_probabilities_and_fidelity(self):
        for delta in (0., .2, .8):
            for strength in (0., .6, 1.):
                for bit in (0, 1):
                    branches = accepted_branches(dm(BELL), bit, delta, strength, 0.)
                    for y in (0, 1):
                        joint = sum(branches[y::2])
                        probability = np.trace(joint).real
                        self.assertAlmostEqual(probability, (1-delta**2)/4, places=12)
                        fidelity = np.vdot(BELL, joint @ BELL).real/probability
                        self.assertAlmostEqual(fidelity, (1+np.sqrt(1-strength**2))/2, places=12)

    def test_sequential_accepted_state_and_fidelity_are_delta_independent(self):
        for delta in (0., .2, .8, .99):
            for strength, noise in ((0., 0.), (.4, .0), (.6, .13), (1., 0.), (.3, 1.)):
                rho = dm(BELL)
                for k in range(1, 5):
                    unnormalized = sum(accepted_branches(rho, k % 2, delta, strength, noise))
                    probability = np.trace(unnormalized).real
                    self.assertAlmostEqual(probability, (1-delta**2)/2, places=12)
                    rho = unnormalized/probability
                    longitudinal = (1-noise)**k
                    transverse = ((1-noise)*np.sqrt(1-strength**2))**k
                    expected = (np.eye(4) + transverse*(np.kron(X, X)-np.kron(Y, Y))
                                + longitudinal*np.kron(Z, Z))/4
                    np.testing.assert_allclose(rho, expected, atol=1e-12)
                    self.assertAlmostEqual(np.vdot(BELL, rho @ BELL).real,
                                           (1+longitudinal+2*transverse)/4, places=12)

    def test_exact_sequential_acceptance_and_each_bit_error(self):
        for delta in (0., .2, .75):
            for strength, noise in ((0., 0.), (.5, 0.), (1., 0.), (.7, .17), (.4, 1.)):
                for length in (1, 2, 4):
                    probability, errors = exact_block(length, delta, strength, noise)
                    self.assertAlmostEqual(probability, ((1-delta**2)/2)**(length-1), places=12)
                    expected = [.5]*(length-1) + [.5-delta*strength*(1-noise)**length/2]
                    np.testing.assert_allclose(errors, expected, atol=1e-12)
                    self.assertAlmostEqual(np.mean(errors),
                                           .5-delta*strength*(1-noise)**length/(2*length), places=12)

    def test_optimal_balancing_filter_success_including_boundary_states(self):
        for delta in (0., .2, .7, 1.):
            for strength in (0., .2, .5, .7, 1.):
                for bit in (0, 1):
                    rho = sum(bsm(dm(BELL), bit, delta))
                    success = 0.
                    for conditional in readout(rho, strength):
                        # Construct a contractive Alice filter from the actual
                        # conditional Schmidt coefficients, not from R(delta,g).
                        populations = np.diag(conditional).real[[0, 3]]
                        target = populations.min()
                        diagonal = np.sqrt(np.divide(target, populations,
                                                      out=np.zeros(2), where=populations > 0))
                        filtered = conjugate(conditional, np.kron(np.diag(diagonal), I))
                        np.testing.assert_allclose(filtered, 2*target*dm(BELL), atol=1e-12)
                        self.assertTrue(np.all(diagonal <= 1+1e-12))
                        success += np.trace(filtered).real
                    self.assertAlmostEqual(success, 1-max(delta, strength), places=12)

    def test_public_reset_flag_can_leak_without_transmitting_ancilla(self):
        product = np.diag([1., 0, 0, 0])
        for delta in (0., .2, .8, 1.):
            distributions = []
            for bit in (0, 1):
                branches = reset(product, bit, delta)
                self.assert_physical_instrument(branches, product)
                distributions.append([np.trace(branch).real for branch in branches])
            best_guess = .5*np.max(distributions, axis=0).sum()
            self.assertAlmostEqual(best_guess, (1+delta)/2, places=12)

    def test_delta_one_no_multi_round_acceptance_but_valid_final_teleportation(self):
        for strength, noise in ((0., 0.), (.6, .2), (1., 0.), (1., 1.)):
            for length in (2, 3):
                probability, errors = exact_block(length, 1., strength, noise)
                self.assertEqual(probability, 0.)
                self.assertIsNone(errors)
            probability, errors = exact_block(1, 1., strength, noise)
            self.assertAlmostEqual(probability, 1., places=12)
            self.assertAlmostEqual(errors[0], .5-strength*(1-noise)/2, places=12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
