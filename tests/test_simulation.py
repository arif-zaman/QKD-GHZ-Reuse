"""Check the logical model against explicitly constructed physical circuits.

These tests construct gates and measurement branches independently using
NumPy and the standard-library unittest runner.
"""
import unittest
from unittest import mock

import numpy as np

from qkd_reuse import simulation as sim


IDENTITY = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.diag([1, -1]).astype(complex)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def outer(vector):
    return np.outer(vector, vector.conj())


def full_gate(operator, target, qubits):
    factors = [operator if position == target else IDENTITY
               for position in range(qubits)]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def full_cnot(control, target, qubits):
    result = np.zeros((2**qubits, 2**qubits), complex)
    for column in range(2**qubits):
        row = column
        if (column >> (qubits - 1 - control)) & 1:
            row ^= 1 << (qubits - 1 - target)
        result[row, column] = 1
    return result


def conjugate(rho, operator):
    return operator @ rho @ operator.conj().T


def encoded_vector(bit):
    return np.sqrt([0.6, 0.4] if bit == 0 else [0.4, 0.6])


def random_density(rng, dimension):
    matrix = (
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
    )
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


def independent_intermediate(rho, bit):
    # Encode logical Alice A as the physical repetition register A1,A2.
    # Order before BSM: S,A1,A2,B. This retains all Bob error sectors.
    isometry = np.zeros((8, 4), complex)
    for a in (0, 1):
        for b in (0, 1):
            isometry[4*a + 2*a + b, 2*a+b] = 1
    expanded = isometry @ rho @ isometry.conj().T
    joint = np.kron(outer(encoded_vector(bit)), expanded)
    circuit = full_gate(HADAMARD, 0, 4) @ full_cnot(0, 1, 4)
    joint = conjugate(joint, circuit)
    result = []
    for s in (0, 1):
        for m in (0, 1):
            indices = [8*s + 4*m + rem for rem in range(4)]
            branch = joint[np.ix_(indices, indices)]
            correction = np.eye(4, dtype=complex)
            if m:
                correction = np.kron(PAULI_X, PAULI_X) @ correction
            if s:
                correction = np.kron(PAULI_Z, IDENTITY) @ correction
            result.append(conjugate(branch, correction))
    return result


def independent_final(rho, bit):
    # Order S,A,B; explicitly operate and project both measured qubits.
    joint = np.kron(outer(encoded_vector(bit)), rho)
    circuit = full_gate(HADAMARD, 0, 3) @ full_cnot(0, 1, 3)
    joint = conjugate(joint, circuit)
    result = []
    for s in (0, 1):
        for m in (0, 1):
            indices = [4*s + 2*m + b for b in (0, 1)]
            branch = joint[np.ix_(indices, indices)]
            correction = np.eye(2, dtype=complex)
            if m:
                correction = PAULI_X @ correction
            if s:
                correction = PAULI_Z @ correction
            result.append(conjugate(branch, correction))
    return result


def independent_reset(rho, bit, mode):
    a, b = encoded_vector(bit)
    ancilla = outer(np.array([b, a])) if mode == 'coherent' else np.diag([b*b, a*a])
    joint = conjugate(np.kron(ancilla, rho), full_cnot(0, 1, 3))
    result = []
    for herald in (0, 1):
        indices = [4*c + 2*herald + bob for c in (0, 1) for bob in (0, 1)]
        branch = joint[np.ix_(indices, indices)]
        if herald:
            branch = conjugate(branch, np.kron(PAULI_X, IDENTITY))
        result.append(branch)
    return result


def instrument_branches(function, *args):
    """Expose pre-sampling branches, without duplicating implementation maps."""
    captured = []

    def capture(branches, rng):
        captured.extend(branch.copy() for branch in branches)
        return np.zeros(len(branches[0]), dtype=int), branches[0].copy()

    with mock.patch.object(sim, 'sample', side_effect=capture):
        function(*args, np.random.default_rng(80))
    return captured


def single_branches(function, rho, *args):
    return [branch[0] for branch in instrument_branches(function, rho[None], *args)]


class _CircuitFixtures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11983)
        self.states = [random_density(rng, 4) for _ in range(5)]
        self.states += [sim.BELL_DM, outer(np.array([0, 1, 0, 0], complex))]

    def assert_instrument(self, branches, input_state):
        total = 0.
        for branch in branches:
            np.testing.assert_allclose(branch, branch.conj().T, atol=1e-12)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(branch).min()), -1e-12)
            total += np.trace(branch).real
        self.assertAlmostEqual(total, np.trace(input_state).real, places=12)


class CircuitEquivalenceTests(_CircuitFixtures):
    def test_intermediate_bsm_matches_physical_four_qubit_circuit(self):
        for bit in (0, 1):
            for rho in self.states:
                actual = single_branches(sim.intermediate_bsm, rho, np.array([bit]))
                full = independent_intermediate(rho, bit)
                # Phase outcomes are individually equiprobable after correction.
                np.testing.assert_allclose(full[0], full[2], atol=1e-12)
                np.testing.assert_allclose(full[1], full[3], atol=1e-12)
                np.testing.assert_allclose(actual[0], full[0]+full[2], atol=1e-12)
                np.testing.assert_allclose(actual[1], full[1]+full[3], atol=1e-12)
                self.assert_instrument(actual, rho)

    def test_final_bsm_matches_physical_three_qubit_circuit(self):
        for bit in (0, 1):
            for rho in self.states:
                actual = single_branches(sim.final_bsm, rho, np.array([bit]))
                expected = independent_final(rho, bit)
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                self.assert_instrument(actual, rho)

    def test_reset_modes_and_repeated_failure_match_ancilla_circuits(self):
        for bit in (0, 1):
            for mode in ('coherent', 'mixed'):
                for rho in self.states:
                    for retry in range(3):
                        actual = single_branches(sim.reset, rho, np.array([bit]), mode)
                        expected = independent_reset(rho, bit, mode)
                        np.testing.assert_allclose(actual, expected, atol=1e-12)
                        self.assert_instrument(actual, rho)
                        # Retrying the SAME ancilla on the actual failed state.
                        rho = expected[1]

    def test_depolarization_is_bob_only_replacement_channel(self):
        for rho in self.states:
            alice = np.einsum('abcb->ac', rho.reshape(2, 2, 2, 2))
            for noise in (0., .005, .3, 1.):
                expected = (1-noise)*rho + noise*np.kron(alice, IDENTITY/2)
                actual = sim.depolarize(rho[None], noise)[0]
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                self.assert_instrument([actual], rho)
        qubit = random_density(np.random.default_rng(17), 2)
        np.testing.assert_allclose(sim.depolarize(qubit[None], .3)[0], .7*qubit+.3*IDENTITY/2)

    def test_all_readout_strengths_give_physical_complete_instruments(self):
        for rho in self.states:
            for strength in (0., .2, .65, 1.):
                branches = single_branches(sim.readout, rho, strength)
                self.assert_instrument(branches, rho)
                if strength == 1:
                    for branch in branches:
                        # Strong Z readout kills Bob computational-basis coherence.
                        tensor = branch.reshape(2, 2, 2, 2)
                        np.testing.assert_allclose(tensor[:, 0, :, 1], 0, atol=1e-12)
                        np.testing.assert_allclose(tensor[:, 1, :, 0], 0, atol=1e-12)


class ExactProtocolTests(_CircuitFixtures):
    @staticmethod
    def accepted_stage(rho, bit, strength, noise):
        branches = []
        for after_bsm in single_branches(sim.intermediate_bsm, rho, np.array([bit])):
            noisy = sim.depolarize(after_bsm[None], noise)[0]
            for after_readout in single_branches(sim.readout, noisy, strength):
                branches.append(
                    single_branches(
                        sim.reset, after_readout, np.array([bit]), 'coherent'
                    )[0]
                )
        return branches

    def test_two_round_enumeration_erases_accepted_first_bit(self):
        for mode in ('coherent', 'mixed'):
            for strength in (0., .3, 1.):
                accepted_probability = 0.
                errors = np.zeros(2)
                fidelity_weight = 0.
                first_bit_transcripts = [dict(), dict()]
                for x0 in (0, 1):
                    first = single_branches(sim.intermediate_bsm, sim.BELL_DM, np.array([x0]))
                    for m, after_bsm in enumerate(first):
                        readouts = single_branches(sim.readout, after_bsm, strength)
                        self.assert_instrument(readouts, after_bsm)
                        for y0, after_readout in enumerate(readouts):
                            resets = single_branches(
                                sim.reset, after_readout, np.array([x0]), mode
                            )
                            self.assert_instrument(resets, after_readout)
                            kept = resets[0]
                            fidelity_weight += .5*np.vdot(sim.BELL, kept @ sim.BELL).real
                            for x1 in (0, 1):
                                finals = single_branches(sim.final_bsm, kept, np.array([x1]))
                                self.assert_instrument(finals, kept)
                                for final_bsm, last in enumerate(finals):
                                    final_readouts = single_branches(
                                        sim.readout, last, strength
                                    )
                                    for y1, last_readout in enumerate(final_readouts):
                                        probability = np.trace(last_readout).real/4
                                        accepted_probability += probability
                                        errors += probability*np.array([y0 != x0, y1 != x1])
                                        record = (m, y0, x1, final_bsm, y1)
                                        first_bit_transcripts[x0][record] = probability
                self.assertAlmostEqual(accepted_probability, .48, places=12)
                np.testing.assert_allclose(errors/.48, [.5, .5-.1*strength], atol=1e-12)
                expected_fidelity = (1+np.sqrt(1-strength**2))/2 if mode == 'coherent' else .5
                self.assertAlmostEqual(fidelity_weight/.48, expected_fidelity, places=12)
                self.assertEqual(
                    first_bit_transcripts[0].keys(), first_bit_transcripts[1].keys()
                )
                for record in first_bit_transcripts[0]:
                    self.assertAlmostEqual(
                        first_bit_transcripts[0][record],
                        first_bit_transcripts[1][record],
                        places=12,
                    )

    def test_reset_acceptance_is_point_48_for_noisy_arbitrary_inputs(self):
        for rho in self.states:
            for mode in ('coherent', 'mixed'):
                for bit in (0, 1):
                    total = 0.
                    bsm_branches = single_branches(
                        sim.intermediate_bsm, rho, np.array([bit])
                    )
                    for after_bsm in bsm_branches:
                        noisy = sim.depolarize(after_bsm[None], .2)[0]
                        for after_readout in single_branches(sim.readout, noisy, .7):
                            kept = single_branches(
                                sim.reset, after_readout, np.array([bit]), mode
                            )[0]
                            total += np.trace(kept).real
                    self.assertAlmostEqual(total, .48, places=12)

    def test_noisy_accepted_map_and_bit_erasure_for_arbitrary_density_matrices(self):
        xx = np.kron(PAULI_X, PAULI_X)
        for rho in self.states:
            for noise, strength in ((.005, .5), (.3, .7), (1., 1.)):
                first = self.accepted_stage(rho, 0, strength, noise)
                second = self.accepted_stage(rho, 1, strength, noise)
                np.testing.assert_allclose(first, second, atol=1e-12)
                self.assertAlmostEqual(sum(np.trace(v).real for v in first), .48, places=12)
                # Independent trace-replacement channel then dephase Bob coherences.
                twirled = (rho + xx @ rho @ xx)/2
                alice = np.einsum('abcb->ac', twirled.reshape(2, 2, 2, 2))
                expected = (1-noise)*twirled + noise*np.kron(alice, IDENTITY/2)
                tensor = expected.reshape(2, 2, 2, 2)
                tensor[:, 0, :, 1] *= np.sqrt(1-strength**2)
                tensor[:, 1, :, 0] *= np.sqrt(1-strength**2)
                np.testing.assert_allclose(sum(first)/.48, expected, atol=1e-12)

    def test_noisy_accepted_fidelity_for_five_sequential_resets(self):
        yy = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]]))
        xx, zz = np.kron(PAULI_X, PAULI_X), np.kron(PAULI_Z, PAULI_Z)
        for noise in (0., .03, .6, 1.):
            for strength in (0., .4, 1.):
                rho = sim.BELL_DM.copy()
                for k in range(1, 6):
                    branches = self.accepted_stage(rho, k % 2, strength, noise)
                    self.assertAlmostEqual(
                        sum(np.trace(v).real for v in branches), .48, places=12
                    )
                    rho = sum(branches)/.48
                    longitudinal = (1-noise)**k
                    transverse = ((1-noise)*np.sqrt(1-strength**2))**k
                    expected = (np.eye(4)+transverse*(xx-yy)+longitudinal*zz)/4
                    np.testing.assert_allclose(rho, expected, atol=1e-12)
                    self.assertAlmostEqual(np.vdot(sim.BELL, rho @ sim.BELL).real,
                                           (1+longitudinal+2*transverse)/4, places=12)

    def test_noisy_exact_two_round_error_formula(self):
        for noise in (.005, .2, 1.):
            for strength in (.4, 1.):
                probability = 0.
                errors = np.zeros(2)
                for x0 in (0, 1):
                    branches = self.accepted_stage(sim.BELL_DM, x0, strength, noise)
                    for record, kept in enumerate(branches):
                        y0 = record % 2
                        for x1 in (0, 1):
                            for final in single_branches(sim.final_bsm, kept, np.array([x1])):
                                noisy = sim.depolarize(final[None], noise)[0]
                                readouts = single_branches(sim.readout, noisy, strength)
                                for y1, measured in enumerate(readouts):
                                    weight = np.trace(measured).real/4
                                    probability += weight
                                    errors += weight*np.array([y0 != x0, y1 != x1])
                self.assertAlmostEqual(probability, .48, places=12)
                np.testing.assert_allclose(errors/.48,
                                           [.5, .5-.1*strength*(1-noise)**2], atol=1e-12)


class EndToEndTests(unittest.TestCase):
    def test_recorded_draw_order_has_stable_seeded_counts(self):
        # This fixture checks the recorded sampler's draw order at NumPy
        # 2.3.5, including failed-reset branches and all preceding readouts.
        result = sim.experiment(
            5, .5, .005, 4000, 20261001, max_reset_attempts=4
        )
        self.assertEqual(result['accepted_blocks'], 2779)
        self.assertEqual(result['accepted_error_by_round'],
                         [1292, 1261, 1303, 1316, 1278])
        self.assertEqual(result['reset_attempts'], 27534)
        self.assertEqual(result['reached_by_round'],
                         [4000, 3666, 3367, 3066, 2779])
        self.assertAlmostEqual(result['accepted_raw_error_rate'],
                               0.4641957538682979, places=12)
        self.assertAlmostEqual(result['accepted_raw_error_standard_error_by_block'],
                               0.0042240731769501725, places=12)

    def test_sampled_three_round_statistics_and_fidelity(self):
        length, strength, attempts = 3, .65, 60000
        result = sim.experiment(length, strength, 0., attempts, 66191)
        expected_acceptance = .48**(length-1)
        sigma_acceptance = np.sqrt(expected_acceptance*(1-expected_acceptance)/attempts)
        self.assertLess(
            abs(result['block_acceptance'] - expected_acceptance),
            6 * sigma_acceptance,
        )
        expected_error = .5-.1*strength/length
        self.assertLess(abs(result['accepted_raw_error_rate']-expected_error),
                        6*result['accepted_raw_error_standard_error_by_block'])
        for index in range(length-1):
            expected_fidelity = (1+np.sqrt(1-strength**2)**(index+1))/2
            self.assertLess(
                abs(result['mean_fidelity_after_accepted_reset'][index]
                    - expected_fidelity),
                .012,
            )
        self.assertGreaterEqual(result['minimum_sampled_state_eigenvalue'], -1e-10)
        self.assertEqual(result['quantum_transmissions'], attempts)
        self.assertEqual(result['bsm_rounds'], sum(result['reached_by_round']))
        self.assertEqual(result['readouts'], result['bsm_rounds'])
        self.assertEqual(result['reset_attempts'], sum(result['reached_by_round'][:-1]))
        self.assertEqual(
            result['successful_reset_stages'], sum(result['reached_by_round'][1:])
        )

    def test_length_one_has_final_teleportation_and_no_reset(self):
        result = sim.experiment(1, 1., 0., 30000, 77123, link_success=.65, test_fraction=.2)
        accepted = result['delivered_blocks']-result['reserved_test_blocks']
        self.assertEqual(result['accepted_blocks'], accepted)
        self.assertEqual(result['discarded_data_blocks'], 0)
        self.assertEqual(result['quantum_transmissions'], 30000)
        self.assertEqual(result['bsm_rounds'], accepted)
        self.assertEqual(result['readouts'], accepted)
        self.assertEqual(result['reset_attempts'], 0)
        self.assertEqual(result['successful_reset_stages'], 0)
        self.assertEqual(result['classical_protocol_bits'], 2*accepted)
        self.assertEqual(result['reached_by_round'], [accepted])
        self.assertEqual(result['mean_fidelity_after_accepted_reset'], [None])
        self.assertLess(abs(result['accepted_raw_error_rate']-.4), 6*np.sqrt(.24/accepted))

    def test_all_lost_or_reserved_resources_still_count_transmissions(self):
        for link, test in ((0., 0.), (1., 1.)):
            result = sim.experiment(3, .5, 0., 100, 96, link_success=link, test_fraction=test)
            self.assertEqual(result['quantum_transmissions'], 100)
            self.assertEqual(result['accepted_blocks'], 0)
            self.assertEqual(result['bsm_rounds'], 0)
            self.assertEqual(result['readouts'], 0)
            self.assertEqual(result['reset_attempts'], 0)
            self.assertEqual(result['accepted_bits'], 0)
            self.assertIsNone(result['accepted_raw_error_rate'])
            self.assertEqual(result['exact_block_acceptance'], 0.)
            self.assertIsNone(result['exact_accepted_raw_error_rate'])
            self.assertEqual(result['reached_by_round'], [0, 0, 0])
        self.assertEqual(result['reserved_test_blocks'], 100)

    def test_retry_limit_stops_and_counts_every_failed_attempt(self):
        def always_fail(rho, bits, mode, rng):
            return np.ones(len(bits), dtype=int), rho

        with mock.patch.object(sim, 'reset', side_effect=always_fail):
            result = sim.experiment(4, .3, 0., 200, 83, max_reset_attempts=3)
        self.assertEqual(result['accepted_blocks'], 0)
        self.assertEqual(result['discarded_data_blocks'], 200)
        self.assertEqual(result['reset_attempts'], 600)
        self.assertEqual(result['readouts'], 200)
        self.assertEqual(result['successful_reset_stages'], 0)
        self.assertEqual(result['classical_protocol_bits'], 1000)
        self.assertIsNone(result['exact_block_acceptance'])


class InputValidationTests(unittest.TestCase):
    def test_experiment_rejects_invalid_counts_seeds_and_probabilities(self):
        base = dict(length=3, strength=.5, noise=0., attempts=20, seed=4)
        invalid = {
            'length': (0, -1, 1.5, True),
            'attempts': (0, -1, 2.0, False),
            'max_reset_attempts': (0, -1, 2.5, True),
            'seed': (-1, .5, None, True),
            'strength': (-.1, 1.1, np.nan, np.inf, '0.5'),
            'noise': (-.1, 1.1, np.nan, -np.inf),
            'test_fraction': (-.1, 1.1, np.nan, np.inf),
            'link_success': (-.1, 1.1, np.nan, np.inf),
            'reset_mode': ('unknown', '', None),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(parameter=name, value=value):
                    kwargs = dict(base, **{name: value})
                    with self.assertRaises(ValueError):
                        sim.experiment(**kwargs)

    def test_numpy_scalar_inputs_produce_json_serializable_results(self):
        import json

        result = sim.experiment(
            np.int64(2), np.float64(.5), np.float64(0.),
            np.int64(20), np.int64(4)
        )
        json.dumps(result, allow_nan=False)

    def test_zero_sampled_acceptance_does_not_erase_defined_exact_error(self):
        result = sim.experiment(12, .5, 0., 1, 96)
        self.assertEqual(result['accepted_blocks'], 0)
        self.assertIsNone(result['accepted_raw_error_rate'])
        self.assertGreater(result['exact_block_acceptance'], 0)
        self.assertIsNotNone(result['exact_accepted_raw_error_rate'])

    def test_instrument_sampler_rejects_invalid_branch_probabilities(self):
        rng = np.random.default_rng(7)
        for branch in (
            -sim.BELL_DM,
            2 * sim.BELL_DM,
            np.full((4, 4), np.nan),
        ):
            with self.subTest(branch=branch):
                with self.assertRaises(ValueError):
                    sim.sample([branch[None]], rng)

    def test_invalid_encoding_and_reset_mode_are_rejected(self):
        with self.assertRaises(ValueError):
            sim.encoding(np.array([0, 2]))
        with self.assertRaises(ValueError):
            sim.reset(sim.BELL_DM[None], np.array([0]), 'unknown',
                      np.random.default_rng(1))

    def test_wilson_handles_empty_trials_and_rejects_invalid_counts(self):
        self.assertEqual(sim.wilson(0, 0), [None, None])
        for successes, trials in ((1, 0), (-1, 10), (1.5, 10), (1, -2)):
            with self.subTest(successes=successes, trials=trials):
                with self.assertRaises(ValueError):
                    sim.wilson(successes, trials)


if __name__ == '__main__':
    unittest.main(verbosity=2)
