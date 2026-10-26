"""Independent algebraic checks of conditional erasure and general filters.

No simulator imports, measurement sampling, external data, or file writes.
"""
import unittest

import numpy as np


I = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Z = np.diag([1, -1]).astype(complex)
BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def dm(vector):
    return np.outer(vector, vector.conj())


def conj(operator, rho):
    return operator @ rho @ operator.conj().T


def normalized(rho):
    probability = np.trace(rho).real
    if probability <= 1e-14:
        raise ValueError('Zero-probability conditional state')
    return rho / probability


def encoding(bit, delta=.2):
    return np.sqrt([(1 + (-1) ** bit * delta) / 2,
                    (1 - (-1) ** bit * delta) / 2])


def encoded_dm(bit, delta=.2):
    a, b = encoding(bit, delta)
    return dm(np.array([a, 0, 0, b], dtype=complex))


def weak_kraus(strength):
    return [np.diag(np.sqrt([(1 + strength) / 2, (1 - strength) / 2])),
            np.diag(np.sqrt([(1 - strength) / 2, (1 + strength) / 2]))]


def local_map(rho, kraus):
    return sum(conj(np.kron(operator, I), rho) for operator in kraus)


def bob_map(rho, operator):
    return conj(np.kron(np.eye(rho.shape[0] // 2), operator), rho)


def bob_marginal(rho):
    dim_a = rho.shape[0] // 2
    return np.trace(rho.reshape(dim_a, 2, dim_a, 2), axis1=0, axis2=2)


def alice_marginal(rho):
    dim_a = rho.shape[0] // 2
    return np.trace(rho.reshape(dim_a, 2, dim_a, 2), axis1=1, axis2=3)


def trace_distance(rho, eta):
    return np.abs(np.linalg.eigvalsh(rho - eta)).sum() / 2


def negativity(rho):
    dim_a = rho.shape[0] // 2
    partial_transpose = rho.reshape(dim_a, 2, dim_a, 2).transpose(0, 3, 2, 1)
    values = np.linalg.eigvalsh(partial_transpose.reshape(rho.shape))
    return -values[values < 0].sum()


def record_probabilities(sigma, strength):
    return np.array([np.trace(bob_map(sigma, k)).real
                     for k in weak_kraus(strength)])


def random_dm(rng, dimension):
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension))
    return normalized(matrix @ matrix.conj().T)


class GeneralFilterChecks(unittest.TestCase):
    def assert_physical_map(self, kraus):
        effect = sum(operator.conj().T @ operator for operator in kraus)
        self.assertLessEqual(np.linalg.eigvalsh(effect).max(), 1 + 1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(effect).min(), -1e-12)

    def test_manuscript_filter_erases_full_state_and_record(self):
        accepted = []
        for bit in (0, 1):
            a, b = encoding(bit)
            sigma = local_map(encoded_dm(bit), [np.diag([b, a])])
            self.assertAlmostEqual(np.trace(sigma).real, .48)
            np.testing.assert_allclose(record_probabilities(sigma, .5), [.24, .24])
            accepted.append([normalized(bob_map(sigma, k)) for k in weak_kraus(.5)])
        np.testing.assert_allclose(accepted[0], accepted[1], atol=1e-12)

    def test_rectangular_nondiagonal_multikraus_maps_and_ordering(self):
        rng = np.random.default_rng(271828)
        for delta in (0., .2, .73, .98):
            raw = [rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
                   for _ in range(3)]
            effect_max = np.linalg.eigvalsh(sum(n.conj().T @ n for n in raw)).max()
            scale = np.sqrt((1 - delta) / (4 * effect_max))
            sigmas = []
            for bit, relative_scale in ((0, 1.), (1, .6)):
                inv_encoding = np.diag(1 / encoding(bit, delta))
                maps = [relative_scale * scale * n @ inv_encoding for n in raw]
                self.assert_physical_map(maps)
                rho = encoded_dm(bit, delta)
                sigma = local_map(rho, maps)
                sigmas.append(sigma)
                for k in weak_kraus(.51):
                    np.testing.assert_allclose(local_map(bob_map(rho, k), maps),
                                               bob_map(sigma, k), atol=1e-12)
            np.testing.assert_allclose(sigmas[1], .36 * sigmas[0], atol=1e-12)
            for strength in (0., .5, .99):
                probabilities = [record_probabilities(s, strength) / np.trace(s).real
                                 for s in sigmas]
                np.testing.assert_allclose(probabilities[0], probabilities[1], atol=1e-12)
                for k in weak_kraus(strength):
                    np.testing.assert_allclose(normalized(bob_map(sigmas[0], k)),
                                               normalized(bob_map(sigmas[1], k)), atol=1e-12)

    def test_unequal_acceptance_preserves_record_but_biases_prior(self):
        probabilities = []
        for bit, amplitude_scale in ((0, 1.), (1, .5)):
            a, b = encoding(bit)
            sigma = local_map(encoded_dm(bit), [amplitude_scale * np.diag([b, a])])
            probabilities.append(record_probabilities(sigma, .5))
        probabilities = np.array(probabilities)
        np.testing.assert_allclose(probabilities.sum(axis=1), [.48, .12])
        np.testing.assert_allclose(probabilities / probabilities.sum(axis=1)[:, None], .5)
        for prior in (np.array([.5, .5]), np.array([.3, .7])):
            posterior = prior * probabilities.sum(axis=1)
            posterior /= posterior.sum()
            for y in (0, 1):
                posterior_y = prior * probabilities[:, y]
                posterior_y /= posterior_y.sum()
                np.testing.assert_allclose(posterior_y, posterior)
        self.assertAlmostEqual(probabilities[0].sum() / probabilities.sum(), .8)

    def test_identity_filter_retains_correlation_without_state_erasure(self):
        sigmas = [encoded_dm(bit) for bit in (0, 1)]
        joint = np.array([record_probabilities(s, .5) / 2 for s in sigmas])
        self.assertAlmostEqual(joint[0, 0] + joint[1, 1], .55)
        self.assertGreater(trace_distance(normalized(bob_map(sigmas[0], weak_kraus(.5)[0])),
                                          normalized(bob_map(sigmas[1], weak_kraus(.5)[0]))), .1)

    def test_resetting_alice_only_is_not_full_resource_reset(self):
        reset_alice = np.array([[1, 1], [0, 0]], dtype=complex) / np.sqrt(2)
        self.assert_physical_map([reset_alice])
        sigmas = [local_map(encoded_dm(bit), [reset_alice]) for bit in (0, 1)]
        probabilities = np.array([record_probabilities(s, .5) for s in sigmas])
        np.testing.assert_allclose(probabilities.sum(axis=1), .5)
        self.assertAlmostEqual((probabilities[0, 0] + probabilities[1, 1]) /
                               probabilities.sum(), .55)
        for sigma in sigmas:
            for k in weak_kraus(.5):
                branch = normalized(bob_map(sigma, k))
                np.testing.assert_allclose(alice_marginal(branch), np.diag([1, 0]))
                self.assertLess(negativity(branch), 1e-12)
        self.assertGreater(trace_distance(bob_marginal(normalized(sigmas[0])),
                                          bob_marginal(normalized(sigmas[1]))), .1)

    def test_bob_marginal_suffices_without_equality_of_full_states(self):
        sigmas = []
        for bit, unitary in ((0, I), (1, H)):
            a, b = encoding(bit)
            sigmas.append(local_map(encoded_dm(bit), [unitary @ np.diag([b, a])]))
        self.assertGreater(trace_distance(normalized(sigmas[0]), normalized(sigmas[1])), .5)
        for strength in (0., .5, .99, 1.):
            for k in weak_kraus(strength):
                np.testing.assert_allclose(bob_marginal(normalized(bob_map(sigmas[0], k))),
                                           bob_marginal(normalized(bob_map(sigmas[1], k))))
            np.testing.assert_allclose(record_probabilities(sigmas[0], strength),
                                       record_probabilities(sigmas[1], strength))

    def test_projective_boundary_breaks_conditional_state_implication(self):
        sigmas = [encoded_dm(bit) for bit in (0, 1)]
        for y, k in enumerate(weak_kraus(1.)):
            target = np.zeros(4)
            target[3 * y] = 1
            for sigma in sigmas:
                np.testing.assert_allclose(normalized(bob_map(sigma, k)), dm(target))
        probabilities = np.array([record_probabilities(s, 1.) for s in sigmas])
        self.assertAlmostEqual((probabilities[0, 0] + probabilities[1, 1]) / 2, .6)

    def test_averaged_bob_state_preserves_outcome_probabilities(self):
        rng = np.random.default_rng(314159)
        for strength in (0., .17, .5, .99, 1.):
            operators = weak_kraus(strength)
            for _ in range(12):
                sigma = random_dm(rng, 6)
                averaged = sum(bob_marginal(bob_map(sigma, k)) for k in operators)
                born_on_average = [np.trace(k.conj().T @ k @ averaged).real
                                   for k in operators]
                np.testing.assert_allclose(born_on_average,
                                           record_probabilities(sigma, strength), atol=1e-12)

    def test_full_rank_effects_do_not_make_measure_prepare_injective(self):
        probabilities = np.zeros((2, 2))
        for bit in (0, 1):
            a, b = encoding(bit)
            product_input = np.kron(np.diag([1., 0.]), np.diag([a*a, b*b]))
            for y, weak in enumerate(weak_kraus(.5)):
                effect = weak.conj().T @ weak
                self.assertGreater(np.linalg.eigvalsh(effect).min(), 0)
                maps = []
                for j in (0, 1):
                    operator = np.zeros((2, 2), dtype=complex)
                    operator[0, j] = np.sqrt(effect[j, j])
                    maps.append(operator)
                np.testing.assert_allclose(sum(m.conj().T @ m for m in maps), effect)
                branch = sum(bob_map(product_input, operator) for operator in maps)
                probabilities[bit, y] = np.trace(branch).real
                np.testing.assert_allclose(normalized(branch), np.diag([1, 0, 0, 0]))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertAlmostEqual((probabilities[0, 0] + probabilities[1, 1]) / 2, .55)

    def test_fine_flags_can_reveal_joint_information_hidden_by_coarse_map(self):
        kappa = .2
        probabilities = np.zeros((2, 2, 2))  # x, announced z, private y
        coarse = []
        for bit in (0, 1):
            a, b = encoding(bit)
            maps = [np.diag([np.sqrt(kappa) / a, 0]),
                    np.diag([0, np.sqrt(kappa) / b])]
            self.assert_physical_map(maps)
            coarse.append(local_map(encoded_dm(bit), maps))
            for j, operator in enumerate(maps):
                sigma = local_map(encoded_dm(bit), [operator])
                probabilities[bit, bit ^ j, :] = record_probabilities(sigma, .5)
        np.testing.assert_allclose(coarse[0], coarse[1], atol=1e-12)
        np.testing.assert_allclose(probabilities.sum(axis=(1, 2)), .4)
        np.testing.assert_allclose(probabilities.sum(axis=1)[0], probabilities.sum(axis=1)[1])
        np.testing.assert_allclose(probabilities.sum(axis=2)[0], probabilities.sum(axis=2)[1])
        correct = sum(probabilities[x, z, y] for x in (0, 1)
                      for z in (0, 1) for y in (0, 1) if x == (y ^ z))
        self.assertAlmostEqual(correct / probabilities.sum(), .75)

    def test_exact_common_target_product_boundary_and_zero_strength(self):
        for bit in (0, 1):
            sigma_product = local_map(encoded_dm(bit), [np.diag([1, 0])])
            for k in weak_kraus(.5):
                np.testing.assert_allclose(normalized(bob_map(sigma_product, k)),
                                           np.diag([1, 0, 0, 0]))
            a, b = encoding(bit)
            sigma_entangled = local_map(encoded_dm(bit), [np.diag([b, a])])
            for k in weak_kraus(0.):
                np.testing.assert_allclose(normalized(bob_map(sigma_entangled, k)), dm(BELL))
            branches = [normalized(bob_map(sigma_entangled, k)) for k in weak_kraus(.5)]
            self.assertGreater(trace_distance(*branches), .1)
            self.assertTrue(all(negativity(branch) > .1 for branch in branches))

    def test_outcome_dependent_filter_can_restore_bell_state(self):
        for bit in (0, 1):
            amplitudes = encoding(bit)
            filters = []
            for k in weak_kraus(.5):
                conditional_amplitudes = amplitudes * np.diag(k)
                scale = conditional_amplitudes.min()
                operator = np.diag(scale / conditional_amplitudes)
                self.assert_physical_map([operator])
                filters.append(operator)
                branch = local_map(bob_map(encoded_dm(bit), k), [operator])
                np.testing.assert_allclose(normalized(branch), dm(BELL), atol=1e-12)
                self.assertAlmostEqual(np.trace(branch).real, 2 * scale ** 2)
            self.assertGreater(np.linalg.norm(filters[0] - filters[1]), .1)

    def test_zero_probability_endpoint_is_not_a_conditional_state(self):
        for bit in (0, 1):
            a, b = encoding(bit, 1.)
            sigma = local_map(encoded_dm(bit, 1.), [np.diag([b, a])])
            np.testing.assert_allclose(sigma, 0)
            with self.assertRaises(ValueError):
                normalized(sigma)



if __name__ == '__main__':
    unittest.main(verbosity=2)
