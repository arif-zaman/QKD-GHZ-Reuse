"""Exact retry predictions for the stated symmetric GHZ model.

Bob reads out once per round after one depolarizing channel. Nothing acts on
Bob inside the reset loop. Each failed stage discards the entire block.
Costs include rejected blocks, at ideal delivery and with no reserved tests.
These predictions concern the outcome-only decoder, not a secret-key rate.
"""
from __future__ import annotations

import math
from numbers import Integral, Real


def _validate(length, delta, strength, noise, max_attempts):
    for name, value in (("length", length), ("max_attempts", max_attempts)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    values = []
    for name, value in (("delta", delta), ("strength", strength), ("noise", noise)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number in [0, 1]")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite real number in [0, 1]")
        values.append(value)
    return int(length), *values, int(max_attempts)


def _geometric_sum(base, terms):
    """Sum base**j for j=0,...,terms-1 without cancellation near one."""
    if terms == 0:
        return 0.0
    if base == 1.0:
        return float(terms)
    if base == 0.0:
        return 1.0
    return -math.expm1(terms * math.log(base)) / (1.0 - base)


def _assemble(length, delta, strength, noise, stage_acceptance,
              stage_attempts, prefinal_contrast):
    """Combine one-stage predictions into block errors and expected costs.

    Stage j is reached with probability stage_acceptance**j. Summing these
    reach probabilities includes operations spent on blocks rejected later.
    The error combines accepted pre-final rounds with the final readout.
    """
    stage_acceptance = min(1.0, max(0.0, stage_acceptance))
    block_acceptance = stage_acceptance ** (length - 1)
    mean_bsm_rounds = _geometric_sum(stage_acceptance, length)
    mean_reset_attempts = stage_attempts * _geometric_sum(stage_acceptance, length - 1)
    if length > 1 and stage_acceptance == 0.0:
        raw_error = None
    else:
        lam = 1.0 - noise
        prefinal = (prefinal_contrast or 0.0) * lam * _geometric_sum(lam, length - 1)
        raw_error = 0.5 - strength * (delta * lam**length + prefinal) / (2 * length)
        raw_error = min(1.0, max(0.0, raw_error))
    return {
        "stage_acceptance": stage_acceptance,
        "block_acceptance": block_acceptance,
        "raw_error": raw_error,
        "mean_reset_attempts": mean_reset_attempts,
        "mean_bsm_rounds": mean_bsm_rounds,
        "mean_public_bits": 2 * mean_bsm_rounds + mean_reset_attempts,
        "mean_attempts_per_reached_reset_stage": stage_attempts,
        "accepted_prefinal_contrast": prefinal_contrast,
    }


def fixed_filter_predictions(length, delta, strength, noise=0., max_attempts=1):
    """Return exact predictions for repeated use of the same reset filter.

    Parameters
    ----------
    length : int
        Candidate-bit rounds per block; there are ``length - 1`` reset stages.
    delta : float
        Symmetric encoding contrast in [0, 1].
    strength : float
        Bob's weak-readout strength in [0, 1].
    noise : float, default 0
        Depolarizing probability in [0, 1], applied once before each readout.
    max_attempts : int, default 1
        Maximum number of reset attempts at each reached stage.

    Returns
    -------
    dict
        Stage and block acceptance, accepted outcome-only raw error, mean
        reset/BSM/public-bit costs, mean attempts per reached reset stage,
        and accepted pre-final contrast. Public bits count two per BSM and
        one per reset attempt. Costs include failed blocks. ``raw_error`` is
        None for zero-acceptance multi-round blocks; a single-round block
        requires no reset.

    Notes
    -----
    A positive but extremely small block probability may underflow in
    floating point without making its conditional error undefined. In that
    case ``block_acceptance`` is zero but ``raw_error`` remains a number.
    """
    length, delta, strength, noise, max_attempts = _validate(
        length, delta, strength, noise, max_attempts)
    a_pop, b_pop = (1.0 + delta) / 2, (1.0 - delta) / 2
    sum_a = _geometric_sum(a_pop, max_attempts)
    sum_b = _geometric_sum(b_pop, max_attempts)
    success = a_pop * b_pop * (sum_a + sum_b)
    contrast = None if success == 0.0 else (sum_a - sum_b) / (sum_a + sum_b)
    attempts = a_pop * sum_a + b_pop * sum_b
    return _assemble(length, delta, strength, noise, success, attempts, contrast)


def _updated_stage(delta, max_attempts):
    """Unconditional success/reach weights for normalized amplitude squaring.

    The recursion avoids directly evaluating powers with exponent 2**j.
    When squaring reaches an absorbing failure state in floating point,
    the remaining identical reach probabilities are accumulated at once.
    """
    reach, current_contrast = 1.0, delta
    success_terms, attempt_terms = [], []
    for index in range(max_attempts):
        attempt_terms.append(reach)
        conditional_success = (1.0 - current_contrast) * (1.0 + current_contrast) / 2
        success_terms.append(reach * conditional_success)
        reach *= 1.0 - conditional_success
        if reach == 0.0:
            break
        if conditional_success == 0.0:
            attempt_terms.append((max_attempts - index - 1) * reach)
            break
        current_contrast = 2 * current_contrast / (1.0 + current_contrast**2)
    return math.fsum(success_terms), math.fsum(attempt_terms)


def amplitude_updated_predictions(length, delta, strength, noise=0., max_attempts=1):
    """Predictions for the explicitly defined known-encoding squaring schedule.

    After failure Alice replaces (a_j,b_j) by their normalized squares.
    These are known encoding coefficients, not inferred amplitudes of the
    state conditioned on Bob's private outcome. Every success-index sector
    erases the current bit; Bob's readout deformation remains. This function
    does not model extra readouts, noise, or Bob-outcome adaptation in the loop.
    Parameters, return fields, and cost conventions match
    :func:`fixed_filter_predictions`. There is one readout per reached
    round, even when multiple reset attempts are allowed.
    """
    length, delta, strength, noise, max_attempts = _validate(
        length, delta, strength, noise, max_attempts)
    success, attempts = _updated_stage(delta, max_attempts)
    contrast = 0.0 if success > 0.0 else None
    return _assemble(length, delta, strength, noise, success, attempts, contrast)
