# Model and reproduction conventions

## One block

Alice prepares an (L+1)-qubit GHZ state and transmits one qubit to Bob. Her
remaining register occupies a two-dimensional logical subspace. The simulator
therefore stores a 4x4 joint density matrix in the order
`|0_L 0>, |0_L 1>, |1_L 0>, |1_L 1>`.
Bob's errors can populate the odd-parity sectors; these are retained.
This reduction is exact for the operations below. It excludes arbitrary
leakage and gate errors on individual physical GHZ qubits.

A block has L candidate-bit rounds and L-1 intervening reset stages. In each
round:

1. Alice chooses an independent, uniform bit and prepares its encoding ancilla.
   For bit zero, the populations are (0.6, 0.4); bit one swaps them.
2. A complete Bell-state measurement (BSM) instrument is sampled and its
   announced correction is applied.
   The final BSM consumes Alice's last logical qubit, leaving Bob's 2x2 state.
3. Bob's channel acts once, then his weak-Z readout is sampled. He uses the
   readout outcome as his bit estimate.
4. Except in the last round, Alice attempts a reset. Its observed outcome, the
   herald, is zero for success and one for failure. After a failure, the state
   is retained with its failure correction;
   a retry uses the same fixed filter. Exhausting the retry limit rejects the
   whole block.

Each quantum instrument specifies both outcome probabilities and conditional
state changes. Protocol choices use the selected encoding and observed heralds;
density matrices and fidelity are used only to calculate and diagnose the
simulated evolution. A reset herald need not certify restored entanglement.

## Parameters

| Symbol | Python name | Meaning |
|---|---|---|
| L | `length` | Candidate-bit rounds per block; positive integer |
| delta | `delta` (analysis only) | Symmetric encoding population contrast |
| g | `strength` | Weak-Z readout strength, from 0 to 1 |
| p | `noise` | Bob's per-round replacement/depolarization probability |
| R | `max_reset_attempts` (simulation), `max_attempts` (analysis) | Maximum attempts per reset stage |

The channel is `(1-p)*rho + p*Tr_B(rho) tensor I/2`, applied after the corrected
BSM and before readout, including the final round. No additional channel or
readout is applied during the reset loop. `p` is a dimensionless probability,
not a loss rate, storage time, or hardware noise model.

The Monte Carlo API also supports `link_success` and `test_fraction`. These
model delivery and reservation only; reserved blocks do not execute a security
test. All manuscript experiments use perfect delivery and no reserved tests.

## Reset policies and exact predictions

For encoding amplitudes (a,b), the coherent fixed success and failure filters
are `diag(b,a)` and `diag(a,b)` after the stated failure correction. The mixed
ancilla instead dephases Alice's logical qubit before these branches; equal
success probabilities do not imply equal retained coherence.

For one coherent attempt, let `r=(1-delta**2)/2`. Then

```text
P_accept = r**(L-1)
Q_raw    = 1/2 - delta*g*(1-p)**L/(2*L)
```

Before the final round, the accepted record is independent of the encoded bit.
`Q_raw` uses Bob's outcome-only estimates. It is not an optimal decoder bound
for the full transcript or a secret-key fraction.

For fixed-filter retries, set `A=(1+delta)/2`, `B=(1-delta)/2`. At p=0:

```text
s_R      = 1 - A**(R+1) - B**(R+1)
d_R      = (delta - A**(R+1) + B**(R+1)) / s_R
P_accept = s_R**(L-1)
Q_raw    = 1/2 - g*(delta + (L-1)*d_R)/(2*L)
```

Attempts within a stage are generally dependent. Summing over a complete
stage gives the same acceptance probability for every earlier history, so
later survival does not further bias an earlier accepted round's errors.
At delta=0 the normalized state is unchanged by either filter and reset
outcomes are independent with success probability 1/2.

The amplitude-updated schedule replaces known coefficients (a_j,b_j), after
each failure, by their normalized squares. It does not infer amplitudes from
Bob's private outcome. Every successful attempt removes the encoded bit's
dependence from that branch. `analysis.py` computes both policies' acceptance,
error, and costs, including the specified noisy extension. Independent circuit
tests check these formulas rather than reusing them to generate the expected
answer.

At delta=1 and L>=2 no reset stage succeeds; conditional errors are `None`.
L=1 requires no reset. Extremely small positive block probabilities can
underflow to zero numerically while the analytical conditional error remains
defined; this is documented in the analysis API.

## Statistics and resource accounting

- Each attempted block uses one quantum transmission and one GHZ preparation.
  Rejected blocks contribute to costs. Local operations stop when a stage fails.
- `accepted_bits` counts retained candidate labels, not secret bits.
  `raw_bits_per_transmission` divides these labels by all attempted blocks.
- `accepted_raw_error_rate` pools bit errors over fully accepted blocks.
  Its standard error is computed from the error fraction of each independent
  accepted block, allowing correlations among bits within a block.
- Acceptance intervals are pointwise 95% Wilson intervals. An individual
  interval may exclude theory. The 20-seed check reports every planned run;
  interval coverage and count variance are descriptive statistics.
- With no accepted block, sampled conditional error is `None`; with fewer than
  two accepted blocks, its standard error is `None`. CSV represents these as
  empty cells and JSON as `null`.
- `reset_attempts`, `bsm_rounds`, and ancilla counts include failed blocks.
  Modeled public communication is two bits per BSM and one herald per reset
  attempt. It excludes testing messages, authentication, reconciliation, and
  privacy amplification.
- Fidelity and minimum-eigenvalue checks are offline numerical diagnostics,
  not inputs to the simulated acceptance decision.

## What the checks establish

`test_simulation.py` compares the compressed instruments with full physical
CNOT/Hadamard/projection circuits. `test_parameterized_family.py` enumerates
branches for general contrast. `test_analysis.py` propagates error-weighted
states through both retry schedules. `test_general_filters.py` tests the
conditional-erasure criterion and failures of its premises, including retained
flags, unequal acceptance, noninvertible readout, and outcome-dependent maps.

These tests and the sampled results establish agreement with the specified
operations. They do not establish composable security, finite-key performance,
a new positive-rate protocol, or a hardware operating advantage.
