# LLM-SRBench :: LSR-Synth / phys_osc / po3
#
# Find the mathematical function skeleton that represents Acceleration in Nonl-linear Harmonic Oscillator,
# given data on Time, and Velocity at time t.
#
# System under study: a non-linear damped harmonic oscillator in physics.
#
# Every numeric constant must come from `params` (a numpy array of 10 floats),
# which is fitted to the training data by BFGS before the skeleton is scored.
# Do not hard-code fitted numbers, and do not read the data from disk.

import numpy as np

# EVOLVE-BLOCK-START
def equation(t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for Acceleration in Nonl-linear Harmonic Oscillator

    Args:
        t: A numpy array representing observations of Time.
        v: A numpy array representing observations of Velocity at time t.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing Acceleration in Nonl-linear Harmonic Oscillator as the result of applying the mathematical function to the inputs.
    """
    output = params[0] * t + params[1] * v + params[2]
    return output
# EVOLVE-BLOCK-END
