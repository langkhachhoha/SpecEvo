# LLM-SRBench :: LSR-Synth / bio_pop_growth / bpg9
#
# Find the mathematical function skeleton that represents Population growth rate,
# given data on Time, and Population at time t.
#
# System under study: biological population growth.
#
# Every numeric constant must come from `params` (a numpy array of 10 floats),
# which is fitted to the training data by BFGS before the skeleton is scored.
# Do not hard-code fitted numbers, and do not read the data from disk.

import numpy as np

# EVOLVE-BLOCK-START
def equation(t: np.ndarray, P: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for Population growth rate

    Args:
        t: A numpy array representing observations of Time.
        P: A numpy array representing observations of Population at time t.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing Population growth rate as the result of applying the mathematical function to the inputs.
    """
    output = params[0] * t + params[1] * P + params[2]
    return output
# EVOLVE-BLOCK-END
