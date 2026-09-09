# LLM-SRBench :: LSR-Synth / matsci / matsci5
#
# Find the mathematical function skeleton that represents Stress,
# given data on Strain, and Temperature.
#
# System under study: the stress-strain response of a material in materials science.
#
# Every numeric constant must come from `params` (a numpy array of 10 floats),
# which is fitted to the training data by BFGS before the skeleton is scored.
# Do not hard-code fitted numbers, and do not read the data from disk.

import numpy as np

# EVOLVE-BLOCK-START
def equation(epsilon: np.ndarray, T: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for Stress

    Args:
        epsilon: A numpy array representing observations of Strain.
        T: A numpy array representing observations of Temperature.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing Stress as the result of applying the mathematical function to the inputs.
    """
    output = params[0] * epsilon + params[1] * T + params[2]
    return output
# EVOLVE-BLOCK-END
