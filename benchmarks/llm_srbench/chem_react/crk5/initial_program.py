# LLM-SRBench :: LSR-Synth / chem_react / crk5
#
# Find the mathematical function skeleton that represents Rate of change of concentration in chemistry reaction kinetics,
# given data on Time, and Concentration at time t.
#
# System under study: chemistry reaction kinetics.
#
# Every numeric constant must come from `params` (a numpy array of 10 floats),
# which is fitted to the training data by BFGS before the skeleton is scored.
# Do not hard-code fitted numbers, and do not read the data from disk.

import numpy as np

# EVOLVE-BLOCK-START
def equation(t: np.ndarray, A: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for Rate of change of concentration in chemistry reaction kinetics

    Args:
        t: A numpy array representing observations of Time.
        A: A numpy array representing observations of Concentration at time t.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing Rate of change of concentration in chemistry reaction kinetics as the result of applying the mathematical function to the inputs.
    """
    output = params[0] * t + params[1] * A + params[2]
    return output
# EVOLVE-BLOCK-END
