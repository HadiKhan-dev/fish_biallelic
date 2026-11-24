"""Convert binary calls between source and canonical search coding."""

import numpy as np


def complement_calls(calls):
    """Complement binary alleles and preserve withheld calls."""
    return np.where(calls < 0, calls, 1 - calls)
