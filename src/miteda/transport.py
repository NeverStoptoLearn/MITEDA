"""Analytical 2D continuous-point-source transport using AdePy."""
import numpy as np
from adepy.uniform import point2


def calculate_concentration(c0, x, y, t, v, n, al, ah, Q, B, xc, yc):
    if min(v, n, al, ah, B) <= 0 or t <= 0 or Q < 0 or c0 < 0:
        raise ValueError('Transport parameters and observation time must be valid')
    # Evaluate source and observation coordinates without substituting a different model.
    value = np.asarray(point2(c0, np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                              t, v, n, al, ah, Q / B, xc, yc), dtype=float)
    if not np.isfinite(value).all():
        raise FloatingPointError('AdePy returned non-finite concentrations')
    return value.item() if value.size == 1 else value
