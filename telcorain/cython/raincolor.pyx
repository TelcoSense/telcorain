# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

# Define NumPy dtypes
ctypedef np.uint8_t uint8_t
ctypedef np.float64_t float64_t

# Thresholds
cdef float64_t TH[14]
TH[:] = [
    0.1,
    0.115307,
    0.205048,
    0.364633,
    0.648420,
    1.153072,
    2.050483,
    3.646332,
    6.484198,
    11.53072,
    20.50483,
    36.46332,
    64.84198,
    115.3072,
]

# Colors
cdef uint8_t COL[14][4]
COL[:] = [
    (57, 0, 112, 255),
    (47, 1, 169, 255),
    (0, 0, 252, 255),
    (0, 108, 192, 255),
    (0, 160, 0, 255),
    (0, 188, 0, 255),
    (52, 216, 0, 255),
    (156, 220, 0, 255),
    (224, 220, 0, 255),
    (252, 176, 0, 255),
    (252, 132, 0, 255),
    (252, 88, 0, 255),
    (252, 0, 0, 255),
    (160, 0, 0, 255),
]

cdef inline int find_bin(float64_t v):
    if v < 0.1:
        return -1
    for i in range(14):
        if v < TH[i]:
            return i
    return 13

def rain_to_rgba(np.ndarray[float64_t, ndim=2] grid):
    """
    Cython-accelerated: convert rainfall intensities to RGBA.
    Output is identical to your original Python code.
    """

    cdef int h = grid.shape[0]
    cdef int w = grid.shape[1]
    cdef int i, j, b

    cdef np.ndarray[uint8_t, ndim=3] out = np.zeros((h, w, 4), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            v = grid[i, j]
            if v != v:  # NaN check
                continue
            b = find_bin(v)
            if b >= 0:
                out[i, j, 0] = COL[b][0]
                out[i, j, 1] = COL[b][1]
                out[i, j, 2] = COL[b][2]
                out[i, j, 3] = COL[b][3]
            # else: transparent (already zeros)

    return out
