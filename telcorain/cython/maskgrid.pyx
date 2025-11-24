# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

from shapely.geometry import Point
from shapely.prepared import PreparedGeometry
from shapely import geos

ctypedef np.float64_t float64_t


def mask_grid_fast(
    np.ndarray[float64_t, ndim=2] data_grid,
    np.ndarray[float64_t, ndim=2] x_grid,
    np.ndarray[float64_t, ndim=2] y_grid,
    PreparedGeometry prep_poly,
    tuple bbox,
):
    """
    Cython-accelerated masking using raw GEOS API.
    Very fast version of your Python implementation.
    """

    cdef float64_t minx = bbox[0]
    cdef float64_t miny = bbox[1]
    cdef float64_t maxx = bbox[2]
    cdef float64_t maxy = bbox[3]

    cdef int rows = x_grid.shape[0]
    cdef int cols = x_grid.shape[1]

    cdef np.ndarray out = data_grid.copy()

    # Direct pointer to GEOS geometry
    cdef void* ptr = prep_poly.context._geom
    cdef const GEOSPreparedGeometry* pg = <const GEOSPreparedGeometry*> ptr

    cdef float64_t lon, lat
    cdef GEOSContextHandle_t ctx = geos.geos_handle

    for i in range(rows):
        for j in range(cols):

            lon = x_grid[i, j]
            lat = y_grid[i, j]

            # quick bounding box reject
            if lon < minx or lon > maxx or lat < miny or lat > maxy:
                out[i, j] = np.nan
                continue

            # construct GEOS point directly
            cdef GEOSGeometry* pt = GEOSGeom_createPoint_r(
                ctx,
                GEOSCoordSeq_create_r(ctx, 1, 2)
            )

            GEOSCoordSeq_setX_r(ctx, pt._coords, 0, lon)
            GEOSCoordSeq_setY_r(ctx, pt._coords, 0, lat)

            if not GEOSPreparedContains_r(ctx, pg, pt):
                out[i, j] = np.nan

            GEOSGeom_destroy_r(ctx, pt)

    return out
