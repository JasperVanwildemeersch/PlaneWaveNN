#     Vectorized getSurfNN variant.

def getSurfNN(ampOut, x, y, z, zCav, dsUnitVec):
   # Inputs:
   #   ampOut: (Npts,3) complex displacement at surface points (points on outer shell)
   #   x,y,z: (Npts,) positions (float)
   #   zCav: scalar
   #   dsUnitVec: (Npts,3) unit surface normals (precomputed for those points)
   # Returns:
   #   ISurf: (Npts,3) complex (same as MATLAB: Ids/r^3 * rVec)

    rvec = np.column_stack((x, y, z - zCav))  # shift
    rdist = np.linalg.norm(rvec, axis=1)
    tiny = 1e-20
    rdist = np.maximum(rdist, tiny)
    rdist3 = rdist**3
    # Ids = sum(ampOut .* dsUnitVec, axis=1) / rdist^3
    Ids = np.sum(ampOut * dsUnitVec, axis=1) / rdist3
    ISurf = (Ids[:, None] * rvec)
    return ISurf
