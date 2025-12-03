#     Vectorized getVolNN for already computed displacement uTot at positions xyz.

def getVolNN(uTot, xyz, zCav):
    # Inputs:
    #  uTot: (Npts,3) complex displacement vectors
    #  xyz: (Npts,3) positions
    #  zCav: scalar
    # Returns:
    #  IFull: (Npts,3) complex
   
    rvec = xyz.copy()          # (Npts,3) floats
    rvec[:,2] = rvec[:,2] - zCav
    rdist = np.linalg.norm(rvec, axis=1)  # (Npts,)
    # avoid divide-by-zero
    tiny = 1e-20
    rdist = np.maximum(rdist, tiny)
    rdist3 = rdist**3
    rcap = rvec / rdist[:, None]    # (Npts,3)
    # I1 = ampOut / r^3 (component-wise)
    I1 = uTot / rdist3[:, None]
    # I2: projection times 3 rcap rcap / r^3
    dot = np.sum(rcap * uTot, axis=1)   # (Npts,) complex
    I2 = 3.0 * (dot[:, None] * rcap) / rdist3[:, None]
    IFull = I1 - I2
    return IFull
