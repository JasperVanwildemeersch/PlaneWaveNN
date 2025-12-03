def createSurfPoints(R, Ntheta, Nphi, theta1, theta2):

    theta, wtheta = lgwt(Ntheta, theta1, theta2)   # both shape (Ntheta,)

    phi = np.linspace(0, 2*np.pi, Nphi + 1)
    phi = phi[:-1]   # remove duplicate endpoint
    wphi = (2*np.pi) / Nphi

    PHI, THETA = np.meshgrid(phi, theta, indexing='ij')   # PHI,THETA: (Nphi × Ntheta)

    thetaPhi = np.column_stack((THETA.ravel(), PHI.ravel()))

    WTHETA = np.tile(wtheta, (Nphi, 1))                 # (Nphi × Ntheta)
    SIN_THETA = np.tile(np.sin(theta), (Nphi, 1))       # (Nphi × Ntheta)

    weights = R**2 * wphi * (SIN_THETA * WTHETA)
    weights = weights.ravel()

    return thetaPhi, weights
