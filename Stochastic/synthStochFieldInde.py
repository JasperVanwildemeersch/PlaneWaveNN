#This is a more vectorized version of the previous synthStochFieldInde code used in SpatialCorrelation. It is also adjusted for the superposition of P and S waves.

def synthStochField(xyz, kP, kS, khatP, khatS, phaseP, phaseS, polS):
    # shapes
    Npts = xyz.shape[0]
    Nwaves = khatP.shape[0]

    # compute kvecs: (Nwaves,3)
    kvecP = (kP * khatP)        # (Nwaves,3)
    kvecS = (kS * khatS)        # (Nwaves,3)
    # compute phase arguments for all points and waves:
    argP = xyz.dot(kvecP.T) + phaseP.reshape((1, Nwaves))
    argS = xyz.dot(kvecS.T) + phaseS.reshape((1, Nwaves))

    ampP = np.exp(-1j * argP)     # (Npts, Nwaves)
    ampS = np.exp(-1j * argS)     # (Npts, Nwaves)

    uP = ampP.dot(khatP)         # complex (Npts,3)
    uS = ampS.dot(polS)          # complex (Npts,3)

    # normalization consistent with Soumen matlab code: divide by sqrt(Nwaves)
    norm = np.sqrt(Nwaves)
    uP /= norm
    uS /= norm

    return uP, uS
