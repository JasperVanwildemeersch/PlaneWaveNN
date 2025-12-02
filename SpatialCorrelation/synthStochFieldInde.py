def synthStochFieldInde(xyz, Nwaves, f, vP, vS):
   
    Npts = xyz.shape[0]
    omega = 2 * np.pi * f
    kP = omega / vP
    kS = omega / vS

    #P-Waves:

    thetaP = np.arccos(2 * np.random.rand(Nwaves) - 1)
    phiP = 2 * np.pi * np.random.rand(Nwaves)
    khatP = np.column_stack([
        np.sin(thetaP) * np.cos(phiP),
        np.sin(thetaP) * np.sin(phiP),
        np.cos(thetaP)
    ])
    phaseP = 2 * np.pi * np.random.rand(Nwaves)

    kvecsP = kP * khatP  # (Nwaves, 3)
    argsP = xyz @ kvecsP.T + phaseP  # (Npts, Nwaves)
    ampsP = np.exp(-1j * argsP)      # (Npts, Nwaves)

    # Weighted sum of longitudinal components
    uP = ampsP @ khatP / np.sqrt(Nwaves)  # (Npts, 3)

    #S-Waves

    thetaS = np.arccos(2 * np.random.rand(Nwaves) - 1)
    phiS = 2 * np.pi * np.random.rand(Nwaves)
    khatS = np.column_stack([
        np.sin(thetaS) * np.cos(phiS),
        np.sin(thetaS) * np.sin(phiS),
        np.cos(thetaS)
    ])
    phaseS = 2 * np.pi * np.random.rand(Nwaves)
    mixAngle = 2 * np.pi * np.random.rand(Nwaves)
   # Build orthonormal polarization bases (eSH, eSV) for each wave
    # Choose base vector that is not parallel to khatS
    ref_vecs = np.where(np.abs(khatS[:, 2:3]) < 0.99, 
                        np.tile([0, 0, 1], (Nwaves, 1)), 
                        np.tile([0, 1, 0], (Nwaves, 1)))

    eSH = np.cross(khatS, ref_vecs)
    eSH /= np.linalg.norm(eSH, axis=1, keepdims=True)
    eSV = np.cross(eSH, khatS)

    # Random linear combination of SH and SV
    polS = np.cos(mixAngle)[:, None] * eSH + np.sin(mixAngle)[:, None] * eSV

    kvecsS = kS * khatS
    argsS = xyz @ kvecsS.T + phaseS  # (Npts, Nwaves)
    ampsS = np.exp(-1j * argsS)      # (Npts, Nwaves)

    # Vectorized summation
    uS = ampsS @ polS / np.sqrt(Nwaves)  # (Npts, 3)

    return uP, uS
