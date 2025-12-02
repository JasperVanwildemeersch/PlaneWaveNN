def computeSpatialCoherence(xyz, idx1, idx2, Nreal, nWaves, f, vP, vS):
    omega = 2 * np.pi * f
    k = omega / vP

    gamma_sum = 0.0 + 0.0j
    norm1 = 0.0
    norm2 = 0.0

    for _ in range(Nreal):
        # One realization: get P and S synthetic fields
        uP, uS = synthStochFieldInde(xyz, nWaves, f, vP, vS)

        u1 = uP[idx1, :] + uS[idx1, :]
        u2 = uP[idx2, :] + uS[idx2, :]

        gamma_sum += np.vdot(u1, u2)  # complex inner product (u1 * conj(u2))
        norm1 += np.vdot(u1, u1).real
        norm2 += np.vdot(u2, u2).real

    gamma = gamma_sum / np.sqrt(norm1 * norm2)
    gamma_real = np.real(gamma)
    freqs = f

    return freqs, gamma_real
