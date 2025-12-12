# this script estimates the Newtonian acceleration from a stochastic underground seismic field produced from a superposition of plane P and S waves

# The script works for fAll containing only 1 frequency: the code must be run for each frequency separately to store each file separately.

# Vectorized, optimized stochastic plane-wave Newtonian-noise simulation
# Implements optimizations:
#  Vectorize P+S wave superposition
#  Precompute things that dont depend on radius r
#  Precompute directional vectors for the stochastic field
#  Move cavity masking out of main loops
#  Evaluate P+S on all radii at once per realization

#Takes about 24 secs per realization for Nwaves=100

#New helper functions:

def buildPolarUnitVectors(thetaFlat, phiFlat):
    #Return unit direction vectors for each angular point (Nang,3)
    sinT = np.sin(thetaFlat)
    cosT = np.cos(thetaFlat)
    cosP = np.cos(phiFlat)
    sinP = np.sin(phiFlat)
    ux = sinT * cosP
    uy = sinT * sinP
    uz = cosT
    return np.column_stack((ux, uy, uz))
    
def computeShSvPolarizations(khatS):
    #Given khatS (Nwaves,3) compute two orthonormal perpendicular vectors for each wave and produce a random polarization polS = cos(a)*eSH + sin(a)*eSV given mix angles passed externally.
    #input: propagation direction of s-wave
    #output: unit vector components sh and sv, perpendicular to propagation direction, linear combination of these gives s-wave polarization/displacement
    Nwaves = khatS.shape[0]
    eSH = np.zeros_like(khatS)
    eSV = np.zeros_like(khatS)
    for i in range(Nwaves):
        n = khatS[i]
        if abs(n[2]) < 0.99:
            v = np.array([0.0, 0.0, 1.0])
        else:
            v = np.array([0.0, 1.0, 0.0])
        sh = np.cross(n, v)
        sh_norm = np.linalg.norm(sh)
        if sh_norm < 1e-12:
            # fallback
            sh = np.cross(n, np.array([1.0, 0.0, 0.0]))
            sh_norm = np.linalg.norm(sh)
            if sh_norm < 1e-12:
                sh = np.array([1.0, 0.0, 0.0])
                sh_norm = 1.0
        eSH[i] = sh / sh_norm
        eSV[i] = np.cross(eSH[i], n)  # automatically unit if eSH and n are unit and perpendicular
    return eSH, eSV

# parameters

sim_case = "halfSpace"  # "fullSpace" or "halfSpace"
vP = 4000.0  
vS = 3000.0  
rho = 2800.0  # (kg/m^3)
nRea = 100    # number of realizations
Nwaves = 100  # plane waves per realization
G = 6.67430e-11  # gravitational constant
fAll = [4.0]    # frequencies in Hz
R = 5000.0    # radius of integration

# Radial / angular discretization (same as getP/SWaveNN)

if sim_case == "halfSpace":
    nR = 120
    nTheta = 40
    nPhi = 200
    rCav = 20.0
    zCav = -250.0
    xCav = yCav = 0.0

    bounds = [0, -zCav-2*rCav, -zCav+2*rCav, R]
    pointsPerRegion = [30, 30, 60]  # sum = nR is beter
    if sum(pointsPerRegion) != nR:
        # make simple split proportional if not exact
        pointsPerRegion = [max(1, int(nR * p / sum(pointsPerRegion))) for p in pointsPerRegion]
        # fix last to match nR exactly
        pointsPerRegion[-1] = nR - sum(pointsPerRegion[:-1])

    r_subs = [] #r_subs = quadrature nodes
    w_subs = [] #w_subs = quadrature weights
    for n_pts, (low, high) in zip(pointsPerRegion, zip(bounds[:-1], bounds[1:])):
        r_sub, w_sub = lgwt(n_pts, low, high)
        r_subs.append(r_sub)
        w_subs.append(w_sub)

    #rVec: array of radius values, rVec[i] is a radial shell, wR[i] gives the integration weight of this shell
    rVec = np.concatenate(r_subs)
    wR = np.concatenate(w_subs)
    
    # Theta: Integrate over θ∈[π/2,π] since θ is angle with z-axis: surface is at θ = π/2, downward direction is θ = π
    theta1, theta2 = np.pi/2, np.pi
    theta, wTheta = lgwt(nTheta, theta1, theta2)
    
    #phi: azimuthal angle: uniform distribution φ∈[0,2π]
    phi = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
    wPhi = 2*np.pi / nPhi
elif sim_case == "fullSpace":
    nR = 120
    nTheta = 80
    nPhi = 200
    rCav = 20.0
    zCav = 0.0
    xCav = yCav = 0.0

    bounds = [0, rCav, R]
    pointsPerRegion = [10, 110]
    if sum(pointsPerRegion) != nR:
        pointsPerRegion = [max(1, int(nR * p / sum(pointsPerRegion))) for p in pointsPerRegion]
        pointsPerRegion[-1] = nR - sum(pointsPerRegion[:-1])

    r_subs = []
    w_subs = []
    for n_pts, (low, high) in zip(pointsPerRegion, zip(bounds[:-1], bounds[1:])):
        r_sub, w_sub = lgwt(n_pts, low, high)
        r_subs.append(r_sub)
        w_subs.append(w_sub)
    rVec = np.concatenate(r_subs)
    wR = np.concatenate(w_subs)
    theta1, theta2 = 0.0, np.pi
    theta, wTheta = lgwt(nTheta, theta1, theta2)
    phi = np.linspace(0, 2*np.pi, nPhi, endpoint=False)
    wPhi = 2*np.pi / nPhi

# Flatten theta-phi grid and precompute unit vectors ( dont depend on r)

TH, PH = np.meshgrid(theta, phi)  # shape MATLAB is (nPhi, nTheta), but here it's (len(phi), len(theta))
# produce flattened arrays with length Nang = nTheta*nPhi
thetaFlat = TH.T.flatten()  # ensure theta varies fastest similar to MATLAB's meshgrid(theta,phi)
phiFlat = PH.T.flatten()
Nang = thetaFlat.size

# unit direction vectors for all angular points (Nang,3) in carthesian coordinates
unitDirs = buildPolarUnitVectors(thetaFlat, phiFlat)  # columns [sinθ cosφ, sinθ sinφ, cosθ]

# angular quadrature weights (independent of r)
wtTheta = np.tile(wTheta, nPhi)   # (nTheta*nPhi,) so we must have wTheta nPhi times
angularWeights = np.sin(thetaFlat) * wtTheta * wPhi   # (Nang,) actual computation of the angular weights dependent of both (theta, phi)

# Precompute full coordinate grid (nR x Nang)

rVec = np.asarray(rVec)
wR = np.asarray(wR)
nR = rVec.size

dtype_coord = np.float64
x_all = (rVec[:, None].astype(dtype_coord)) * unitDirs[None, :, 0]  # (nR, Nang)
y_all = (rVec[:, None].astype(dtype_coord)) * unitDirs[None, :, 1]
z_all = (rVec[:, None].astype(dtype_coord)) * unitDirs[None, :, 2]

# Precompute cavity mask once (nR, Nang)

cavity_mask = (x_all**2 + y_all**2 + (z_all - zCav)**2) < (rCav**2)  # boolean, True when inside cavity

# Precompute flattened indices for all non-cavity points
#gives indices of grid points outside cavity
r_inds, ang_inds = np.nonzero(~cavity_mask)   # arrays of same length = Npoints_total
Npoints_total = r_inds.size

# Precompute arrays needed for volume integration weights later
#select grid points (r>rCav) outside cavity with their weights
r_vals = rVec[r_inds]          # (Npoints_total,)
wr_vals = wR[r_inds]           # radial GL weights for each point
w_ang_vals = angularWeights[ang_inds]  # angular quadrature weight for each angular index

# Prepare surface integration unit normals and weights for outer shell (r = rVec[-1])

#check if surface is in cavity, if not then True
outer_mask = ~cavity_mask[-1]   # (Nang,)

#Create all grid points [theta, phi] on the spherical shell with r=rMax, so the integration surface and their weights
thetaPhi, weights_surf = createSurfPoints(np.max(rVec), nTheta, nPhi, theta1, theta2)

#This constructs the outward unit normal vector at each surface point:
dsUnitVec_all = np.column_stack((np.sin(thetaPhi[:,0]) * np.cos(thetaPhi[:,1]),
                                 np.sin(thetaPhi[:,0]) * np.sin(thetaPhi[:,1]),
                                 np.cos(thetaPhi[:,0])))  # (Nang,3)

# Initialize output arrays
IVolTotAll = np.zeros((nRea, len(fAll), 3), dtype=np.complex128)
ISurfTotAll = np.zeros_like(IVolTotAll)
uCavASDX = np.zeros((nRea, len(fAll)), dtype=np.float64)
uCavASDY = np.zeros((nRea, len(fAll)), dtype=np.float64)
uCavASDZ = np.zeros((nRea, len(fAll)), dtype=np.float64)

# MAIN: loop over realizations and frequencies

t_start = time.time()
for reaNo in range(nRea):
    print(f"Realization {reaNo+1}/{nRea}", flush=True)
    # Random wave parameters for this realization (precompute)

    #P-waves
    # random directions on sphere (theta: [0,pi], phi: [0,2pi])
    # random phase [0,2pi]
    thetaWP = np.arccos(2*np.random.rand(Nwaves) - 1.0)   # (Nwaves,)
    phiWP = 2.0 * np.pi * np.random.rand(Nwaves)
    phaseP = 2.0 * np.pi * np.random.rand(Nwaves)

    #S-waves
    thetaWS = np.arccos(2*np.random.rand(Nwaves) - 1.0)
    phiWS = 2.0 * np.pi * np.random.rand(Nwaves)
    phaseS = 2.0 * np.pi * np.random.rand(Nwaves)

    #random polarization angle [0, 2pi] used in combination with ComputeShSvPolarization: polS = cos(mixAngle) * eSH + sin(mixAngle) * eSV
    mixAngle = 2.0 * np.pi * np.random.rand(Nwaves)  # polarization mix angle
    
    # compute khatP, khatS (Nwaves,3)
    #Compute cartesian unit vectors from spherical angles
    khatP = np.column_stack((np.sin(thetaWP)*np.cos(phiWP),
                             np.sin(thetaWP)*np.sin(phiWP),
                             np.cos(thetaWP)))
    khatS = np.column_stack((np.sin(thetaWS)*np.cos(phiWS),
                             np.sin(thetaWS)*np.sin(phiWS),
                             np.cos(thetaWS)))
    # k magnitudes
    # but we'll compute kP = omega/vP per frequency below

    #S-waves: basis vecors for displacement (perpendicular to propagation)
    eSH, eSV = computeShSvPolarizations(khatS)
    polS = np.cos(mixAngle)[:,None] * eSH + np.sin(mixAngle)[:,None] * eSV  # (Nwaves,3)

    # This loop computes the response at each frequency, using the wave directions and phases for this realization determined earlier
    for fNo, f in enumerate(fAll):
        omega = 2.0 * np.pi * f # ω=2πf
        kP = omega / vP # k=ω/v
        kS = omega / vS
        
        # Evaluate displacement at cavity center

        #cavity coordinates
        xyz_cav = np.array([[xCav, yCav, zCav]], dtype=np.float64)
        
        #calculate displacement at the cavity due to the stochastic superposition of seismic waves
        uP_cav, uS_cav = synth_stoch_field_vectorized(xyz_cav, kP, kS, khatP, khatS, phaseP, phaseS, polS)
        uCavTot = uP_cav + uS_cav

        #Extract ASD x- y- and z-components and store for each realization and frequency
        uCavASDX[reaNo, fNo] = np.abs(uCavTot[0,0]) #Why abs(): The displacement is complex:u=A*e^(−iϕ), so amplitude A = |u|
        uCavASDY[reaNo, fNo] = np.abs(uCavTot[0,1]) #shape (nRea, nFreq)
        uCavASDZ[reaNo, fNo] = np.abs(uCavTot[0,2]) #Example: uCavASDZ[reaNo, fNo] = |u_z(f)| for this realization
        #So for each realization (reaNo) and frequency (fNo) you store the displacement amplitude along each axis.

        # Evaluate the synthetic field on ALL non-cavity points (all radii x angular)
        # We create xyz_points once per realization & freq (vectorized).

        # Build xyz_points from precomputed x_all, y_all, z_all arrays using flattened indices
        x_points = x_all.ravel()[~cavity_mask.ravel()]  # another way but simpler to use r_inds, ang_inds
        # safer: use r_inds/ang_inds
        x_points = x_all[r_inds, ang_inds]
        y_points = y_all[r_inds, ang_inds]
        z_points = z_all[r_inds, ang_inds]
        xyz_points = np.column_stack((x_points, y_points, z_points))  # shape (Npoints_total,3)

        # compute uP, uS for all those points in one shot
        uP_all, uS_all = synthStochField(xyz_points, kP, kS, khatP, khatS, phaseP, phaseS, polS)
        uTot_all = uP_all + uS_all   # (Npoints_total,3) complex

        # Volume integral contribution: vectorized

        IVol_all = getVolNN(uTot_all, xyz_points, zCav)  # (Npoints_total,3) complex

        # full 3D integration weights: r^2 * angularWeight * radialWeight
        full_weights = (r_vals**2 * w_ang_vals * wr_vals)    # (Npoints_total,)
        IVolTot = np.sum(IVol_all * full_weights[:, None], axis=0)  # (3,) complex

        # Surface integral: evaluate only at outer shell where r = rVec[-1]
        outer_ang_idx = np.nonzero(outer_mask)[0]  # angular indices on outer shell that are not in cavity
        if outer_ang_idx.size > 0:
            xS = x_all[-1, outer_ang_idx]
            yS = y_all[-1, outer_ang_idx]
            zS = z_all[-1, outer_ang_idx]
            xyz_outer = np.column_stack((xS, yS, zS))
            # displacement at outer surface points (vectorized)
            uP_outer, uS_outer = synthStochField(xyz_outer, kP, kS, khatP, khatS, phaseP, phaseS, polS)
            uTot_outer = uP_outer + uS_outer
            dsUnitVecMask = dsUnitVec_all[outer_ang_idx]
            weightsMask = weights_surf[outer_ang_idx]
            ISurf_all = getSurfNN(uTot_outer, xS, yS, zS, zCav, dsUnitVecMask)
            ISurfTot = np.sum(ISurf_all * weightsMask[:, None], axis=0)
        else:
            ISurfTot = np.zeros(3, dtype=np.complex128)

        # Save
        IVolTotAll[reaNo, fNo, :] = IVolTot
        ISurfTotAll[reaNo, fNo, :] = ISurfTot

t_end = time.time()
print(f"Done. Time elapsed: {t_end - t_start:.1f} s")

plt.figure(figsize=(12,4))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.plot(np.arange(1,nRea+1), G*rho*np.abs(IVolTotAll[:,0,i]), 'bo', label='Volume')
    plt.plot(np.arange(1,nRea+1), G*rho*np.abs(ISurfTotAll[:,0,i]), 'ro', label='Surface')
    plt.title(['X','Y','Z'][i])
    plt.xlabel('Realization')
    plt.ylabel('NN (m/s^2/√Hz)')
    plt.legend()
plt.tight_layout()
plt.show()

# Save to mat file

fPathSave = 'SaveSim/'
fNameSave = f"{fPathSave}NewFreq{int(fAll[0])}Hz.mat"
savemat(fNameSave, {
    'uCavASDX': uCavASDX,
    'uCavASDY': uCavASDY,
    'uCavASDZ': uCavASDZ,
    'IVolTotAll': IVolTotAll,
    'ISurfTotAll': ISurfTotAll,
    'nRea': nRea,
    'f': fAll[0]
})
print("Saved:", fNameSave)
