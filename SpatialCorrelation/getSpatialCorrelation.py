nRea = 100       # number of realizations
nWaves = 100     # number of plane waves per realization

# Two observation points
x = np.array([0, 0])
y = np.array([0, 0])
z = np.array([0, 250])
xyz = np.column_stack((x, y, z))  # shape [2 x 3]

vP = 4000       # P-wave velocity [m/s]
vS = vP * 0.75  # S-wave velocity [m/s]

fAll = np.arange(1, 20.01, 0.2)  # Hz

#compute spatial coherence

gammaReal = np.zeros_like(fAll)

for i, f in enumerate(fAll):
    freqs, gammaReal[i] = computeSpatialCoherence(xyz, idx1=0, idx2=1,
                                                     Nreal=nRea, nWaves=nWaves,
                                                     f=f, vP=vP, vS=vS)

# theoretical sin(kr)/kr

r = np.linalg.norm(xyz[0,:] - xyz[1,:])
kr = 2 * np.pi * fAll * r / ((vP + vS)/2)

plt.figure(figsize=(8,5))
plt.plot(fAll, gammaReal, 'b', label='Simulated coherence')
plt.plot(fAll, np.sin(kr)/kr, 'r--', label='sin(kr)/kr')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Real[Coherence]')
plt.title('Spatial Coherence between surface and cavity')
plt.legend()
plt.grid(True)
plt.show()
