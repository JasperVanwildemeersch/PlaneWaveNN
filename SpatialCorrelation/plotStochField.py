# parameters
nR = 100          
R = 4000          
nTheta = 90      
Nwaves = 100      
f = 3             # frequency [Hz]
vP = 4000         
vS = vP * 0.75    # S-wave velocity [m/s] 3/4 of P-wave
xC, yC, zC = 0, 0, -250  # cavity cenrer coordinates

rVec, wR = lgwt(nR, 0, R)
theta, wTheta = lgwt(nTheta, np.pi/2, np.pi)
phi = np.array([0, np.pi])  # X–Z plane

# Generate points (XZ slice at y = 0)
x_list, z_list = [], []
for ph in phi:
    for r in rVec:
        x_list.append(r * np.sin(theta) * np.cos(ph))
        z_list.append(r * np.cos(theta))

# Convert to arrays
x = np.concatenate(x_list)
z = np.concatenate(z_list)
y = np.zeros_like(x)
xyz = np.column_stack((x, y, z))

# Distance to cavity center 
dist_to_cav = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
minInd = np.argmin(dist_to_cav)

# Generate stochastic field
uPAll, uSAll = synthStochFieldInde(xyz, Nwaves, f, vP, vS)
uAll = uPAll + uSAll

#  Compute ASD
uAllASD = np.abs(uAll)
uCavASD = np.abs(uAll[minInd, :])

# Normalize Z-component ASD
norm_Z = uAllASD[:, 2] / uCavASD[2]

plt.figure(figsize=(7, 6))
sc = plt.scatter(x, z, c=norm_Z, cmap='viridis', s=40)
plt.xlabel('X-coordinate (m)')
plt.ylabel('Z-coordinate (m)')
plt.title(f'ASD of Z-component of stochastic field at f = {f} Hz')
plt.colorbar(sc, label='Normalized ASD (Z / Z_cavity)')
plt.clim(norm_Z.min(), norm_Z.max())
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
