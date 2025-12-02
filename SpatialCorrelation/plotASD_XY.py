def plotASD_XZ(uTotal, xyz, f, y0, tol=1e-6):
# Select XZ plane
    sel = np.abs(xyz[:, 1] - y0) < tol
    xyz_sel = xyz[sel]
    u_sel = uTotal[sel]

    if xyz_sel.size == 0:
        raise ValueError(f"No points found near y = {y0} within tolerance {tol}")

    #Compute ASD
    # ASD = sqrt(2) * |amplitude| for one-sided PSD
    asdX = np.sqrt(2) * np.abs(u_sel[:, 0])
    asdZ = np.sqrt(2) * np.abs(u_sel[:, 2])

    # Reshape into X-Z grid
    xVals = np.unique(np.round(xyz_sel[:, 0], decimals=8))
    zVals = np.unique(np.round(xyz_sel[:, 2], decimals=8))

    Nx, Nz = len(xVals), len(zVals)

    # Assume points are ordered as all z for each x
    try:
        asdXgrid = asdX.reshape((Nx, Nz)).T
        asdZgrid = asdZ.reshape((Nx, Nz)).T
    except ValueError:
        # Fallback: rebuild grid explicitly (if ordering differs)
        X, Z = np.meshgrid(xVals, zVals, indexing='xy')
        asdXgrid = np.full_like(X, np.nan, dtype=float)
        asdZgrid = np.full_like(Z, np.nan, dtype=float)
        for i, (x, z, ax, az) in enumerate(zip(xyz_sel[:, 0], xyz_sel[:, 2], asdX, asdZ)):
            ix = np.argmin(np.abs(xVals - x))
            iz = np.argmin(np.abs(zVals - z))
            asdXgrid[iz, ix] = ax
            asdZgrid[iz, ix] = az

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(10 * np.log10(asdXgrid), 
                         extent=[xVals.min(), xVals.max(), zVals.min(), zVals.max()],
                         origin='lower', aspect='auto')
    axes[0].set_title(f'ASD X-component at f = {f:.1f} Hz')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('z (m)')
    fig.colorbar(im1, ax=axes[0], label='ASD (dB)')

    im2 = axes[1].imshow(10 * np.log10(asdZgrid), 
                         extent=[xVals.min(), xVals.max(), zVals.min(), zVals.max()],
                         origin='lower', aspect='auto')
    axes[1].set_title(f'ASD Z-component at f = {f:.1f} Hz')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('z (m)')
    fig.colorbar(im2, ax=axes[1], label='ASD (dB)')

    plt.tight_layout()
    plt.show()
