def load_sim_file(filepath):
    #Try to load MATLAB .mat saved simulation file.
    #If it's not a .mat, try numpy .npz.
    #Returns a dict-like object (numpy arrays).
    if not os.path.exists(filepath):
        # try .npz alternative with same base name
        base, _ = os.path.splitext(filepath)
        npz_path = base + '.npz'
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            return dict(data)
        raise FileNotFoundError(f"File not found: {filepath} (also tried {npz_path})")
    
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    return mat

def get_array(matdict, key):
    #Extract array from MATLAB dictionary
    if key in matdict:
        return np.asarray(matdict[key])
        # unwrap MATLAB object arrays

    for k, v in matdict.items():
        if isinstance(v, dict) and key in v:
            return np.asarray(v[key])
    raise KeyError(f"Variable '{key}' not found in MAT-file.")

def assemble_nn(fAll=None, sim_dir='SaveSim', prefix='NewFreq', rho=2800.0, G=6.67430e-11):
    #Returns:
    #  (fAll, IVolEst, ISurfEst, ITotEst), figure.    
    
    if fAll is None:
        fAll = np.array([2.0, 4.0])
    else:
        fAll = np.asarray(fAll, dtype=float)
    
    lenF = len(fAll)
    IVolEst = np.zeros((lenF, 3), dtype=float)
    ISurfEst = np.zeros((lenF, 3), dtype=float)
    ITotEst = np.zeros((lenF, 3), dtype=float)

    for i, f in enumerate(fAll):
        f_str = ('%g' % f)
        fname_mat = os.path.join(sim_dir, f"{prefix}{f_str}Hz.mat")
        try:
            data = load_sim_file(fname_mat)
        except FileNotFoundError:
            fname_mat = f"{prefix}{f_str}Hz.mat"
            data = load_sim_file(fname_mat)

        # Load arrays
        uCavASDX = get_array(data, 'uCavASDX')
        uCavASDY = get_array(data, 'uCavASDY')
        uCavASDZ = get_array(data, 'uCavASDZ')
        IVolTotAll = get_array(data, 'IVolTotAll')
        ISurfTotAll = get_array(data, 'ISurfTotAll')

        # Ensure proper dimensions and select current frequency
        if uCavASDX.ndim > 1:
            uCavASDX_f = uCavASDX[:, i] if uCavASDX.shape[1] >= lenF else uCavASDX.ravel()
            uCavASDY_f = uCavASDY[:, i] if uCavASDY.shape[1] >= lenF else uCavASDY.ravel()
            uCavASDZ_f = uCavASDZ[:, i] if uCavASDZ.shape[1] >= lenF else uCavASDZ.ravel()
        else:
            uCavASDX_f = uCavASDX
            uCavASDY_f = uCavASDY
            uCavASDZ_f = uCavASDZ

        if IVolTotAll.ndim == 3 and IVolTotAll.shape[1] >= lenF:
            IVolTotAll_f = IVolTotAll[:, i, :]
            ISurfTotAll_f = ISurfTotAll[:, i, :]
        else:
            IVolTotAll_f = IVolTotAll
            ISurfTotAll_f = ISurfTotAll

        # Compute scalars safely
        sFASDX = float(np.mean(np.abs(uCavASDX_f)))
        sFASDY = float(np.mean(np.abs(uCavASDY_f)))
        sFASDZ = float(np.mean(np.abs(uCavASDZ_f)))

        rms_IVol = np.sqrt(np.mean(np.abs(IVolTotAll_f)**2, axis=0))
        rms_ISurf = np.sqrt(np.mean(np.abs(ISurfTotAll_f)**2, axis=0))
        rms_ITot = np.sqrt(np.mean(np.abs(IVolTotAll_f - ISurfTotAll_f)**2, axis=0))

        # Multiply by G*rho*sFASD per component
        IVolEst[i, 0] = G * rho * sFASDX * rms_IVol[0]
        IVolEst[i, 1] = G * rho * sFASDY * rms_IVol[1]
        IVolEst[i, 2] = G * rho * sFASDZ * rms_IVol[2]

        ISurfEst[i, 0] = G * rho * sFASDX * rms_ISurf[0]
        ISurfEst[i, 1] = G * rho * sFASDY * rms_ISurf[1]
        ISurfEst[i, 2] = G * rho * sFASDZ * rms_ISurf[2]

        ITotEst[i, 0] = G * rho * sFASDX * rms_ITot[0]
        ITotEst[i, 1] = G * rho * sFASDY * rms_ITot[1]
        ITotEst[i, 2] = G * rho * sFASDZ * rms_ITot[2]

        print(f"Processed frequency {f_str} Hz -> file {fname_mat}")

    # Plot
    comps = ['X', 'Y', 'Z']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, ax in enumerate(axes):
        ax.plot(fAll, IVolEst[:, idx], 'b-', label='Volume', linewidth=2)
        ax.plot(fAll, ISurfEst[:, idx], 'ro', label='Surface')
        ax.plot(fAll, ITotEst[:, idx], 'mo', label='Total')
        ax.plot(fAll, G*rho*4*np.pi/3*np.ones_like(fAll), 'g', linewidth=2, label=r'4\pi/3 G\rho')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(f'{comps[idx]} component')
        if idx == 0:
            ax.set_ylabel('NN ASD (m/s^2 / √Hz)')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("AssembleNN23456", dpi=360)
    plt.show()

    return fAll, IVolEst, ISurfEst, ITotEst

if __name__ == '__main__':
    frequencies = [2.0, 3.0, 4.0, 5.0, 6.0]
    assemble_nn(fAll=frequencies, sim_dir='SaveSim', prefix='NewFreq')
