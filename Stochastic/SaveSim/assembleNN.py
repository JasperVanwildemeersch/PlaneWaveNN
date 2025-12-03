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

    # load .mat
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    # MATLAB .mat often contains field names and meta-entries; we pick out the arrays we need
    return mat

def get_array(matdict, key):
    if key in matdict:
        arr = matdict[key]

        # unwrap MATLAB object arrays
        try:
            return np.asarray(arr)
        except Exception:
            pass
    # search inside nested structures (MATLAB sometimes nests fields)
    for k, v in matdict.items():
        if isinstance(v, dict) and key in v:
            try:
                return np.asarray(v[key])
            except Exception:
                pass

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
            # try without sim_dir (in case files are in cwd)
            fname_mat = f"{prefix}{f_str}Hz.mat"
            data = load_sim_file(fname_mat)

        try:
            uCavASDX = get_array(data, 'uCavASDX').astype(np.complex128)
            uCavASDY = get_array(data, 'uCavASDY').astype(np.complex128)
            uCavASDZ = get_array(data, 'uCavASDZ').astype(np.complex128)

            IVolTotAll = get_array(data, 'IVolTotAll').astype(np.complex128)
            ISurfTotAll = get_array(data, 'ISurfTotAll').astype(np.complex128)
        except KeyError as e:
            raise RuntimeError(f"Cannot find expected variable in {fname_mat}: {e}")

        # Ensure shapes
        def ensure_2d(arr):
            a = np.array(arr, copy=False)
            if a.ndim == 1:
                return a.reshape((-1, 1))  # (N,) -> (N,1)
            return a

        uCavASDX = np.atleast_1d(uCavASDX).astype(np.complex128).ravel()
        uCavASDY = np.atleast_1d(uCavASDY).astype(np.complex128).ravel()
        uCavASDZ = np.atleast_1d(uCavASDZ).astype(np.complex128).ravel()

        IVolTotAll = np.atleast_2d(IVolTotAll).astype(np.complex128)
        ISurfTotAll = np.atleast_2d(ISurfTotAll).astype(np.complex128)

        # Compute single-frequency cavity ASD scaling: sFASD = 1 / sqrt(mean(abs(uCavASD)^2))
        sFASDX = 1.0 / np.sqrt(np.mean(np.abs(uCavASDX)**2))
        sFASDY = 1.0 / np.sqrt(np.mean(np.abs(uCavASDY)**2))
        sFASDZ = 1.0 / np.sqrt(np.mean(np.abs(uCavASDZ)**2))

        rms_IVol = np.sqrt(np.mean(np.abs(IVolTotAll)**2, axis=0))  # length at least 3
        rms_ISurf = np.sqrt(np.mean(np.abs(ISurfTotAll)**2, axis=0))
        rms_ITot = np.sqrt(np.mean(np.abs(IVolTotAll - ISurfTotAll)**2, axis=0))
        # Multiply by G * rho * sFASD per component
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

#plot
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
    plt.show()

    return fAll, IVolEst, ISurfEst, ITotEst


if __name__ == '__main__':
    # example usage: adapt fAll to the files you have in SaveSim folder
    frequencies = [2.0, 4.0]
    assemble_nn(fAll=frequencies, sim_dir='SaveSim', prefix='NewFreq')
