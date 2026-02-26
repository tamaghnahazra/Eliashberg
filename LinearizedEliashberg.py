from triqs.gf import *
import gc
from triqs.plot.mpl_interface import *
from triqs.lattice import *
from triqs.gf.tools import *    
from triqs_tprf.symmetries import *
import datetime

from triqs_tprf.lattice import lattice_dyson_g0_wk, lattice_dyson_g_wk, chi_wr_from_chi_wk, chi_tr_from_chi_wr, fourier_wk_to_wr, fourier_wr_to_tr, fourier_tr_to_wr, fourier_wr_to_wk
from triqs.gf import MeshImFreq
from triqs_tprf.utilities import temperature_to_beta
from triqs.gf import Gf, MeshProduct
from triqs_tprf.eliashberg import solve_eliashberg
from triqs_tprf.symmetries import _invert_momentum
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
from scipy.linalg import eigh

import pyfftw
import functools
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.special import expit
pi = np.pi
sigma = np.array([np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])])

# find the symmetry characters of the k-space form factors
# excluding the possible nodal lines
# input rank-2 tensor of kFF(kx,ky)
# note that this just reports the sign of the symmetry operation excluding nodal lines, cannot return anything except 1 or -1
def character(kFF,nk):
    inv=np.mean([np.sign(kFF[nk-i,nk-j])*np.sign(kFF[i, j]) for i in range(1, kFF.shape[0]) for j in range(1,kFF.shape[1]) if i != j and i != nk-j and i!=nk/2-1 and j!=nk/2-1])
    sv=np.mean([np.sign(kFF[i, nk-j])*np.sign(kFF[i, j]) for i in range(1, kFF.shape[0]) for j in range(1,kFF.shape[1]) if i != j and i != nk-j and i!=nk/2-1 and j!=nk/2-1])
    sd=np.mean([np.sign(kFF[j,i])*np.sign(kFF[i, j]) for i in range(1, kFF.shape[0]) for j in range(1,kFF.shape[1]) if i != j and i != nk-j and i!=nk/2-1 and j!=nk/2-1])
    return [inv,sv,sd]

# weighted version of character function
# replaces sgn[kFF1]sgn[kFF2] with kFF1*kFF2/(({|kFF1|+|kFF2|)/2)^2
def character_weighted(kFF,nk):
    inv=np.mean([kFF[nk-i,nk-j]*kFF[i, j]/(((np.abs(kFF[nk-i,nk-j])+np.abs(kFF[i, j]))/2)**2) for i in range(1, kFF.shape[0]) for j in range(1,kFF.shape[1]) if i != j and i != nk-j and i!=nk/2-1 and j!=nk/2-1])
    sv=np.mean([kFF[i, nk-j]*kFF[i, j]/(((np.abs(kFF[i, nk-j])+np.abs(kFF[i, j]))/2)**2) for i in range(1, kFF.shape[0]) for j in range(1,kFF.shape[1]) if i != j and i != nk-j and i!=nk/2-1 and j!=nk/2-1])
    sd=np.mean([kFF[j,i]*kFF[i, j]/(((np.abs(kFF[j,i])+np.abs(kFF[i, j]))/2)**2) for i in range(1, kFF.shape[0]) for j in range(1,kFF.shape[1]) if i != j and i != nk-j and i!=nk/2-1 and j!=nk/2-1])
    return [inv,sv,sd]
# get the symmetry characters of all elements in Gamma
# input: list of LScomponents \phi_k groupled by LS irrep
def get_all_characters(Gamma, nk):
    characters = []
    for group in Gamma:
        group_characters = []
        for element in group:
            group_characters.append(character(element, nk))
        characters.append(group_characters)
    return characters

# get the maximum absolute values of all elements in Gamma
# input: list of LScomponents \phi_k groupled by LS irrep
def get_max_abs_values(Gamma):
    max_abs_values = []
    for group in Gamma:
        group_max_abs = []
        for element in group:
            group_max_abs.append(np.max(np.abs(element)))
        max_abs_values.append(group_max_abs)
    return max_abs_values

def get_SigmaLScomponents_1orb(matrix, isDelta=False):
    # For 1 orbital, basis is just Identity (scalar 1), Spin basis is sigma (Pauli matrices)
    # If norb > 2, we take the first 2x2 block (assuming first orbital's spin sector)
    norb = matrix.shape[-1]
    components = np.zeros(shape=(*matrix.shape[:2], 1, 4), dtype=complex)
    
    # Project the first 2x2 spin sector
    matrix_2x2 = matrix[..., :2, :2]
    
    for j in range(4):
        mat = sigma[j]
        if isDelta:
            # Library uses: kron(GellMann, 1j*sigma[j]@sigma[2])
            # For us: 1 * 1j*sigma[j]@sigma[2]
            mat = 1j * sigma[j] @ sigma[2]
        
        # Project: Tr(M * basis) / Tr(basis * basis) -> Tr(M * basis) / 2
        components[:,:,0,j] = np.einsum('wkab,ba->wk', matrix_2x2, mat) / 2.0
    if isDelta:
        components[:,:,:,0] *= -1
        components[:,:,:,2] *= -1
        
    return components

def get_Gamma_1orb(LScomponents):
    # Singlet (Spin=0), Triplet (Spin=1,2,3)
    singlet = [LScomponents[:,:,0,0]]
    triplet = [LScomponents[:,:,0,1], LScomponents[:,:,0,2], LScomponents[:,:,0,3]]
    return singlet, triplet, [], []

def prep_for_plot_SigmaRe_1orb(sigma_wk, nk, norb):
    # Avoid .data on ndarrays (memoryview issue), use it for TRIQS objects
    if hasattr(sigma_wk, 'data') and not isinstance(sigma_wk, np.ndarray):
        data = np.asarray(sigma_wk.data)
    else:
        data = np.asarray(sigma_wk)

    # Use first frequency index
    SigmaRe = data[0].reshape(nk, nk, norb, norb)
    
    Gamma = get_Gamma_1orb(get_SigmaLScomponents_1orb(SigmaRe))
    return Gamma

def prep_for_plot_Delta_1orb(vs, nk, norb, oddfreq=False, n_w=None):
    if hasattr(vs, 'data') and not isinstance(vs, np.ndarray):
        data = np.asarray(vs.data)
    else:
        data = np.asarray(vs)
        
    if data.ndim == 1:
        # If 1D, it's likely an eigenvector from solve_linearized_gap_dynamic
        if n_w is None:
             raise ValueError("n_w must be provided to reshape 1D dynamic eigenvector.")
        data = data.reshape(2 * n_w, nk, nk, norb, norb)
    elif data.ndim == 4:
        data = data.reshape(data.shape[0], nk, nk, norb, norb)
        
    nw_idx = data.shape[0] // 2
    v0 = data[nw_idx] # first positive
    vm1 = data[nw_idx-1] # first negative
    
    if oddfreq:
        Delta = (v0 - vm1)/2
    else:
        Delta = (v0 + vm1)/2
        
    # Max normalize
    max_val = data.flatten()[np.argmax(np.abs(data))]
    if np.abs(max_val) > 1e-10:
        Delta /= max_val
        
    Gamma = get_Gamma_1orb(get_SigmaLScomponents_1orb(Delta, isDelta=True))
    return Gamma

def plot_Gamma_1orb(Gamma, nk, uniform_colorbar=False, round_character=True):
    # Simplified plotting for 1-orbital
    # Gamma is [[s], [t_x, t_y, t_z], [], []]
    
    labels = ["Singlet", "Triplet X", "Triplet Y", "Triplet Z"]
    all_comps = Gamma[0] + Gamma[1] # Singlet and triplet groups
    
    # Calculate max abs values for all components
    max_abs_values = [np.max(np.abs(c)) if c.size > 0 else 0.0 for c in all_comps]
    global_max = max(max_abs_values + [1e-10])
    threshold = global_max / 100
    
    # Identify indices above threshold (sorted by magnitude)
    indices = []
    for i, val in enumerate(max_abs_values):
        if val >= threshold:
            indices.append(i)
    indices.sort(key=lambda x: max_abs_values[x], reverse=True)
    
    print(f"Dominant components (index, title, max_val):")
    for idx in indices:
        print(f"  {idx}: {labels[idx]} - {max_abs_values[idx]}")

    if not indices:
        print("No dominant components found to plot.")
        return

    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    
    for i, idx in enumerate(indices):
        ax = axes[0, i]
        comp = all_comps[idx]
        im = ax.imshow(comp.real, origin='lower', cmap='RdBu', interpolation='nearest')
        ax.set_title(labels[idx])
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
# ===========================================================================
# D4h Character Table and Irrep Projection
# ===========================================================================

# Mapping of the 16 operations implemented in code to the 10 D4h classes:
_D4H_CLASSES = {
    'identity':     'E',
    'C4z_anti':     '2C4',
    'C4z_clock':    '2C4',
    'C2z':          'C2',
    'C2x':          '2Cp2',
    'C2y':          '2Cp2',
    'C2_xplusy':    '2Cpp2',
    'C2_xminusy':   '2Cpp2',
    'inversion':    'i',
    'S4':           '2S4',
    'S4_inv':       '2S4',
    'sigma_h':      'sh',
    'sigma_x':      '2sv',
    'sigma_y':      '2sv',
    'sigma_d':      '2sd',
    'sigma_dp':     '2sd'
}

_D4H_CHAR_VALS = {
    # Irrep: {Class: Character}
    'A1g': {'E': 1, '2C4': 1, 'C2': 1, '2Cp2': 1, '2Cpp2': 1, 'i': 1, '2S4': 1, 'sh': 1, '2sv': 1, '2sd': 1},
    'A2g': {'E': 1, '2C4': 1, 'C2': 1, '2Cp2':-1, '2Cpp2':-1, 'i': 1, '2S4': 1, 'sh': 1, '2sv':-1, '2sd':-1},
    'B1g': {'E': 1, '2C4':-1, 'C2': 1, '2Cp2': 1, '2Cpp2':-1, 'i': 1, '2S4':-1, 'sh': 1, '2sv': 1, '2sd':-1},
    'B2g': {'E': 1, '2C4':-1, 'C2': 1, '2Cp2':-1, '2Cpp2': 1, 'i': 1, '2S4':-1, 'sh': 1, '2sv':-1, '2sd': 1},
    'Eg':  {'E': 2, '2C4': 0, 'C2':-2, '2Cp2': 0, '2Cpp2': 0, 'i': 2, '2S4': 0, 'sh':-2, '2sv': 0, '2sd': 0},
    'A1u': {'E': 1, '2C4': 1, 'C2': 1, '2Cp2': 1, '2Cpp2': 1, 'i':-1, '2S4':-1, 'sh':-1, '2sv':-1, '2sd':-1},
    'A2u': {'E': 1, '2C4': 1, 'C2': 1, '2Cp2':-1, '2Cpp2':-1, 'i':-1, '2S4':-1, 'sh':-1, '2sv': 1, '2sd': 1},
    'B1u': {'E': 1, '2C4':-1, 'C2': 1, '2Cp2': 1, '2Cpp2':-1, 'i':-1, '2S4': 1, 'sh':-1, '2sv':-1, '2sd': 1},
    'B2u': {'E': 1, '2C4':-1, 'C2': 1, '2Cp2':-1, '2Cpp2': 1, 'i':-1, '2S4': 1, 'sh':-1, '2sv': 1, '2sd':-1},
    'Eu':  {'E': 2, '2C4': 0, 'C2':-2, '2Cp2': 0, '2Cpp2': 0, 'i':-2, '2S4': 0, 'sh': 2, '2sv': 0, '2sd': 0},
}

# Global D4h character table for the 16 operations implemented
D4H_CHARACTER_TABLE = {
    irrep: {op: vals[cls] for op, cls in _D4H_CLASSES.items()}
    for irrep, vals in _D4H_CHAR_VALS.items()
}

def project_delta(psi, target_irrep, character_table=D4H_CHARACTER_TABLE, apply_symmetry=None):
    """
    Projects a wavefunction psi(kx, ky, alpha, beta) into a specific Irrep of D4h.

    Parameters:
    - psi: 4D numpy array (kx, ky, norb, norb)
    - target_irrep: String key for the desired irrep (e.g., 'A1g', 'B1g', 'Eg')
    - character_table: Dict {irrep_name: {op_name: character}}
    - apply_symmetry: Function(op_name, psi) -> transformed_psi

    Returns:
    - psi_proj: The projected wavefunction array.
    """
    if target_irrep not in character_table:
        raise ValueError(f"Irrep {target_irrep} not found in character table.")
    
    if apply_symmetry is None:
        raise ValueError("apply_symmetry function must be provided (e.g., solver.apply_symmetry).")

    irrep_chars = character_table[target_irrep]
    
    # 1. Determine Group Constants
    h = len(irrep_chars) # Order of the group (16 for D4h)
    
    # Dimension of the irrep is the character of the Identity operation ('identity')
    d_gamma = int(np.real(irrep_chars.get('identity', 1)))
    
    # 2. Initialize the accumulator (ensure complex type)
    psi_proj = np.zeros_like(psi, dtype=np.complex128)
    
    # 3. Iterate over symmetry operations and sum
    for op_name, chi in irrep_chars.items():
        # Apply symmetry operation R to psi: R |psi>
        psi_transformed = apply_symmetry(op_name, psi)
        
        # Accumulate: chi*(R) * R |psi>
        psi_proj += np.conj(chi) * psi_transformed

    # 4. Apply Normalization Factor (d_gamma / h)
    psi_proj *= (d_gamma / h)
    
    return psi_proj


class EliashbergSolver:
    """
    A general Eliashberg solver class.
    Derived from EliashbergSolverSRO but made more general by checking
    norb and allowing overloading of calculate_hk and calculate_gamma.
    """
    def __init__(self, nk=12, n_w=24, T=700., mu=0., Xi=2.58, Q=2*pi*0.3, norb=6, Vertices=None, g=1.36/np.sqrt(3), taumeshfactor=6, tol=1e-3, eps=1., fixed_density=False, filling=2./3., tau=1./0.001):
        self.nk = nk # number of k points
        self.norb = norb # number of orbitals
        self.n_w = n_w # number of positive fermionic Matsubara frequencies

        self.taumeshfactor = taumeshfactor # how much bigger is the imaginary time mesh than the Matsubara frequency mesh
        self.tol = tol # tolerance for convergence
        self.eps = eps # small parameter for numerical stability, should be deprecated
        self.T = T # temperature in Kelvin
        self.mu = mu # chemical potential
        self.Xi = Xi # correlation length of the incipient order or mass of the bosons
        self.Q = Q # momentum of the incipient order
        self.Vertices = Vertices
        self.g = g

        self.filling = filling
        self.fixed_density=fixed_density
        self.tau = tau

    # Abstract function to calculate H(k)
    def calculate_hk(self, k1, k2, k3):
        raise NotImplementedError("Subclasses must implement calculate_hk")

    # Abstract function to calculate gamma_ph
    def calculate_gamma(self, wmesh_boson_kmesh, target_shape):
        raise NotImplementedError("Subclasses must implement calculate_gamma")

    # Abstract function to calculate static gamma_ph
    def calculate_gamma_static(self, kmesh, target_shape):
        raise NotImplementedError("Subclasses must implement calculate_gamma_static")

    def get_filling(self, gf_wk):
        return np.sum(np.array([gf_wk[:,k].density() for k in gf_wk.mesh[1]])).real/self.nk**2/self.norb

    def numbereqn(self, mu, e_k, sigma_wk):
        gf_wk = lattice_dyson_g_wk(mu=mu, e_k=e_k, sigma_wk=sigma_wk) 
        return np.sum(np.array([gf_wk[:,k].total_density() for k in gf_wk.mesh[1]])).real/self.nk**2/self.norb - self.filling
    
    def fourier_wk_to_tr(self, gf_wk):
        gf_wr = fourier_wk_to_wr(gf_wk)
        gf_tr = fourier_wr_to_tr(gf_wr, nt=self.taumeshfactor*self.n_w+1) # this inserts a minus sign, AGD convention for fermion GfImTime
        return gf_tr

    def fourier_tr_to_wk(self, gf_tr):
        gf_wr = fourier_tr_to_wr(gf_tr, nw=self.n_w) # this inserts another minus sign, AGD convention for fermion GfImTime
        gf_wk = fourier_wr_to_wk(gf_wr)
        return gf_wk

    def fermion_antisymmetrize(self, Delta):
        inv_idx = self._inv_idx()
        Delta_reshaped = Delta.data.reshape(2 * self.n_w, self.nk, self.nk, self.norb, self.norb)
        inv_flip = Delta_reshaped[::-1, inv_idx, :, :, :][:, :, inv_idx, :, :].transpose(0, 1, 2, 4, 3)
        Delta.data[:] = 0.5 * (Delta_reshaped - inv_flip).reshape(2 * self.n_w, self.nk ** 2, self.norb, self.norb)
        return Delta

    def fermion_antisymmetrize_static(self, Delta_k):
        """$\Delta_{ab}(k) = -\Delta_{ba}(-k)$."""
        inv_idx = self._inv_idx()
        # Delta_k shape is (nk, nk, norb, norb)
        inv_flip = Delta_k[inv_idx, :, :, :][:, inv_idx, :, :].transpose(0, 1, 3, 2)
        return 0.5 * (Delta_k - inv_flip)

    # -----------------------------------------------------------------------
    # Generic symmetry-analysis framework
    # -----------------------------------------------------------------------
    def _k_transforms(self):
        """Dictionary of {op_name: transform_func} for k-space mappings.
        Subclasses should implement this.
        """
        return {}

    def apply_symmetry(self, op, Delta_k):
        """Transform gap by named operation: Δ_new(k) = U @ Δ(k_trans) @ U_T"""
        k_trans = self._k_transforms()
        if op not in k_trans:
            raise ValueError(f"Operation '{op}' not supported by {self.__class__.__name__}")
        
        U_func = getattr(self, f"_U_{op}", None)
        if U_func is None:
            raise AttributeError(f"Subclass must implement _U_{op}()")
        
        U = U_func()
        Delta_trans = k_trans[op](Delta_k)
        
        # Apply unitary: U @ Delta @ U_T
        # Delta_trans shape: (nk, nk, norb, norb)
        # U shape: (norb, norb)
        return np.einsum('ab,xybc,cd->xyad', U, Delta_trans, U.T)

    def _symmetry_character(self, Delta_orig, Delta_trans, title, figsize=None, threshold=1e-3, to_plot=True):
        """Plot the ratio Δ_orig / Δ_trans for each component if to_plot is True,
        else return the global average character on valid k-points."""
        nk = self.nk
        
        # Small epsilon to avoid division by zero
        eps = 1e-15
        
        # Find points where original gap is significant
        mask = np.abs(Delta_orig) > threshold * np.max(np.abs(Delta_orig))
        
        if not np.any(mask):
            return 0.0
            
        # Character(k) = Tr(Delta_orig(k)^\dagger @ Delta_trans(k)) / |Delta_orig(k)|^2
        num = np.sum(np.conj(Delta_orig) * Delta_trans, axis=(-2, -1))
        den = np.sum(np.conj(Delta_orig) * Delta_orig, axis=(-2, -1))
        
        char_k = np.real(num / (den + eps))
        
        # Filter masked points for calculation
        char_avg = np.mean(char_k[mask[:,:,0,0]]) if mask.ndim > 2 else np.mean(char_k[mask])

        if to_plot:
            fig, ax = plt.subplots(figsize=figsize or (6, 5))
            im = ax.imshow(char_k, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='RdBu_r', vmin=-1.1, vmax=1.1)
            ax.set_title(f"{title}\nAverage: {char_avg:.3f}")
            ax.set_xlabel('$k_x$')
            ax.set_ylabel('$k_y$')
            plt.colorbar(im)
            plt.show()
            
        return char_avg

    def symmetry_character(self, op, eigvec=None, figsize=None, threshold=1e-3, to_plot=True):
        """Calculate or plot the character of a named symmetry operation."""
        Delta_k = self._prepare_gap_for_plotting(eigvec)
        Delta_trans = self.apply_symmetry(op, Delta_k)
        
        title = f"{op.replace('_', ' ').capitalize()} character"
        return self._symmetry_character(Delta_k, Delta_trans, title, figsize, threshold, to_plot=to_plot)

    def check_symmetries(self, eigvec=None, threshold=1e-3, verbose=True):
        """Check characters under all symmetries of the group.
        Default: vs_static[:, 0], then vs_dynamic[:, 0]."""
        Delta_k = self._prepare_gap_for_plotting(eigvec)
        
        ops = sorted(self._k_transforms().keys())
        characters = {}
        
        if verbose:
            print(f"{'Operation':<15} | Character")
            print("-" * 35)

        for op in ops:
            char = self.symmetry_character(op, eigvec=Delta_k, to_plot=False, threshold=threshold)
            characters[op] = char
            if verbose:
                print(f"{op:<15} | {char:>10.6f}")

        # Identify closest irrep from D4H_CHARACTER_TABLE
        best_irrep = "Unknown"
        max_overlap = -1.0
        
        for irrep, irrep_chars in D4H_CHARACTER_TABLE.items():
            # Character overlap score: 1/|G| * sum_g chi_obs(g) * (chi_irrep(g) / chi_irrep(E))
            # chi_irrep(E) is the dimension of the irrep (1 for A, B; 2 for E)
            dim = irrep_chars.get('identity', 1)
            overlap = 0.0
            valid_count = 0
            for op, val in characters.items():
                if op in irrep_chars:
                    overlap += val * (irrep_chars[op] / dim)
                    valid_count += 1
            
            if valid_count > 0:
                overlap /= valid_count
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_irrep = irrep
        
        if verbose:
            print("-" * 35)
            print(f"Closest Irrep: {best_irrep} (overlap: {max_overlap:.4f})")
                
        return characters, best_irrep

    # -----------------------------------------------------------------------
    # Generic symmetry-analysis utilities (no specific symmetry assumed)
    # -----------------------------------------------------------------------
    def _inv_idx(self):
        """Index map for k → -k on a periodic grid of size nk: i ↦ (nk - i) % nk."""
        return np.concatenate(([0], np.arange(self.nk - 1, 0, -1))).astype(int)

    def _prepare_gap_for_plotting(self, eigvec=None, oddfreq=False):
        """Normalise eigvec to ndarray of shape (nk, nk, norb, norb).
        Falls back to delta_wk, vs_static[:, 0], then vs_dynamic[:, 0].
        If dynamic gap, takes the first positive/negative Matsubara frequencies."""
        if eigvec is None:
            if hasattr(self, 'delta_wk') and self.delta_wk is not None:
                eigvec = self.delta_wk
            elif hasattr(self, 'vs_static'):
                eigvec = self.vs_static[:, 0]
            elif hasattr(self, 'vs_dynamic'):
                eigvec = self.vs_dynamic[:, 0]
            else:
                raise ValueError("No eigenvector or gap function found. Pass eigvec or run a solver first.")

        if hasattr(eigvec, 'data') and not isinstance(eigvec, np.ndarray):
            data = np.asarray(eigvec.data)
        else:
            data = np.asarray(eigvec)

        static_size = self.nk**2 * self.norb**2
        dynamic_size = 2 * self.n_w * static_size

        if data.size == dynamic_size:
            # Dynamic gap: shape (2*n_w, nk, nk, norb, norb)
            Delta = data.reshape(2 * self.n_w, self.nk, self.nk, self.norb, self.norb)
            if oddfreq:
                Deltak = (Delta[self.n_w] - Delta[self.n_w - 1]) / 2.0
            else:
                Deltak = (Delta[self.n_w] + Delta[self.n_w - 1]) / 2.0
            data = Deltak
        elif data.size == static_size:
            data = data.reshape(self.nk, self.nk, self.norb, self.norb)
        else:
            raise ValueError(f"Unexpected size: {data.size}. Expected {static_size} or {dynamic_size}")

        max_val = data.flat[np.abs(data).argmax()]
        if np.abs(max_val) > 1e-15:
            data /= max_val
            
        return data

    def _symmetry_character(self, Delta_orig, Delta_trans, title, figsize=None, threshold=1e-3, to_plot=True):
        """Plot the ratio Δ_orig / Δ_trans for each component if to_plot is True,
        else return the global average character on valid k-points."""
        n_grid = self.norb
        
        if to_plot:
            fig, axs = plt.subplots(n_grid, n_grid,
                                   figsize=(3*n_grid, 3*n_grid) if figsize is None else figsize,
                                   squeeze=False)
            fig.suptitle(title)
            
        sum_char = 0.0
        count_valid = 0

        for a in range(n_grid):
            for b in range(n_grid):
                D_orig  = Delta_orig [:, :, a, b]
                D_trans = Delta_trans[:, :, a, b]
                character = np.full((self.nk, self.nk), np.nan)
                d_max = np.max(np.abs(D_orig))
                
                if d_max > 1e-12:
                    mask  = np.abs(D_orig) > threshold * d_max
                    valid = mask & (np.abs(D_trans) > threshold * d_max)
                    
                    if np.any(valid):
                        char_vals = (D_orig[valid] / D_trans[valid]).real
                        sum_char += np.sum(char_vals)
                        count_valid += np.sum(valid)
                        
                        character[valid] = char_vals
                        character = np.clip(character, -2, 2)
                        
                if to_plot:
                    ax = axs[a, b]
                    im = ax.imshow(character, cmap='coolwarm', origin='lower',
                                  vmin=-1.5, vmax=1.5, interpolation='nearest')
                    if d_max > 1e-8:
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    if a == n_grid - 1: ax.set_xlabel('$k_x$')
                    if b == 0:          ax.set_ylabel('$k_y$')
                    ax.set_title(f'orb ({a},{b}) max={d_max:.1e}', fontsize=8)
                    
        if to_plot:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        return sum_char / count_valid if count_valid > 0 else np.nan


    def dyson_solver(self,seed_sigma=None,zero_Gtau0=True):
        h_k = Gf(mesh=MeshBrillouinZone(bz=BrillouinZone(BravaisLattice(units=np.eye(2))), n_k=self.nk), target_shape=(self.norb, self.norb))
        for k in h_k.mesh:
            kx, ky, kz = k
            h_k[k] = self.calculate_hk(kx, ky, kz)

        beta = temperature_to_beta(self.T)
        wmesh = MeshImFreq(beta=beta, S='Fermion', n_max=self.n_w)
        g0_wk = lattice_dyson_g0_wk(mu=self.mu, e_k=h_k, mesh=wmesh)
        wmesh_boson = MeshImFreq(beta=temperature_to_beta(self.T), S='Boson', n_max=self.n_w)
        wmesh_boson_kmesh = MeshProduct(wmesh_boson, g0_wk.mesh[1])
        
        # Call calculate_gamma to set self.gamma_ph and factorized parts
        self.calculate_gamma(wmesh_boson_kmesh, g0_wk.target_shape)
        
        # Factorized Dyson solver
        if hasattr(self, 'chi_wk') and hasattr(self, 'vertex_sum'):
            chi_wr = chi_wr_from_chi_wk(self.chi_wk)
            chi_tr = chi_tr_from_chi_wr(chi_wr, ntau=self.taumeshfactor*self.n_w+1)
            del chi_wr
            v_sum = self.vertex_sum
            
            def compute_sigma_tr(g_tr_data, chi_tr_data, v_sum, out_sigma_tr_data):
                # sigma_tr[a,d] = g^2 * chi_tr * v_sum[a,b,c,d] * g_tr[b,c]
                # chi_tr_data has 4 extra indices due to (1,1,1,1) shape
                chi_scalar = chi_tr_data[:, :, 0, 0, 0, 0]
                out_sigma_tr_data[:] = self.g**2 * np.einsum('tr,abcd,trbc->trad', chi_scalar, v_sum, g_tr_data)

        else:
            print("Warning: Factorized interaction not found. Falling back to gamma_ph if exists.")
            if not hasattr(self, 'gamma_ph'):
                 raise AttributeError("Interaction (gamma_ph or chi_wk) not found. Check calculate_gamma.")
            gamma_wr = chi_wr_from_chi_wk(self.gamma_ph)
            gamma_tr = chi_tr_from_chi_wr(gamma_wr, ntau=self.taumeshfactor*self.n_w+1)
            del gamma_wr
            def compute_sigma_tr(g_tr_data, gamma_tr_obj, v_sum, out_sigma_tr_data):
                out_sigma_tr_data[:] = np.einsum('trabcd,trbc->trad', gamma_tr_obj.data, g_tr_data)
            chi_tr = gamma_tr # hack to pass the object
            v_sum = None

        g_wk = g0_wk.copy()

        # Dyson loop iter #1
        g_tr = self.fourier_wk_to_tr(g_wk)
        
        sigma_tr = g_tr.copy()
        if seed_sigma is None:
            compute_sigma_tr(g_tr.data, chi_tr.data, v_sum, sigma_tr.data)
            if zero_Gtau0:
                sigma_tr.data[0] = 0.
            sigma_wk = self.fourier_tr_to_wk(sigma_tr)
        else:
            sigma_wk = g_wk.copy()
            sigma_wk.data[:] = seed_sigma.data[:]
            sigma_tr = self.fourier_wk_to_tr(sigma_wk)

        if(self.fixed_density):
            self.mu=root_scalar(self.numbereqn,args=(h_k,sigma_wk),bracket=[-self.bandwidth,self.bandwidth],method='brentq',xtol=1e-3).root
        
        g_wk = lattice_dyson_g_wk(mu=self.mu, e_k=h_k, sigma_wk=sigma_wk)
        residual = np.sum(np.abs(g0_wk.data - g_wk.data)) / self.nk**2 / self.norb
        
        # Dyson loop iters #2..
        while np.abs(residual) > self.tol:
            g_tr = self.fourier_wk_to_tr(g_wk)
            
            # New Sigma_tr calculation
            new_sigma_tr_data = np.zeros_like(sigma_tr.data)
            compute_sigma_tr(g_tr.data, chi_tr.data, v_sum, new_sigma_tr_data)

            sigma_tr.data[:] = (1. - self.eps) * sigma_tr.data + self.eps * new_sigma_tr_data
            
            if zero_Gtau0:
                sigma_tr.data[0] = 0.
            sigma_wk = self.fourier_tr_to_wk(sigma_tr)
            oldg_wk = g_wk.copy()
            
            if(self.fixed_density):
                self.mu=root_scalar(self.numbereqn,args=(h_k,sigma_wk),bracket=[-self.bandwidth,self.bandwidth],method='brentq',xtol=1e-3).root
                print(f"mu={self.mu}")
            
            g_wk = lattice_dyson_g_wk(mu=self.mu, e_k=h_k, sigma_wk=sigma_wk)
            residual = np.sum(np.abs(oldg_wk.data - g_wk.data)) / self.nk**2 / self.norb
            print(residual)
            self.sigma_wk = sigma_wk
            del oldg_wk
            gc.collect()
        print("Converged normal state with residual = ", residual, " and final filling = ", self.get_filling(g_wk), " at ", datetime.datetime.now())
        print("mu=",self.mu)

        self.sigma_wk = sigma_wk
        self.g_wk = g_wk

        # if not self.dyson_only:
        #     gamma_pp = Gf(mesh=wmesh_boson_kmesh, target_shape=g0_wk.target_shape*2)
        #     gamma_pp.data[:] = -2 * np.transpose(gamma_ph.data, axes=(0, 1, 2, 5, 4, 3)) #gamma_pp_abcd(k,k',q=0) = gamma_ph_adcb(q=k-k') # \times -2 because there is a -1/2 in the definition of the eliashberg product as implemented in solve_eliashberg 
        #     self.En, self.vs = solve_eliashberg(gamma_pp, g_wk, symmetrize_fct=self.fermion_antisymmetrize, k=solncount)
        #     print("Pair eigenvalues: ", self.En, " at ", datetime.datetime.now())

    def twoparticle_GG(self, h_k=None, g_wk=None, add_linewidth=True):
        """
        Calculates S_{abcd}(k, iω) = G_{ac}(k, iω) G_{db}(-k, -iω).
        If add_linewidth=False: Analytical calculation from h_k (static).
        If add_linewidth=True: Numerical calculation from g_wk (dynamic).
        """
        beta = temperature_to_beta(self.T)
        kmesh = h_k.mesh if h_k is not None else g_wk.mesh[1]
        S_k = Gf(mesh=kmesh, target_shape=(self.norb, self.norb, self.norb, self.norb))
        
        if not add_linewidth:
            # Static limit analytical calculation
            e_k = np.zeros((self.nk**2, self.norb))
            u_k = np.zeros((self.nk**2, self.norb, self.norb), dtype=complex)
            
            for kid in range(self.nk**2):
                evals, evecs = eigh(h_k.data[kid])
                e_k[kid] = evals
                u_k[kid] = evecs
            
            inv_idx = self._inv_idx()
            e_k_reshaped = e_k.reshape(self.nk, self.nk, self.norb)
            u_k_reshaped = u_k.reshape(self.nk, self.nk, self.norb, self.norb)
            
            e_mk = e_k_reshaped[inv_idx, :, :][:, inv_idx, :].reshape(self.nk**2, self.norb)
            u_mk = u_k_reshaped[inv_idx, :, :, :][:, inv_idx, :, :].reshape(self.nk**2, self.norb, self.norb)
                
            # Form-factor F_{m,m'}(k) = [f(-ξ_{-k, m'}) - f(ξ_{k, m})] / (ξ_{k, m} + ξ_{-k, m'} - 2μ)
            E_sum = e_k[:, :, None] + e_mk[:, None, :] # (nk^2, norb, norb)
            xi_k = e_k - self.mu
            xi_mk = e_mk - self.mu
            
            # Vectorized F calculation
            diff = E_sum - 2 * self.mu
            # Avoid division by zero
            safe_diff = np.where(np.abs(diff) < 1e-10, 1e-10, diff)
            F = (expit(-beta * -xi_mk[:, None, :]) - expit(-beta * xi_k[:, :, None])) / safe_diff
            # Handle diff=0 case (limit is beta * f' (xi))
            F = np.where(np.abs(diff) < 1e-10, beta * np.exp(beta * xi_k[:, :, None]) / (np.exp(beta * xi_k[:, :, None]) + 1.)**2, F)
                            
            S_k.data[:] = np.einsum('kam,kpm,kbn,kqn,kmn->kapqb', u_k, u_k.conj(), u_mk, u_mk.conj(), F)
        else:
            # Dynamic calculation from Matsubara G
            g_data = g_wk.data # shape: (2*n_w, nk^2, norb, norb)
            inv_idx = self._inv_idx()
            
            g_reshaped = g_data.reshape(2 * self.n_w, self.nk, self.nk, self.norb, self.norb)
            g_reversed = g_reshaped[::-1, inv_idx, :, :, :][:, :, inv_idx, :, :].reshape(2 * self.n_w, self.nk**2, self.norb, self.norb)
            
            # S_{ab,pq} = (1/beta) * sum_w G_{ap}(w,k) * G_{bq}(-w,-k)
            # Since G(-w,-k) is effectively G_reversed
            S_k.data[:] = (1.0 / beta) * np.einsum('wkap,wkbq->kapqb', g_data, g_reversed)
    
        return S_k

    def perform_fft_xy(self, tensor, dir='FFTW_FORWARD'):
        # Create aligned arrays for input and output
        input_array = pyfftw.empty_aligned(tensor.shape, dtype='complex128')
        output_array = pyfftw.empty_aligned(tensor.shape, dtype='complex128')

        # Copy the data to the input array
        input_array[:] = tensor

        # Create FFTW object to perform FFT on the first two indices
        fft_object = pyfftw.FFTW(input_array, output_array, axes=(0, 1), direction=dir)

        # Execute the FFT
        fft_object()
        
        return output_array

    def convolveDSV(self, S_k, V_kp, delta_k_flat, project_to=None):
        delta_dat = delta_k_flat.reshape(self.nk, self.nk, self.norb, self.norb)
        S_dat = S_k.data.reshape(self.nk, self.nk, *S_k.data.shape[1:])
        V_dat = V_kp.data.reshape(self.nk, self.nk, *V_kp.data.shape[1:])
        deltaS_k = np.einsum('xypq,xyapqb->xyab', delta_dat, S_dat)

        deltaS_r = self.perform_fft_xy(deltaS_k, dir='FFTW_BACKWARD')
        V_r = self.perform_fft_xy(V_dat, dir='FFTW_BACKWARD')
        delta_out = self.perform_fft_xy(np.einsum('xyab,xycabd->xycd', deltaS_r, V_r), dir='FFTW_FORWARD')

        # fermion antisymmetrization
        delta_fixed = self.fermion_antisymmetrize_static(delta_out)

        if project_to is not None:
             delta_fixed = project_delta(delta_fixed, project_to, apply_symmetry=self.apply_symmetry)

        return delta_fixed.reshape(delta_k_flat.size)

    def solve_linearized_gap_static(self, add_linewidth=False, solncount=1, v0=None, tol=1e-8, project_to=None):
        h_k = Gf(mesh=MeshBrillouinZone(bz=BrillouinZone(BravaisLattice(units=np.eye(2))), n_k=self.nk), target_shape=(self.norb, self.norb))
        for k in h_k.mesh:
            kx, ky, kz = k
            h_k[k] = self.calculate_hk(kx, ky, kz)
            
        kmesh = h_k.mesh
        self.calculate_gamma_static(kmesh, h_k.target_shape)
        gamma_ph = self.gamma_ph_static
        
        # gamma_pp_abcd(k,k',q=0) = gamma_ph_abdc(q=k-k') , this doesn't match the dynamic solver, but matches the conventions in the convolveDSV and two-particle_GG above
        gamma_pp = Gf(mesh=kmesh, target_shape=h_k.target_shape*2)
        gamma_pp.data[:] = np.transpose(gamma_ph.data, axes=(0, 1, 2, 4, 3))
        
        if add_linewidth:
            beta = temperature_to_beta(self.T)
            wmesh = MeshImFreq(beta=beta, S='Fermion', n_max=self.n_w)
            sigma_w = GfImFreq(mesh=wmesh, target_shape=(self.norb, self.norb))
            for iO in wmesh:
                # Sigma(iwn) = (1/iwn) / (tau/|wn| + 1)
                iwn = iO.value
                val = (1.0 / iwn) / (self.tau / np.abs(iwn) + 1.0)
                sigma_w[iO] = val * np.eye(self.norb)
                
            g_wk = lattice_dyson_g_wk(mu=self.mu, e_k=h_k, sigma_w=sigma_w)
            S_k = self.twoparticle_GG(g_wk=g_wk, add_linewidth=True)
            self.g_wk = g_wk
            self.sigma_wk = sigma_w
        else:
            # beta = temperature_to_beta(self.T)
            # wmesh = MeshImFreq(beta=beta, S='Fermion', n_max=self.n_w)
            # g0_wk = lattice_dyson_g0_wk(mu=self.mu, e_k=h_k, mesh=wmesh)
            S_k = self.twoparticle_GG(h_k=h_k, add_linewidth=False)
            
        # Random initial guess
        if v0 is None:
            n_total = self.nk**2 * self.norb**2
            v0 = np.random.random(n_total) + 0j

        kernel_prod = functools.partial(self.convolveDSV, S_k, gamma_pp, project_to=project_to)
        linop = LinearOperator(matvec=kernel_prod, dtype=complex, shape=(v0.size, v0.size))
        Es, U = eigs(linop, k=solncount, which="LR", tol=tol, v0=v0)
        
        self.En_static = Es
        self.vs_static = U
        print("Pair eigenvalues: ", self.En_static, " at ", datetime.datetime.now())
        return Es, U

    def solve_linearized_gap_dynamic(self, solncount=1, v0=None, tol=1e-8, init_with_static=False, project_to=None):
        """
        Solve the full dynamical linearized gap equation with memory optimizations.
            Δ_{ab}(k,iω) = (1/βV) Σ_{k',iω'} V^pp_{aa''b''b}(k-k', iω-iω') ΔS_{a''b''}(k', iω')
        where:
            V^pp_{aa''b''b}(q,iΩ) = g² Σ_i σ^i_{aa''} χ_i(q,iΩ) (σ^i)^T_{b''b}
            ΔS_{a''b''}(k',iω') = G_{a''a'}(k',iω') Δ_{a'b'}(k',iω') G^T_{b'b''}(-k',-iω')

        Derivation:
            (σ^i)^T_{b''b} = σ^i_{bb''}, so V^pp_{aa''b''b} = g²χ Σ_i σ^i_{aa''} σ^i_{bb''}
            gamma_ph[a,a'',b,b''] = g²χ Σ_i σ^i_{aa''} σ^i_{bb''} = V^pp_{aa''b''b}
            => V_pp = transpose(gamma_ph, (0,1,2,3,5,4))  [swap last two orbital indices]

            G^T_{b'b''} = G_{b''b'}, so ΔS_{a''b''} = G_{a''a'} Δ_{a'b'} G_{b''b'}(-k,-ω)
            => einsum('wkac,wkcd,wkbd->wkab', G, Δ, G_mk_mw)

            Convolution in (τ,r): Δ_{ab} = V^pp_{a,a'',b'',b} ΔS_{a'',b''}
            V^pp at positions [t,r, a, a'', b'', b] = [t,r, 2, 3, 4, 5]
            Contract positions 3,4 (a'',b'') with ΔS. Output at 2,5 (a,b).
            => einsum('trabcd,trbc->trad', V_pp_tr, DeltaS_tr)
            [Same structure as Dyson: sigma_tr = einsum('trabcd,trbc->trad', gamma_tr, g_tr)]
        """
        if not hasattr(self, 'g_wk') or (not hasattr(self, 'gamma_ph') and not hasattr(self, 'chi_wk')):
            print("Prerequisites not found. Running dyson_solver()...")
            self.dyson_solver()
            gc.collect()

        g_wk = self.g_wk
        
        # Factorized interaction for GAP solver
        if hasattr(self, 'chi_wk') and hasattr(self, 'vertex_sum'):
            print("Using factorized interaction for GAP solver.")
            chi_wr = chi_wr_from_chi_wk(self.chi_wk)
            chi_tr = chi_tr_from_chi_wr(chi_wr, ntau=self.taumeshfactor*self.n_w+1)
            del chi_wr
            
            # V^pp_{aa''b''b} = gamma_ph_{aa''bb''} = chi * vertex_sum_{aa''bb''}
            # particle-particle vertex transformation: M^pp_{aa''b''b} = vertex_sum_{aa''bb''}
            v_sum = self.vertex_sum
            v_pp_sum = np.transpose(v_sum, axes=(0, 1, 3, 2))
            
            def apply_interaction(deltaS_tr_data):
                # result[a,d] = g^2 * chi_tr * v_pp_sum[a,b,c,d] * deltaS_tr[b,c]
                chi_scalar = chi_tr.data[:, :, 0, 0, 0, 0]
                return self.g**2 * np.einsum('tr,abcd,trbc->trad', chi_scalar, v_pp_sum, deltaS_tr_data)

        else:
            print("Warning: Factorized interaction not found. Using full 6D tensor if exists.")
            if not hasattr(self, 'gamma_ph'):
                 raise AttributeError("Interaction (gamma_ph or chi_wk) not found.")
            gamma_ph = self.gamma_ph
            # Construct V^pp from gamma_ph: swap last two orbital indices
            V_pp = gamma_ph.copy()
            V_pp.data[:] = np.transpose(gamma_ph.data, axes=(0, 1, 2, 3, 5, 4))
            V_pp_wr = chi_wr_from_chi_wk(V_pp)
            del V_pp
            V_pp_tr = chi_tr_from_chi_wr(V_pp_wr, ntau=self.taumeshfactor * self.n_w + 1)
            del V_pp_wr
            gc.collect()
            
            def apply_interaction(deltaS_tr_data):
                return np.einsum('trabcd,trbc->trad', V_pp_tr.data, deltaS_tr_data)

        # Precompute -k indices
        kmesh = g_wk.mesh[1]
        minusk_indices = np.array([
            np.ravel_multi_index(
                _invert_momentum(np.unravel_index(kid, kmesh.dims), kmesh.dims),
                kmesh.dims
            )
            for kid in range(self.nk**2)
        ])

        # G(-k, -iω): reverse freq axis and apply -k index mapping
        g_data = g_wk.data  # shape: (2*n_w, nk^2, norb, norb)
        g_mk_mw = g_data[::-1][:, minusk_indices]  # G(-k, -iω)

        # Workspace GFs to avoid repeated allocations in kernel_action
        ws_wk = g_wk.copy()
        
        def kernel_action(delta_flat):
            # 1. Multiply by Green's functions
            delta_data = delta_flat.reshape(2 * self.n_w, self.nk**2, self.norb, self.norb)

            # ΔS_{a''b''} = G_{a''a'} Δ_{a'b'} G_{b''b'}(-k,-ω)
            deltaS_data = np.einsum('wkac,wkcd,wkbd->wkab', g_data, delta_data, g_mk_mw)

            # 2. FFT to (τ,r)
            ws_wk.data[:] = deltaS_data
            deltaS_tr = self.fourier_wk_to_tr(ws_wk)

            # 3. Vertex convolution
            res_tr_data = apply_interaction(deltaS_tr.data)
            
            # Update deltaS_tr in-place
            deltaS_tr.data[:] = res_tr_data
            del res_tr_data

            # 4. IFFT back to (w,k)
            result_wk = self.fourier_tr_to_wk(deltaS_tr)
            del deltaS_tr

            # 5. Antisymmetrize
            result_wk = self.fermion_antisymmetrize(result_wk)

            if project_to is not None:
                for i in range(2 * self.n_w):
                    delta_k = result_wk.data[i].reshape(self.nk, self.nk, self.norb, self.norb)
                    delta_proj = project_delta(delta_k, project_to, apply_symmetry=self.apply_symmetry)
                    result_wk.data[i] = delta_proj.reshape(self.nk**2, self.norb, self.norb)

            out = result_wk.data.reshape(delta_flat.size).copy()
            del result_wk
            return out

        # Initial guess
        if v0 is None:
            n_total = 2 * self.n_w * self.nk**2 * self.norb * self.norb
            if init_with_static:
                if not hasattr(self, 'vs_static'):
                    print("Solving static gap equation for initialization...")
                    self.solve_linearized_gap_static()
                
                stride = self.nk**2 * self.norb**2
                v0 = np.zeros(n_total, dtype=complex)
                iw = np.arange(-self.n_w + 0.5, self.n_w + 0.5) * 2 * np.pi / temperature_to_beta(self.T)
                for n, w in enumerate(iw):
                    v0[n*stride:(n+1)*stride] = (1.0 / w**2) * self.vs_static[:, 0]
            else:
                v0 = np.random.random(n_total) + 0j

        linop = LinearOperator(matvec=kernel_action, dtype=complex, shape=(v0.size, v0.size))
        Es, U = eigs(linop, k=solncount, which="LR", tol=tol, v0=v0)

        self.En_dynamic = Es
        self.vs_dynamic = U
        
        # Final cleanup
        if 'V_pp_tr' in locals(): del V_pp_tr
        if 'chi_tr' in locals(): del chi_tr
        gc.collect()
        
        print("Dynamic pair eigenvalues: ", self.En_dynamic, " at ", datetime.datetime.now())
        return Es, U
    def nonlinear_dynamic_gap_solver(self, seed_delta=None, seed_sigma=None, iterations=500, zero_Gtau0=True, tol=1e-5, project_to=None):
        """
        Solves for the Nambu-space self-energy self-consistently:
        hat{Sigma}(k) = (g^2 / beta V) sum_{k',i} hat{V}^i G_Nambu(k') hat{V}^i chi_i(k-k')
        where G_Nambu(k) = [iwn + mu*sigma3 - hat{H}(k) - hat{Sigma}(k)]^-1
        """
        print(f"Starting nonlinear_dynamic_gap_solver at {datetime.datetime.now()}", flush=True)
        
        if  self.fixed_density:
            print("Warning: fixed_density is True, but this solver does not support it yet.")
            
        # 1. Mesh and Symmetry setup
        beta = temperature_to_beta(self.T)
        wmesh = MeshImFreq(beta=beta, S='Fermion', n_max=self.n_w)
        kmesh = MeshBrillouinZone(bz=BrillouinZone(BravaisLattice(units=np.eye(2))), n_k=self.nk)
        wmesh_boson = MeshImFreq(beta=beta, S='Boson', n_max=self.n_w)
        wmesh_boson_kmesh = MeshProduct(wmesh_boson, kmesh)
        
        # Precompute -k indices
        minusk_indices = np.array([
            np.ravel_multi_index(
                _invert_momentum(np.unravel_index(kid, (self.nk, self.nk)), (self.nk, self.nk)),
                (self.nk, self.nk)
            )
            for kid in range(self.nk**2)
        ])
        
        # 2. Hamiltonian in Nambu Space: [[h(k), 0], [0, -h(-k)^T]]
        h_k_linear = np.zeros((self.nk**2, self.norb, self.norb), dtype=complex)
        for kid, k in enumerate(kmesh):
            h_k_linear[kid] = self.calculate_hk(*k)
            
        h_mk_T = h_k_linear[minusk_indices].transpose(0, 2, 1)
        
        H_Nambu = np.zeros((self.nk**2, 2*self.norb, 2*self.norb), dtype=complex)
        H_Nambu[:, :self.norb, :self.norb] = h_k_linear
        H_Nambu[:, self.norb:, self.norb:] = -h_mk_T
        
        # 3. Interaction in Nambu Space
        self.calculate_gamma(wmesh_boson_kmesh, (self.norb, self.norb))
        
        # We need chi_scalar and v_sum_nambu. 
        # If calculate_gamma didn't set self.chi_wk, we use self.gamma_ph
        if hasattr(self, 'chi_wk') and self.chi_wk is not None:
            chi_wr = chi_wr_from_chi_wk(self.chi_wk)
            chi_tr = chi_tr_from_chi_wr(chi_wr, ntau=self.taumeshfactor*self.n_w+1)
            # Handle potential shape difference (scalar vs (1,1,1,1))
            if len(chi_tr.data.shape) > 2:
                chi_scalar = chi_tr.data[:, :, 0, 0, 0, 0]
            else:
                chi_scalar = chi_tr.data
        else:
            # Fallback: recompute chi locally to be safe
            chi_wk_local = Gf(mesh=wmesh_boson_kmesh, target_shape=(1, 1, 1, 1))
            Qx, Qy = np.pi, np.pi
            for iO, k in wmesh_boson_kmesh:
                kx, ky, kz = k
                chi_wk_local[iO,k][0,0,0,0] = self.Chi0/(1/self.Xi**2 + 2 * (2- np.cos(kx - Qx) - np.cos(ky - Qy)) + np.abs(iO.value) * self.gamma * (1+ (np.abs(iO.value))/(self.bandwidth)))
            chi_wr = chi_wr_from_chi_wk(chi_wk_local)
            chi_tr = chi_tr_from_chi_wr(chi_wr, ntau=self.taumeshfactor*self.n_w+1)
            chi_scalar = chi_tr.data[:, :, 0, 0, 0, 0]
            
        if self.Vertices is not None:
            nambu_vertices = []
            for v in self.Vertices:
                hat_V = np.zeros((2*self.norb, 2*self.norb), dtype=complex)
                hat_V[:self.norb, :self.norb] = v
                hat_V[self.norb:, self.norb:] = -v.T
                nambu_vertices.append(hat_V)
            
            v_sum_nambu = np.zeros((2*self.norb,)*4, dtype=complex)
            for v in nambu_vertices:
                v_sum_nambu += np.tensordot(v, v, axes=0)
        else:
            # Reconstruct from self.vertex_sum if available, else recompute
            v_sum = getattr(self, 'vertex_sum', None)
            if v_sum is None:
                v_sum = np.zeros((self.norb,)*4, dtype=complex)
                for v in self.Vertices:
                    v_sum += np.tensordot(v, v, axes=0)
            
            v_sum_nambu = np.zeros((2*self.norb,)*4, dtype=complex)
            v_sum_nambu[:self.norb, :self.norb, :self.norb, :self.norb] = v_sum
            v_sum_nambu[self.norb:, self.norb:, self.norb:, self.norb:] = np.transpose(v_sum, axes=(1,0,3,2))
            v_sum_nambu[:self.norb, :self.norb, self.norb:, self.norb:] = -np.transpose(v_sum, axes=(0,1,3,2))
            v_sum_nambu[self.norb:, self.norb:, :self.norb, :self.norb] = -np.transpose(v_sum, axes=(1,0,2,3))

        # 4. Initialize Sigma and Delta
        sigma_wk = Gf(mesh=MeshProduct(wmesh, kmesh), target_shape=(self.norb, self.norb))
        if seed_sigma is not None: 
            if hasattr(seed_sigma, 'data') and not isinstance(seed_sigma, np.ndarray):
                sigma_wk.data[:] = seed_sigma.data[:]
            else:
                sigma_wk.data[:] = np.asarray(seed_sigma).reshape(sigma_wk.data.shape)
        else:
            # Seed with random noise
            sigma_wk.data[:] = (np.random.random(sigma_wk.data.shape) - 0.5)
        
        delta_wk = Gf(mesh=MeshProduct(wmesh, kmesh), target_shape=(self.norb, self.norb))
        if seed_delta is not None: 
            if hasattr(seed_delta, 'data') and not isinstance(seed_delta, np.ndarray):
                delta_wk.data[:] = seed_delta.data[:]
            else:
                delta_wk.data[:] = np.asarray(seed_delta).reshape(delta_wk.data.shape)
        else:
            # Seed with random noise
            delta_wk.data[:] = (np.random.random(delta_wk.data.shape) - 0.5)
        
        # Helper GF for FFT
        sn_wk = Gf(mesh=MeshProduct(wmesh, kmesh), target_shape=(2*self.norb, 2*self.norb))
        mu_sigma3 = self.mu * np.diag([1]*self.norb + [-1]*self.norb)
        
        # 5. Iteration loop
        for it in range(iterations):
            # Construct Nambu Sigma: [[Sigma(k), Delta(k)], [Delta(k)^adj, -Sigma(-k)^T]]
            sn_data = np.zeros((2*self.n_w, self.nk**2, 2*self.norb, 2*self.norb), dtype=complex)
            s_data = sigma_wk.data
            d_data = delta_wk.data
            
            sn_data[:, :, :self.norb, :self.norb] = s_data
            sn_data[:, :, :self.norb, self.norb:] = d_data
            sn_data[:, :, self.norb:, :self.norb] = d_data.conj().transpose(0, 1, 3, 2)
            
            s_mk_T = s_data[::-1][:, minusk_indices].transpose(0, 1, 3, 2)
            sn_data[:, :, self.norb:, self.norb:] = -s_mk_T
            
            # Construct Nambu Ginv and Invert
            iw = np.array([complex(m.value) for m in wmesh], dtype=complex).reshape(2*self.n_w, 1, 1, 1)
            Ginv = (iw * np.eye(2*self.norb)).astype(complex) + mu_sigma3.reshape(1, 1, 2*self.norb, 2*self.norb) \
                   - H_Nambu.reshape(1, self.nk**2, 2*self.norb, 2*self.norb) - sn_data
            G_Nambu_data = np.linalg.inv(Ginv)
            
            # FFT to (tau, r)
            sn_wk.data[:] = G_Nambu_data
            g_tr = self.fourier_wk_to_tr(sn_wk)
            
            # Update Sigma in Time/Space: hat_Sigma_tr = g^2 * chi_tr * hat_V * G_tr * hat_V
            new_sn_tr_data = self.g**2 * np.einsum('tr,abcd,trbc->trad', chi_scalar, v_sum_nambu, g_tr.data)
            
            if zero_Gtau0:
                new_sn_tr_data[0] = 0.
            
            g_tr.data[:] = new_sn_tr_data
            new_sn_wk = self.fourier_tr_to_wk(g_tr)
            
            # Extract new Sigma and Delta
            new_sn_wk_data = new_sn_wk.data
            
            old_sigma = sigma_wk.data.copy()
            old_delta = delta_wk.data.copy()
            
            sigma_wk.data[:] = new_sn_wk_data[:, :, :self.norb, :self.norb]
            delta_wk.data[:] = new_sn_wk_data[:, :, :self.norb, self.norb:]
            
            # Antisymmetrize Delta
            delta_wk = self.fermion_antisymmetrize(delta_wk)
            
            if project_to is not None:
                for i in range(2 * self.n_w):
                    delta_k = delta_wk.data[i].reshape(self.nk, self.nk, self.norb, self.norb)
                    delta_proj = project_delta(delta_k, project_to, apply_symmetry=self.apply_symmetry)
                    delta_wk.data[i] = delta_proj.reshape(self.nk**2, self.norb, self.norb)
            
            resid_s = np.sum(np.abs(sigma_wk.data - old_sigma)) / sigma_wk.data.size
            resid_d = np.sum(np.abs(delta_wk.data - old_delta)) / delta_wk.data.size
            max_d = np.max(np.abs(delta_wk.data))
            print(f"  Iteration {it}: resid_sigma = {resid_s:.2e}, resid_delta = {resid_d:.2e}, max_delta = {max_d:.2e}", flush=True)
            
            if resid_s < tol and resid_d < tol:
                 print("Converged!", flush=True)
                 break
        
        self.sigma_wk = sigma_wk
        self.delta_wk = delta_wk
        
        h_k = Gf(mesh=kmesh, target_shape=(self.norb, self.norb))
        h_k.data[:] = h_k_linear.reshape(h_k.data.shape)
        self.g_wk = lattice_dyson_g_wk(mu=self.mu, e_k=h_k, sigma_wk=sigma_wk)
        print(f"Completed nonlinear_dynamic_gap_solver at {datetime.datetime.now()}", flush=True)


# implements a minimal model of a square lattice with two orbitals, no SOC
class SquareLattice(EliashbergSolver):
    def __init__(self, nk=16, n_w=1024, g=2.0, T=100.0, mu=-1.0, filling=0.7, fixed_density=False, Vertices=sigma[1:], **kwargs):
        # 2-orbital spinful model (Pauli x,y,z only)
        
        # Susceptibility parameters 
        self.Chi0  = 1.0 
        self.Xi = 2.58
        self.gamma = 20.
        self.bandwidth = 8.0 
        
        super().__init__(
            nk=nk, 
            n_w=n_w, 
            T=T, 
            mu=mu, 
            norb=2, 
            Vertices=Vertices, 
            g=g,
            filling=filling,
            fixed_density=fixed_density,
            **kwargs
        )
        self.t = 1.0
        self.tp = -0.1 
    
    def calculate_hk(self, k1, k2, k3):
        # 1-band tight binding with spin identity
        # Construct identity locally since it's not in Vertices
        epsilon = -2 * self.t * (np.cos(k1) + np.cos(k2)) - 4 * self.tp * np.cos(k1) * np.cos(k2)
        return epsilon * np.eye(2)

    # -----------------------------------------------------------------------
    # Symmetry definitions for Square Lattice (D4h - 16 operations)
    # -----------------------------------------------------------------------
    def _k_transforms(self):
        inv = self._inv_idx()
        return {
            'identity':     lambda d: d,
            'inversion':    lambda d: d[inv][:, inv],
            'sigma_x':      lambda d: d[inv, :],
            'sigma_y':      lambda d: d[:, inv],
            'sigma_d':      lambda d: d.transpose(1, 0, 2, 3),                            # (x,y) -> (y,x)
            'sigma_dp':     lambda d: d[inv][:, inv].transpose(1, 0, 2, 3),               # (x,y) -> (-y,-x)
            'C2z':          lambda d: d[inv][:, inv],
            'C4z_anti':     lambda d: d[:, inv].transpose(1, 0, 2, 3),                    # (x,y) -> (-y,x) (anti)
            'C4z_clock':    lambda d: d[inv, :].transpose(1, 0, 2, 3),                    # (x,y) -> (y,-x) (clock)
            'S4':           lambda d: d[:, inv].transpose(1, 0, 2, 3),                    # (x,y) -> (-y,x) (anti spatially)
            'S4_inv':       lambda d: d[inv, :].transpose(1, 0, 2, 3),                    # (x,y) -> (y,-x) (clock spatially)
            'C2x':          lambda d: d[:, inv],                                          # (x,y) -> (x,-y)
            'C2y':          lambda d: d[inv, :],                                          # (x,y) -> (-x,y)
            'C2_xplusy':    lambda d: d.transpose(1, 0, 2, 3),                            # (x,y) -> (y,x)
            'C2_xminusy':   lambda d: d[inv][:, inv].transpose(1, 0, 2, 3),               # (x,y) -> (-y,-x)
            'sigma_h':      lambda d: d,
        }

    # Unitaries for SquareLattice (Spinful)
    def _U_identity(self):    return sigma[0]
    def _U_inversion(self):   return sigma[0]
    
    def _U_sigma_x(self):     return -1j * sigma[1]
    def _U_sigma_y(self):     return -1j * sigma[2]
    def _U_sigma_d(self):     return (-1j / np.sqrt(2.0)) * (sigma[1] - sigma[2])
    def _U_sigma_dp(self):    return (-1j / np.sqrt(2.0)) * (sigma[1] + sigma[2])
    
    def _U_sigma_h(self):     return -1j * sigma[3]
    def _U_C2z(self):         return -1j * sigma[3]
    
    def _U_C4z_anti(self):    return (sigma[0] - 1j * sigma[3]) / np.sqrt(2.0)
    def _U_C4z_clock(self):   return (sigma[0] + 1j * sigma[3]) / np.sqrt(2.0)
    
    def _U_S4(self):          return self._U_sigma_h() @ self._U_C4z_anti()
    def _U_S4_inv(self):      return self._U_sigma_h() @ self._U_C4z_clock()
    
    def _U_C2x(self):         return -1j * sigma[1]
    def _U_C2y(self):         return -1j * sigma[2]
    def _U_C2_xplusy(self):   return (-1j / np.sqrt(2.0)) * (sigma[1] + sigma[2])
    def _U_C2_xminusy(self):  return (-1j / np.sqrt(2.0)) * (sigma[1] - sigma[2])

    def calculate_gamma(self, wmesh_boson_kmesh, target_shape):
        """
        Calculate Gamma_ph using Paramagnon form-factor and Pauli vertices.
        """
        gamma_ph = Gf(mesh=wmesh_boson_kmesh, target_shape=target_shape*2)
        chi_wk = Gf(mesh=wmesh_boson_kmesh, target_shape=())
        
        Qx = np.pi
        Qy = np.pi
        
        for iO, k in wmesh_boson_kmesh:
            kx,ky,kz = k
            chi_wk[iO,k] += self.Chi0/(1/self.Xi**2 + 2 * (2- np.cos(kx - Qx) - np.cos(ky - Qy)) + np.abs(iO.value) * self.gamma * (1+ (np.abs(iO.value))/(self.bandwidth)))
            
        # Standard spin fluctuation interaction: sum over provided Pauli vertices
        vertex_sum = np.zeros(target_shape*2, dtype=complex)
        for v in self.Vertices:
             vertex_sum += np.tensordot(v, v, axes=0)

        gamma_ph.data[:] = self.g**2 * chi_wk.data[:, :, None, None, None, None] * vertex_sum
        
        print(f"Just loaded gamma_ph at {datetime.datetime.now()}")
        self.gamma_ph = gamma_ph
        return gamma_ph

    def calculate_gamma_static(self, kmesh, target_shape):
        """
        Calculate Static Gamma_ph using Paramagnon form-factor and Pauli vertices.
        """
        import datetime
        gamma_ph = Gf(mesh=kmesh, target_shape=target_shape*2)
        chi_k = Gf(mesh=kmesh, target_shape=())
        
        Qx = np.pi
        Qy = np.pi
        
        for k in kmesh:
            kx,ky,kz = k
            chi_k[k] += self.Chi0/(1/self.Xi**2 + 2 * (2- np.cos(kx - Qx) - np.cos(ky - Qy)))
            
        # Standard spin fluctuation interaction: sum over provided Pauli vertices
        vertex_sum = np.zeros(target_shape*2, dtype=complex)
        for v in self.Vertices:
             vertex_sum += np.tensordot(v, v, axes=0)

        gamma_ph.data[:] = self.g**2 * chi_k.data[:, None, None, None, None] * vertex_sum
        
        print(f"Just loaded static gamma_ph at {datetime.datetime.now()}")
        self.gamma_ph_static = gamma_ph
        return gamma_ph

    def plot_quasiparticle_weight(self, figsize=(8, 7)):
        """Plot the quasiparticle spectral weight Z = 1/(1 - Im Σ(iω₁)/ω₁).

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height).
        """
        if not hasattr(self, 'sigma_wk'):
            raise ValueError("No self-energy available. Run solver() first.")

        # Extract the self-energy at the first Matsubara frequency
        sigma_im = self.sigma_wk[0, 0](0, all).data.reshape(self.nk, self.nk).imag

        # Calculate the renormalization factor Z = 1/(1 - Im Σ(iω₁)/ω₁)
        # where ω₁ = π/β is the first Matsubara frequency
        beta = 1.0 / (8.617333262145e-5 * self.T)  # inverse temperature
        omega1 = pi / beta  # first fermionic Matsubara frequency (negative imaginary part)
        z_factor = 1.0 / (1.0 - sigma_im / omega1)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(z_factor, cmap='coolwarm', origin='lower')
        cbar = plt.colorbar(im, ax=ax)
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title("Quasiparticle Weight Z = 1/(1 - Im Σ(iω₁)/ω₁)")
        cbar.set_label('Z factor')

        return fig, ax

    def plot_delta(self,oddfreq=False):
        """Plot the gap Δ(k) projected into 1-orbital singlet/triplet components."""
        if hasattr(self, 'delta_wk') and self.delta_wk is not None:
            Gamma = prep_for_plot_Delta_1orb(self.delta_wk, self.nk, self.norb, oddfreq=oddfreq, n_w=self.n_w)
        elif hasattr(self, 'vs_dynamic'):
            Gamma = prep_for_plot_Delta_1orb(self.vs_dynamic[:,0], self.nk, self.norb, oddfreq=oddfreq, n_w=self.n_w)
        else:
            raise ValueError("No dynamic gap available. Run solve_linearized_gap_dynamic() or nonlinear_dynamic_gap_solver() first.")
        
        plot_Gamma_1orb(Gamma, self.nk)

    def plot_deltaMF(self, vs=None, idx=0):
        """Plot the static mean-field gap Δ(k)."""
        if vs is None:
            if not hasattr(self, 'vs_static'):
                raise ValueError("No static gap eigenvector available. Run solve_linearized_gap_static() first.")
            vs = self.vs_static[:, idx]

        Delta_flat = vs.reshape(self.nk, self.nk, self.norb, self.norb)
        max_abs_index = np.unravel_index(np.argmax(np.abs(Delta_flat)), Delta_flat.shape)
        if Delta_flat[max_abs_index] != 0:
            Delta_flat /= Delta_flat[max_abs_index]

        Gamma = get_Gamma_1orb(get_SigmaLScomponents_1orb(Delta_flat, isDelta=True))
        plot_Gamma_1orb(Gamma, self.nk)

    def plot_fermi_surface(self, figsize=(8, 7), nkmesh=None):
        """Plot the Fermi surface from the Green's function.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height).
        nkmesh : int, optional
            If provided, interpolate G on a finer nkmesh x nkmesh grid.
            Defaults to self.nk (no interpolation).
        """
        if not hasattr(self, 'g_wk'):
            raise ValueError("No Green's function available. Run solver() first.")

        # Extract G(k) at lowest Matsubara frequency
        if nkmesh is None or nkmesh == self.nk:
            # Coarse mesh (existing logic)
            gw1k = self.g_wk[0, 0](0, all).data.reshape(self.nk, self.nk)
            gwm1mk = self.g_wk[0, 0](-1, all).data.reshape(self.nk, self.nk)
    
            # Create inversion index
            inv_idx = np.concatenate(([0], np.arange(self.nk - 1, 0, -1))).astype(int)
    
            # Compute the product G(ω₀, k) * G(-ω₀, -k) which shows Fermi surface structure
            fs_data = (gw1k * gwm1mk[inv_idx, :][:, inv_idx]).real
            xlabel = 'kx (index)'
            ylabel = 'ky (index)'
            extent = None
            
        else:
            # Find mesh interpolation
            print(f"Interpolating Fermi surface on {nkmesh}x{nkmesh} grid...")
            k_grid = np.linspace(0, 2*np.pi, nkmesh, endpoint=False)
            fs_data = np.zeros((nkmesh, nkmesh))
            
            # Use slice then interpolate with list argument
            gw0_slice = self.g_wk(0, all)
            gw_neg_slice = self.g_wk(-1, all)

            for i, kx in enumerate(k_grid):
                for j, ky in enumerate(k_grid):
                    k_vec = [float(kx), float(ky), 0.0]
                    mk_vec = [float(-kx), float(-ky), 0.0]
                    
                    val_k = gw0_slice(k_vec) # Returns matrix, pass k as list
                    val_mk = gw_neg_slice(mk_vec) # Returns matrix
                    
                    # Extract 0,0 component if matrix
                    if hasattr(val_k, 'shape') and len(val_k.shape) >= 2:
                        vk = val_k[0, 0]
                        vmk = val_mk[0, 0]
                    else:
                        vk = val_k
                        vmk = val_mk
                        
                    val = vk * vmk
                    fs_data[i, j] = val.real
                    
            xlabel = 'kx'
            ylabel = 'ky'
            extent = [0, 2*np.pi, 0, 2*np.pi]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(fs_data, cmap='coolwarm', origin='lower', extent=extent)
        cbar = plt.colorbar(im, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Fermi Surface: G(ω₀,k) × G(-ω₀,-k)")
        cbar.set_label('Product')

        return fig, ax

    def plot_bandstructure(self, figsize=(8, 6)):
        """Plot the band structure along the high-symmetry path Γ-X-M-Γ."""
        # Define high-symmetry points for the square lattice
        pts = {'$\Gamma$': [0, 0], 'X': [np.pi, 0], 'M': [np.pi, np.pi]}
        path_labels = ['$\Gamma$', 'X', 'M', '$\Gamma$']
        nodes = np.array([pts[s] for s in path_labels])
        # Generate interpolated k-path (e.g., 100 points per segment)
        n_pts = 100
        k_path = np.vstack([np.linspace(nodes[i], nodes[i+1], n_pts) for i in range(len(nodes)-1)])
        # Compute eigenvalues at each k-point
        energies = np.array([np.linalg.eigvalsh(self.calculate_hk(kx, ky, 0)) for kx, ky in k_path])
        # Plotting
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(energies, color='tab:blue', alpha=0.7) 
        ax.set_xticks(np.linspace(0, len(k_path), len(path_labels)))
        ax.set_xticklabels(path_labels)
        ax.axvline(n_pts, color='k', linestyle=':', alpha=0.5)
        ax.axvline(2*n_pts, color='k', linestyle=':', alpha=0.5)
        ax.axhline(0, color='red', linestyle='--', alpha=0.3) # Fermi level at 0 if mu=0
        ax.set_ylabel("Energy (eV)")
        ax.set_title("Band Structure along $\Gamma-X-M-\Gamma$")
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
if __name__ == '__main__':
    print("--- Running SquareLattice Minimal Model Test ---")
    SqLat = SquareLattice(fixed_density=True, n_w=1024, filling=0.6, g=2.)
    
    print("\nSolving Dyson Equation...")
    SqLat.dyson_solver()
    
    print("\nSolving Linearized Gap Equation (Dynamic)...")
    SqLat.solve_linearized_gap_dynamic()
    
    print("\n--- Minimal Story ---")
    print(f"Run Parameters: fixed_density=True, n_w=1024, filling={SqLat.filling}, g={SqLat.g}")
    print(f"Leading Eigenvalues: {SqLat.En_dynamic}")
    
    print("\nLeading Eigenvector Symmetry Analysis:")
    SqLat.check_symmetries(verbose=True)

