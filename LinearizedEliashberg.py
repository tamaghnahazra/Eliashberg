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
    components = np.zeros(shape=(*matrix.shape[:2], 1, 4), dtype=complex)
    
    for j in range(4):
        mat = sigma[j]
        if isDelta:
            # Library uses: kron(GellMann, 1j*sigma[j]@sigma[2])
            # For us: 1 * 1j*sigma[j]@sigma[2]
            mat = 1j * sigma[j] @ sigma[2]
        
        # Project: Tr(M * basis) / Tr(basis * basis) -> Tr(M * basis) / 2
        components[:,:,0,j] = np.einsum('wkab,ba->wk', matrix, mat) / 2.0
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
    if hasattr(sigma_wk, 'data'):
        data = sigma_wk.data
    else:
        data = sigma_wk

    # Use first frequency index
    SigmaRe = data[0].reshape(nk, nk, 2, 2)
    
    Gamma = get_Gamma_1orb(get_SigmaLScomponents_1orb(SigmaRe))
    return Gamma

def prep_for_plot_Delta_1orb(vs, nk, norb, oddfreq=False):
    if hasattr(vs, 'data'):
        data = vs.data
    else:
        data = vs
        
    if data.ndim == 4:
        data = data.reshape(data.shape[0], nk, nk, 2, 2)
        
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


class EliashbergSolver:
    """
    A general Eliashberg solver class.
    Derived from EliashbergSolverSRO but made more general by checking
    norb and allowing overloading of calculate_hk and calculate_gamma.
    """
    def __init__(self, nk=12, n_w=24, T=700., mu=0., Xi=2.58, Q=2*pi*0.3, norb=6, Vertices=None, g=1.36/np.sqrt(3), taumeshfactor=6, tol=1e-3, eps=1., dyson_only=False, fixed_density=False, filling=2./3., tau=1./0.001):
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
        self.dyson_only = dyson_only
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
        inv_idx = np.concatenate(([0], np.arange(self.nk - 1, 0, -1))).astype(int)
        Delta_reshaped = Delta.data.reshape(2 * self.n_w, self.nk, self.nk, self.norb, self.norb)
        inv_flip = Delta_reshaped[::-1, inv_idx, :, :, :][:, :, inv_idx, :, :].transpose(0, 1, 2, 4, 3)
        Delta.data[:] = 0.5 * (Delta_reshaped - inv_flip).reshape(2 * self.n_w, self.nk ** 2, self.norb, self.norb)
        return Delta

    def solver(self,solncount=1,seed_sigma=None,zero_Gtau0=True):
        h_k = Gf(mesh=MeshBrillouinZone(bz=BrillouinZone(BravaisLattice(units=np.eye(2))), n_k=self.nk), target_shape=(self.norb, self.norb))
        for k in h_k.mesh:
            kx, ky, kz = k
            h_k[k] = self.calculate_hk(kx, ky, kz)

        beta = temperature_to_beta(self.T)
        wmesh = MeshImFreq(beta=beta, S='Fermion', n_max=self.n_w)
        g0_wk = lattice_dyson_g0_wk(mu=self.mu, e_k=h_k, mesh=wmesh)
        wmesh_boson = MeshImFreq(beta=temperature_to_beta(self.T), S='Boson', n_max=self.n_w)
        wmesh_boson_kmesh = MeshProduct(wmesh_boson, g0_wk.mesh[1])
        
        # Call calculate_gamma to set self.gamma_ph
        self.calculate_gamma(wmesh_boson_kmesh, g0_wk.target_shape)
        gamma_ph = self.gamma_ph # self.gamma_ph is set by calculate_gamma

        gamma_wr = chi_wr_from_chi_wk(gamma_ph)
        gamma_tr = chi_tr_from_chi_wr(gamma_wr, ntau=self.taumeshfactor*self.n_w+1)
        g_wk = g0_wk.copy()
        # Dyson loop iter #1
        g_tr = self.fourier_wk_to_tr(g_wk)
        if seed_sigma is None:
            sigma_tr = g_tr.copy()
            sigma_tr.data[:] = np.einsum('trabcd,trbc->trad', gamma_tr.data, g_tr.data)
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
            sigma_tr.data[:] = (1. - self.eps) * sigma_tr.data + self.eps * np.einsum('trabcd,trbc->trad', gamma_tr.data, g_tr.data)
            if zero_Gtau0:
                sigma_tr.data[0] = 0.
            sigma_wk = self.fourier_tr_to_wk(sigma_tr)
            oldg_wk = g_wk.copy()
            if(self.fixed_density):
                self.mu=root_scalar(self.numbereqn,args=(h_k,sigma_wk),bracket=[-self.bandwidth,self.bandwidth],method='brentq',xtol=1e-3).root
                print(f"mu={self.mu}")
            g_wk = lattice_dyson_g_wk(mu=self.mu if self.fixed_density else 0.0, e_k=h_k, sigma_wk=sigma_wk)
            residual = np.sum(np.abs(oldg_wk.data - g_wk.data)) / self.nk**2 / self.norb
            print(residual)
            self.sigma_wk = sigma_wk
            del oldg_wk
            gc.collect()
        print("Converged normal state with residual = ", residual, " and final filling = ", self.get_filling(g_wk), " at ", datetime.datetime.now())
        print("mu=",self.mu)

        self.sigma_wk = sigma_wk
        self.g_wk = g_wk
        self.gamma_ph = gamma_ph

        if not self.dyson_only:
            gamma_pp = Gf(mesh=wmesh_boson_kmesh, target_shape=g0_wk.target_shape*2)
            gamma_pp.data[:] = -2 * np.transpose(gamma_ph.data, axes=(0, 1, 2, 5, 4, 3)) #gamma_pp_abcd(k,k',q=0) = gamma_ph_adcb(q=k-k') # \times -2 because there is a -1/2 in the definition of the eliashberg product as implemented in solve_eliashberg 
            self.En, self.vs = solve_eliashberg(gamma_pp, g_wk, symmetrize_fct=self.fermion_antisymmetrize, k=solncount)
            print("Pair eigenvalues: ", self.En, " at ", datetime.datetime.now())

    def twoparticle_GG(self, h_k=None, g_wk=None, add_linewidth=True):
        if not add_linewidth:
            # Analytic calculation from h_k
            kmesh = h_k.mesh
            S_k = Gf(mesh=kmesh, target_shape=h_k.target_shape*2)
            
            e_k = np.zeros((self.nk**2, self.norb))
            u_k = np.zeros((self.nk**2, self.norb, self.norb), dtype=complex)
            
            for kid, k in enumerate(kmesh):
                evals, evecs = eigh(h_k.data[kid])
                e_k[kid] = evals
                u_k[kid] = evecs
            
            beta = temperature_to_beta(self.T)
            
            e_k_minusk = np.zeros_like(e_k)
            u_minusk = np.zeros_like(u_k)
            
            for kid, k in enumerate(kmesh):
                minuskid = np.ravel_multi_index(_invert_momentum(np.unravel_index(kid, kmesh.dims), kmesh.dims), kmesh.dims)
                e_k_minusk[kid] = e_k[minuskid]
                u_minusk[kid] = u_k[minuskid]
                
            E_sum = e_k[:, :, None] + e_k_minusk[:, None, :]
            F = np.zeros_like(E_sum)
            for kid in range(self.nk**2):
                for m in range(self.norb):
                    for mp in range(self.norb):
                        diff = E_sum[kid, m, mp] - 2 * self.mu
                        xi_k = e_k[kid, m] - self.mu
                        xi_mk = e_k_minusk[kid, mp] - self.mu
                        if np.abs(diff) < 1e-10:
                            num = beta * np.exp(beta * xi_k) / (np.exp(beta * xi_k) + 1.)**2
                            F[kid, m, mp] = num
                        else:
                            F[kid, m, mp] = (expit(-beta * -xi_mk) - expit(-beta * xi_k)) / diff
                            
            S_data = np.einsum('kam,kpm,kbn,kqn,kmn->kapqb', u_k, u_k.conj(), u_minusk, u_minusk.conj(), F)
            S_k.data[:] = S_data
            return S_k
        else:
            kmesh = g_wk.mesh[1]
            S_k = Gf(mesh=kmesh, target_shape=g_wk.target_shape*2)
            g_data = g_wk.data
            
            beta = temperature_to_beta(self.T)
            
            inv_idx = np.concatenate(([0], np.arange(self.nk - 1, 0, -1))).astype(int)
            g_reshaped = g_data.reshape(g_data.shape[0], self.nk, self.nk, self.norb, self.norb)
            g_inv_flip = g_reshaped[::-1, inv_idx, :, :, :][:, :, inv_idx, :, :]
            g_reversed_flat = g_inv_flip.reshape(g_data.shape[0], self.nk**2, self.norb, self.norb)
            
            S_data = (1.0 / beta) * np.einsum('wkap,wkbq->kapqb', g_data, g_reversed_flat)
            S_k.data[:] = S_data
        
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

    def convolveDSV(self, S_k, V_kp, delta_k_flat):
        delta_dat = delta_k_flat.reshape(self.nk, self.nk, self.norb, self.norb)
        S_dat = S_k.data.reshape(self.nk, self.nk, *S_k.data.shape[1:])
        V_dat = V_kp.data.reshape(self.nk, self.nk, *V_kp.data.shape[1:])
        deltaS_k = np.einsum('xypq,xyapqb->xyab', delta_dat, S_dat)

        deltaS_r = self.perform_fft_xy(deltaS_k, dir='FFTW_BACKWARD')
        V_r = self.perform_fft_xy(V_dat, dir='FFTW_BACKWARD')
        delta_out = self.perform_fft_xy(np.einsum('xyab,xycabd->xycd', deltaS_r, V_r), dir='FFTW_FORWARD')

        # anti-symmetrize and flatten
        delta_mk = np.roll(delta_out[::-1, ::-1], shift=1, axis=(0, 1))
        delta_fixed = (delta_out - np.transpose(delta_mk, axes=(0, 1, 3, 2))) / 2
        return delta_fixed.reshape(delta_k_flat.size)

    def solve_linearized_gap_static(self, add_linewidth=False, solncount=1, seed=None):
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
            
        np.random.seed(seed)
        delta_k = Gf(mesh=kmesh, target_shape=h_k.target_shape)
        delta_k.data[:] = np.random.random(h_k.data.shape[1:])

        kernel_prod = functools.partial(self.convolveDSV, S_k, gamma_pp)
        linop = LinearOperator(matvec=kernel_prod, dtype=complex, shape=(delta_k.data.size, delta_k.data.size))
        Es, U = eigs(linop, k=solncount, which="LR", tol=1e-8, v0=delta_k.data.reshape(delta_k.data.size))
        
        self.En_static = Es
        self.vs_static = U
        print("Pair eigenvalues: ", self.En_static, " at ", datetime.datetime.now())
        return Es, U

# implements a minimal model of a square lattice with two orbitals, no SOC
class SquareLattice(EliashbergSolver):
    def __init__(self, nk=16, n_w=1024, g=2.0, T=100.0, mu=-1.0, filling=0.7, fixed_density=False, **kwargs):
        # 2-orbital spinful model (Pauli x,y,z only)
        # Vertices passed are only the interaction vertices
        vertices = sigma[1:]
        
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
            Vertices=vertices, 
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

    def plot_delta(self):
        """Plot the gap Δ(k) using the 1-orbital plotting function.
        This extracts the singlet and triplet components and plots the dominant ones.
        """
        if not hasattr(self, 'vs'):
            raise ValueError("No gap eigenvector available. Run solver() first.")

        Delta = prep_for_plot_Delta_1orb(self.vs[0], self.nk, self.norb)
        plot_Gamma_1orb(Delta, self.nk)

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
    SqLat = SquareLattice(fixed_density=True, n_w=1024, filling=0.6, g=2.)
    SqLat.solver()
    SqLat.plot_bandstructure()
    SqLat.plot_fermi_surface(nkmesh=64)
    SqLat.plot_quasiparticle_weight()
    SqLat.plot_delta()

