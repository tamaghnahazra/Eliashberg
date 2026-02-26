from triqs.gf import *
from triqs.plot.mpl_interface import *
from triqs.lattice import *
from triqs.gf.tools import *    
import datetime

import numpy as np
import gc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from IPython.display import display, Math

from LinearizedEliashberg import EliashbergSolver, get_all_characters, get_max_abs_values, character, character_weighted, sigma, pi

GellMannMatrix = [None] * 9
#basis is yz, xz, xy
GellMannMatrix[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
GellMannMatrix[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
GellMannMatrix[2] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
GellMannMatrix[3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
GellMannMatrix[4] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, np.sqrt(2)]])
GellMannMatrix[5] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
GellMannMatrix[6] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
GellMannMatrix[7] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
GellMannMatrix[8] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
ell = np.array([np.eye(3),GellMannMatrix[8],-GellMannMatrix[6],GellMannMatrix[2]]) # defined to give spin and orbital the same commutation relations
L0sigma=np.array([
    np.kron(np.eye(3),sigma[0]),
    np.kron(np.eye(3),sigma[1]),
    np.kron(np.eye(3),sigma[2]),
    np.kron(np.eye(3),sigma[3])
])
Lsigma0=np.array([
    np.kron(ell[0],sigma[0]),
    np.kron(ell[1],sigma[0]),
    np.kron(ell[2],sigma[0]),
    np.kron(ell[3],sigma[0])
])
PalleVertices = [None,
    - 0.06 * np.kron(GellMannMatrix[4],sigma[1]) 
    + 0.88 * np.kron(GellMannMatrix[3],sigma[1]) 
    - 0.18 * np.kron(GellMannMatrix[1],sigma[2]) 
    - 0.31 * np.kron(GellMannMatrix[5],sigma[3]) 
    - 0.31 * np.kron(GellMannMatrix[8],sigma[0])
                    ,
    - 0.06 * np.kron(GellMannMatrix[4],sigma[2]) 
    - 0.88 * np.kron(GellMannMatrix[3],sigma[2]) 
    - 0.18 * np.kron(GellMannMatrix[1],sigma[1]) 
    - 0.31 * np.kron(GellMannMatrix[7],sigma[3]) 
    + 0.31 * np.kron(GellMannMatrix[6],sigma[0])
                    ,
    + 0.92 * np.kron(GellMannMatrix[0],sigma[3]) 
    - 0.07 * np.kron(GellMannMatrix[4],sigma[3]) 
    + 0.33 * (np.kron(GellMannMatrix[5],sigma[1]) + np.kron(GellMannMatrix[7],sigma[2]))
    + 0.20 * np.kron(GellMannMatrix[2],sigma[0])
]
# this is specific to the 6 orbitals in the t2g sector, as in Sr2RuO4
LocalSpinOrbital_list = [
    [r"$\mathbb{I}_{xz|yz}$",r"$\sqrt{2}\mathbb{I}_{xy}$",r"$L_{z}S_{z}$",r"$L_{x}S_{x}+L_{y}S_{y}$",r"$Q_{yz}S_{x}-Q_{xz}S_{y}$"],
    [r"$L_{z}$",r"$\mathbb{I}_{xz|yz}S_{z}$",r"$\sqrt{2}\mathbb{I}_{xy}S_{z}$",r"$L_{x}S_{y}-L_{y}S_{x}$",r"$Q_{yz}S_{y}+Q_{xz}S_{x}$"],
    [r"$-Q_{x^{2}-y^{2}}$",r"$L_{x}S_{x}-L_{y}S_{y}$",r"$Q_{yz}S_{x}+Q_{xz}S_{y}$",r"$Q_{xy}S_{z}$"],
    [r"$Q_{xy}$",r"$L_{x}S_{y}+L_{y}S_{x}$",r"$Q_{yz}S_{y}-Q_{xz}S_{x}$"]
    ]

Irreps = [
  r"$A_{1g}$",
  r"$A_{2g}$",
  r"$B_{1g}$",
  r"$B_{2g}$"
]

SpinOrbitalCharacter_lookup = {
    (1, 1, 1): r"$\phi_{A_{1g}}({\bf k})$",
    (1, -1, -1): r"$\phi_{A_{2g}}({\bf k})$",
    (1, 1, -1): r"$\phi_{B_{1g}}({\bf k})$",
    (1, -1, 1): r"$\phi_{B_{2g}}({\bf k})$"
}

SpinOrbitalCharacter = [[1,1,1],[1,-1,-1],[1,1,-1],[1,-1,1]]

product_table=[[0,1,2,3],
                [1,0,3,2],
                [2,3,0,1],
                [3,2,1,0]]

TArr=np.array([10,60,110,160])
xiArr=np.sqrt(np.sqrt(np.array([ 73.69107321965896,49.291206954195914,26.67669675693746,16.559679037111323])*50/(73.69107321965896*7.5)*(2.58**2)))
interpolate_xifit = interp1d(TArr, xiArr, kind='linear', fill_value='extrapolate')

# Modify the function to return a number when a single number is input
def xi(T):
    return interpolate_xifit(np.array([T]))[0]


# get the LiSigmaj components of Sigma... Lambda_i sigma_j in Grgur's notation
# input: rank-4 tensor of Sigma(w,k,i,j) or Sigma(kx,ky,i,j)
# this is specific to a basis of 6 orbitals, as in Sr2RuO4
def get_SigmaLScomponents(matrix, isDelta=False):
    components = np.zeros(shape=(*matrix.shape[:2],9,4))
    for i in range(9):
        for j in range(4):
            if isDelta:
                mat = np.kron(GellMannMatrix[i], 1j*sigma[j]@sigma[2])
            else:
                mat = np.kron(GellMannMatrix[i], sigma[j])
            components[:,:,i,j] = np.einsum('wkab,ba->wk', matrix, mat)/4
    if isDelta:
        components[:,:,:,0]*=-1
        components[:,:,:,2]*=-1
    return components

# group the LS components by local spin-orbital character
# input: rank-4 tensor of LScomponents(w,k,i,j) or LScomponents(kx,ky,i,j)
# this is specific to the 6 orbitals in the t2g sector, as in Sr2RuO4
def get_Gamma(LScomponents):
    A1g = [LScomponents[:,:,0,0],LScomponents[:,:,4,0],LScomponents[:,:,2,3],LScomponents[:,:,8,1]-LScomponents[:,:,6,2],LScomponents[:,:,7,1]-LScomponents[:,:,5,2]]
    A2g = [LScomponents[:,:,2,0],LScomponents[:,:,0,3],LScomponents[:,:,4,3],LScomponents[:,:,8,2]+LScomponents[:,:,6,1],LScomponents[:,:,7,2]+LScomponents[:,:,5,1]]
    B1g = [LScomponents[:,:,3,0],LScomponents[:,:,8,1]+LScomponents[:,:,6,2],LScomponents[:,:,7,1]+LScomponents[:,:,5,2],LScomponents[:,:,1,3]]
    B2g = [LScomponents[:,:,1,0],LScomponents[:,:,8,2]-LScomponents[:,:,6,1],LScomponents[:,:,7,2]-LScomponents[:,:,5,1]]
    return A1g,A2g,B1g,B2g


# decompose a k-space form factor into symmetry components
# input: rank-4 tensor of kFF(kx,ky,i,j)
# output: dictionary of decomposed components with keys: s, dx2y2, dxy, g, p
def decompose_form_factor(kFF, nk):
    """
    Decompose a k-space form factor into symmetry components.
    
    Parameters:
    -----------
    kFF : ndarray
        Input form factor with shape (nk,nk,norb,norb)
    nk : int
        Number of k-points in each direction
        
    Returns:
    --------
    dict : Dictionary containing the decomposed components:
        's': A1g: even under inversion, kx reflection, and kx<->ky
        'dx2y2': B1g: even under inversion and kx reflection, odd under kx<->ky
        'dxy': B2g: even under inversion, odd under kx reflection, even under kx<->ky
        'g': A2g: even under inversion, odd under kx reflection and kx<->ky
        'p': Eu: the parity-odd (under inversion) component
    """
    # Build reflection index that keeps the k=0 entry fixed and reverses the rest
    inv_idx = np.concatenate(([0], np.arange(nk - 1, 0, -1))).astype(int)
    
    # First split into inversion even/odd
    inv_flip = kFF[inv_idx][:,inv_idx]
    even = 0.5 * (kFF + inv_flip)
    odd = 0.5 * (kFF - inv_flip)
    
    # For the even component, decompose under kx reflection
    kx_flip = even[inv_idx]
    kx_even = 0.5 * (even + kx_flip)
    kx_odd = 0.5 * (even - kx_flip)
    
    # For each kx parity component, decompose under kx<->ky
    diag_flip_even = kx_even.transpose(1, 0, *range(2, kFF.ndim))
    kx_even_diag_even = 0.5 * (kx_even + diag_flip_even)
    kx_even_diag_odd = 0.5 * (kx_even - diag_flip_even)
    
    diag_flip_odd = kx_odd.transpose(1, 0, *range(2, kFF.ndim))
    kx_odd_diag_even = 0.5 * (kx_odd + diag_flip_odd)
    kx_odd_diag_odd = 0.5 * (kx_odd - diag_flip_odd)
        
    return {
        's': kx_even_diag_even,
        'dx2y2': kx_even_diag_odd,
        'dxy': kx_odd_diag_even,
        'g': kx_odd_diag_odd,
        'p': odd
    }


# Helper for preparing self-energy real part for plotting
# input sigma_wk as returned from the solver
# output rank-4 tensor of LScomponents(kx,ky,i,j), chaining get_SigmaLScomponents and get_Gamma after symmetrizing freq

def prep_for_plot_SigmaRe(sigma_wk, nk, norb):
    """Average first positive and negative frequencies for Re Sigma."""
    # Handle TRIQS slicing and data extraction safely
    s_p = sigma_wk(0, all)
    s_n = sigma_wk(-1, all)
    
    # Avoid .data on ndarrays, use it for TRIQS/nda objects
    data_p = s_p.data if (hasattr(s_p, 'data') and not isinstance(s_p, np.ndarray)) else s_p
    data_n = s_n.data if (hasattr(s_n, 'data') and not isinstance(s_n, np.ndarray)) else s_n
    
    sigma_re = 0.5 * (np.asarray(data_p) + np.asarray(data_n))
    gamma = get_Gamma(get_SigmaLScomponents(sigma_re.reshape(nk, nk, norb, norb)))
    return gamma

# Helper for preparing self-energy imaginary part for plotting
# input sigma_wk as returned from the solver
# output rank-4 tensor of LScomponents(kx,ky,i,j), chaining get_SigmaLScomponents and get_Gamma after antisymmetrizing freq
def prep_for_plot_SigmaIm(sigma_wk, nk, norb, T):
    """Extract slope (Im Sigma / omega) for plotting."""
    s_p = sigma_wk(0, all)
    s_n = sigma_wk(-1, all)
    
    data_p = s_p.data if (hasattr(s_p, 'data') and not isinstance(s_p, np.ndarray)) else s_p
    data_n = s_n.data if (hasattr(s_n, 'data') and not isinstance(s_n, np.ndarray)) else s_n
    
    # (Im Sigma) / (i * pi * T) 
    sigma_im = (np.asarray(data_p) - np.asarray(data_n)) / (2j * np.pi * T)
    gamma = get_Gamma(get_SigmaLScomponents(sigma_im.reshape(nk, nk, norb, norb)))
    return gamma

# Helper for preparing gap function for plotting
# input vs_wk as returned from the solver or 1D array of Delta_iw,kx,ky,alpha,beta flattened
# output rank-4 tensor of LScomponents(kx,ky,i,j), chaining get_DeltaLScomponents and get_Gamma
def prep_for_plot_Delta(vs, nw, nk, norb, oddfreq=False):
    """Prepare gap function (even or odd frequency) for point-group analysis."""
    # Handle both Gf objects and flat numpy arrays
    # Avoid .data on ndarrays (it's a memoryview), use it for TRIQS objects
    if hasattr(vs, 'data') and not isinstance(vs, np.ndarray):
        delta_data = np.asarray(vs.data).reshape(2 * nw, nk, nk, norb, norb)
    else:
        delta_data = np.asarray(vs).reshape(2 * nw, nk, nk, norb, norb)
    
    # Extract first positive and negative frequencies
    # indices: nw is first positive, nw-1 is first negative for a 2*nw mesh
    d_p = delta_data[nw]
    d_n = delta_data[nw-1]
    
    if oddfreq:
        delta_k = 0.5 * (d_p - d_n)
    else:
        delta_k = 0.5 * (d_p + d_n)

    # Normalize by max absolute value for visual comparison
    max_idx = np.unravel_index(np.abs(delta_k).argmax(), delta_k.shape)
    max_val = delta_k[max_idx]
    if np.abs(max_val) > 1e-12:
        delta_k /= max_val

    gamma = get_Gamma(get_SigmaLScomponents(delta_k, isDelta=True))
    return gamma

# input rank-4 tensor of LScomponents(kx,ky,i,j)
# output list of indices of dominant form-factors; first index is the irrep, second is the form-factor see get_Gamma
# and weights of the form-factors, truncated at 1% of max
# and nearest character of the 
# and plot of Gamma
def plot_Gamma(Gamma,nk,uniform_colorbar=False, round_character=True):
    max_abs_values = get_max_abs_values(Gamma)
    # Find the maximum value and its index in max_abs_values
    max_value = max(max(max_abs_values, key=max))
    max_index, sub_index = [(i, group.index(max_value)) for i, group in enumerate(max_abs_values) if max_value in group][0]

    # print(f"The index with the largest number in max_abs_values is: {max_index}")
    # print(f"The sub-index with the largest number in max_abs_values is: {sub_index}")
    # print(f"The largest number is: {max_value}")

    # Define the threshold and initialize the indices list
    threshold = max_value / 1000
    indices_list = [(max_index, sub_index)]

    max_values_list = [max_value]
    current_max_value = max_value
    while True:
        next_max_value = -np.inf
        next_max_index = -1
        next_sub_index = -1

        for i, group in enumerate(max_abs_values):
            for j, value in enumerate(group):
                if current_max_value > value > next_max_value:
                    next_max_value = value
                    next_max_index = i
                    next_sub_index = j

        if next_max_value < threshold:
            break

        indices_list.append((next_max_index, next_sub_index))
        max_values_list.append(next_max_value)
        current_max_value = next_max_value

    print(f"Indices and sub-indices of values until less than max_value/100: {indices_list}")
    print(f"Corresponding max values: {max_values_list}")

    fig, axs = plt.subplots(1, len(indices_list), figsize=(5*len(indices_list), 5))
    if len(indices_list) == 1:
        axs = [axs]
        
    if uniform_colorbar:
        # Find the global min and max values across all Gamma elements
        global_min = min(np.min(Gamma[max_index][sub_index].real) for max_index, sub_index in indices_list)
        global_max = max(np.max(Gamma[max_index][sub_index].real) for max_index, sub_index in indices_list)
        vmin, vmax = min(global_min, -global_max), max(global_max, -global_min)

    for idx, (max_index, sub_index) in enumerate(indices_list):
        ax = axs[idx]
        if round_character:
            target = np.array(character(Gamma[max_index][sub_index],nk))
        else:
            target = np.array(character_weighted(Gamma[max_index][sub_index],nk))
        closest_key = None
        closest_distance = np.inf

        for key in SpinOrbitalCharacter_lookup.keys():
            distance = np.linalg.norm(target - np.array(key))
            if distance < closest_distance:
                closest_distance = distance
                closest_key = key

        closest_entry = SpinOrbitalCharacter_lookup[closest_key]
        closest_key_index = list(SpinOrbitalCharacter_lookup.keys()).index(closest_key)
        # Find the irreducible representation of the product
        irrep_product = product_table[max_index][closest_key_index]
        irrep_product_entry = Irreps[irrep_product]
        output_str = rf"{LocalSpinOrbital_list[max_index][sub_index]}\," + r" $\varphi(k) [i,\sigma_v,\sigma_d]$ = " + f"{target}" + r" $\implies$ " + f"{closest_entry}, overall {irrep_product_entry}"
        display(Math(output_str))
        # print(f"The index of the closest key is: {closest_key_index}")
        # print('The irreducible representation of the product is:', f'{irrep_product_entry}')
        x, y = np.meshgrid(np.arange(Gamma[max_index][sub_index].real.shape[1]), np.arange(Gamma[max_index][sub_index].real.shape[0]))
        if uniform_colorbar:
            sc = ax.imshow(Gamma[max_index][sub_index].real, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
        else:
            sc = ax.imshow(Gamma[max_index][sub_index].real, cmap='RdBu_r', origin='lower', interpolation='nearest')
        plt.colorbar(sc, ax=ax)
        ax.set_title(f'{LocalSpinOrbital_list[max_index][sub_index]}{closest_entry}', fontsize=24)
        xlabel_text = r"$[i,\sigma_v,\sigma_d]$"
        ax.set_xlabel(f'{xlabel_text}= {target}', fontsize=18)
        for spine in ax.spines.values():
            spine.set_edgecolor(['black','red','green','blue'][irrep_product])
            spine.set_linewidth(5)

    plt.show()

# implements bandstructure from Stangier, Berg, Schmalian, no SOC
class SBSSr2RuO4(EliashbergSolver):
    def __init__(self, nk=12, n_w=1024, T=700., mu=0., Xi=2.58, Q=2*pi*0.3, g=1.36/np.sqrt(3), eps_xx=0.0, nu_xy=0.39, alpha=1.0, beta=1.0, norb=6, Vertices=None, **kwargs):
        if Vertices is None:
             Vertices = L0sigma[1:] # Default to spin-interaction vertices only

        # SRO parameters
        self.eps_xx = eps_xx
        self.nu_xy = nu_xy
        self.alpha = alpha
        self.beta = beta
        
        # Band structure parameters (unstrained) from Appendix B
        self.t1 = 0.119       # eV
        self.t4 = 0.41 * self.t1   # eV
        self.mu_xy = 1.48 * self.t1

        self.t2 = 0.165       # eV
        self.t3 = 0.08 * self.t2   # eV
        self.t5 = 0.13 * self.t2   # eV
        self.eps0_x = -0.18 * self.t2   # epsilon_x^(0) - mu = -0.18 t2 
        self.eps0_y = -0.18 * self.t2
        
        # Susceptibility parameters
        self.Chi0  = 73.69*(7.5/1000)/2.58**2
        self.gamma = 20.
        self.bandwidth = 3.2534755286387123 # Kept largely the same, maybe needs updating based on new dispersion?
        
        super().__init__(nk=nk, n_w=n_w, T=T, mu=mu, Xi=Xi, Q=Q, norb=norb, Vertices=Vertices, g=g, **kwargs)

    def tight_binding_parameters_strain(self, eps_xx, nu_xy=0.39, alpha=1.0, beta=1.0):
        """
        Return strain-renormalized hopping parameters for given epsilon_xx,
        as in Eq. (B2) with rho_xy=0, rho_yy=-nu_xy rho_xx. 
        """
        # gamma band
        t1x = self.t1 * (1.0 - alpha * eps_xx)
        t1y = self.t1 * (1.0 + alpha * nu_xy * eps_xx)
        t4_eff = self.t4 * (1.0 - 0.5 * alpha * (1.0 - nu_xy) * eps_xx)

        # xz/yz bands
        t2x = self.t2 * (1.0 - beta * eps_xx)
        t2y = self.t2 * (1.0 + beta * nu_xy * eps_xx)
        t3x = self.t3 * (1.0 - beta * eps_xx)
        t3y = self.t3 * (1.0 + beta * nu_xy * eps_xx)
        t5_eff = self.t5 * (1.0 - 0.5 * beta * (1.0 - nu_xy) * eps_xx)

        return t1x, t1y, t4_eff, t2x, t2y, t3x, t3y, t5_eff

    def calculate_hk(self, kx, ky, kz):
        """
        Construct the 3x3 single-particle Hamiltonian H(k) of Eq. (6)
        in the {d_xy, d_xz, d_yz} basis for given kx, ky 
        Strain epsilon_xx enters via Appendix B tight-binding parametrization. 
        """
        # strain-dependent hoppings
        t1x, t1y, t4_eff, t2x, t2y, t3x, t3y, t5_eff = \
            self.tight_binding_parameters_strain(self.eps_xx, self.nu_xy, self.alpha, self.beta)

        # gamma (d_xy) band dispersion, Eq. (B1) 
        eps_xy = (
            -2.0 * t1x * np.cos(kx)
            -2.0 * t1y * np.cos(ky)
            -2.0 * t4_eff * np.cos(kx + ky)
            -2.0 * t4_eff * np.cos(kx - ky)
            - self.mu_xy
        )

        # xz/yz dispersions and hybridization, Eqs. (B3),(B4) 
        eps_xz = (
            self.eps0_x
            - 2.0 * t2x * np.cos(kx)
            - 2.0 * t3y * np.cos(ky)
        )
        eps_yz = (
            self.eps0_y
            - 2.0 * t2y * np.cos(ky)
            - 2.0 * t3x * np.cos(kx)
        )
        V_k = (
            -2.0 * t5_eff * np.cos(kx + ky)
            + 2.0 * t5_eff * np.cos(kx - ky)
        )

        # 3x3 Hamiltonian matrix, Eq. (6) 
        # Indices: 0=xy, 1=xz, 2=yz matching the GellMann basis order?
        # Typically xz, yz, xy order is used in standard params but here it seems xy, xz, yz.
        # Need to verify basis. But assuming this implementation is correct for now.
        # The user requested to change the basis to (yz, xz, xy) for consistency
        hk = np.array([
            [eps_yz, V_k,     0.0],
            [V_k,    eps_xz,  0.0],
            [0.0,    0.0,     eps_xy],
        ], dtype=float)

        return np.kron(hk, np.eye(2))

    # -----------------------------------------------------------------------
    # Symmetry definitions for SBSSr2RuO4 (SRO) - D4h 16 operations
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

    # Orbital and Spin Unitaries for SBSSr2RuO4
    def _U_identity(self): return np.eye(self.norb)
    def _U_inversion(self): return np.eye(self.norb)
    
    def _U_sigma_x(self):
        U_orb = np.diag([1, -1, -1])
        U_spin = -1j * sigma[1]
        return np.kron(U_orb, U_spin)

    def _U_sigma_y(self):
        U_orb = np.diag([-1, 1, -1])
        U_spin = -1j * sigma[2]
        return np.kron(U_orb, U_spin)

    def _U_sigma_d(self):
        U_orb = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        U_spin = (-1j / np.sqrt(2)) * (sigma[1] - sigma[2])
        return np.kron(U_orb, U_spin)

    def _U_sigma_dp(self):
        U_orb = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
        U_spin = (-1j / np.sqrt(2)) * (sigma[1] + sigma[2])
        return np.kron(U_orb, U_spin)

    def _U_sigma_h(self):
        U_orb = np.diag([-1, -1, 1])
        U_spin = -1j * sigma[3]
        return np.kron(U_orb, U_spin)

    def _U_C2z(self):
        # C2z = sigma_x * sigma_y
        return self._U_sigma_x() @ self._U_sigma_y()

    def _U_C4z_anti(self):
        # C4z_anti = sigma_x * sigma_d
        return self._U_sigma_x() @ self._U_sigma_d()

    def _U_C4z_clock(self):
        # C4z_clock = sigma_y * sigma_d
        return self._U_sigma_y() @ self._U_sigma_d()

    def _U_S4(self):
        # S4 = sigma_h * C4z_anti
        return self._U_sigma_h() @ self._U_C4z_anti()

    def _U_S4_inv(self):
        # S4_inv = sigma_h * C4z_clock
        return self._U_sigma_h() @ self._U_C4z_clock()

    def _U_C2x(self):
        # C2x = sigma_h * sigma_y
        return self._U_sigma_h() @ self._U_sigma_y()

    def _U_C2y(self):
        # C2y = sigma_h * sigma_x
        return self._U_sigma_h() @ self._U_sigma_x()

    def _U_C2_xplusy(self):
        # C2_axis where axis is y=x. C2 = sigma_h * sigma_d
        return self._U_sigma_h() @ self._U_sigma_d()

    def _U_C2_xminusy(self):
        # C2_axis where axis is y=-x. C2 = sigma_h * sigma_dp
        return self._U_sigma_h() @ self._U_sigma_dp()


    def calculate_gamma(self, wmesh_boson_kmesh, target_shape):
        """
        Calculate Gamma_ph using Paramagnon form-factor.
        Free Parameters:
            Chi0: Static susceptibility prefactor
            Xi: Correlation length
            gamma: Landau damping parameter
            bandwidth: Bandwidth for onset of high frequency scaling
        
        Factorized form: gamma_ph[w, k, a, b, c, d] = chi_wk[w, k] * vertex_sum[a, b, c, d]
        """
        # chi_wr_from_chi_wk expects a 4-index tensor (tensor_valued<4>)
        chi_wk = Gf(mesh=wmesh_boson_kmesh, target_shape=(1, 1, 1, 1))
        
        for iO, k in wmesh_boson_kmesh:
            kx, ky, kz = k
            for Qx in [self.Q, -self.Q]:
                for Qy in [self.Q, -self.Q]:
                    denominator = (1/self.Xi**2 + 2 * (2 - np.cos(kx - Qx) - np.cos(ky - Qy)) + 
                                  np.abs(iO.value) * self.gamma * (1 + (np.abs(iO.value))/(self.bandwidth)))
                    chi_wk[iO, k][0,0,0,0] += self.Chi0 / denominator / 4.0
        
        # Constant vertex tensor in orbital space
        vertex_sum = np.zeros(target_shape*2, dtype=complex)
        for v in self.Vertices:
             vertex_sum += np.tensordot(v, v, axes=0)
             
        # Store factorized parts for memory-efficient solvers
        self.chi_wk = chi_wk
        self.vertex_sum = vertex_sum

        # We keep gamma_ph creation for compatibility, but it might crash for NW=4096.
        # Check if mesh is small enough to avoid memory crash (threshold ~ 4GB)
        mesh_size = len(wmesh_boson_kmesh) * np.prod(target_shape*2) * 16 # 16 bytes per complex
        if mesh_size < 4 * 1e9:
            gamma_ph = Gf(mesh=wmesh_boson_kmesh, target_shape=target_shape*2)
            gamma_ph.data[:] = self.g**2 * chi_wk.data[:, :, 0, 0, 0, 0, None, None, None, None] * vertex_sum
            self.gamma_ph = gamma_ph
        else:
            print("Warning: Large mesh detected. Skipping full gamma_ph allocation to save memory.")

        print(f"Just loaded factorized interaction objects at {datetime.datetime.now()}")
        return None # Return None as solvers should use self.chi_wk

    def calculate_gamma_static(self, kmesh, target_shape):
        """
        Calculate Static Gamma_ph using Paramagnon form-factor at w=0.
        Free Parameters:
            Chi0: Static susceptibility prefactor
            Xi: Correlation length
        Factorized form: gamma_ph_static[k, a, b, c, d] = chi_k[k] * vertex_sum[a, b, c, d]
        """
        chi_k = Gf(mesh=kmesh, target_shape=(1, 1, 1, 1))
        
        for k in kmesh:
            kx, ky, kz = k
            for Qx in [self.Q, -self.Q]:
                for Qy in [self.Q, -self.Q]:
                    denominator = (1/self.Xi**2 + 2 * (2 - np.cos(kx - Qx) - np.cos(ky - Qy)))
                    chi_k[k][0,0,0,0] += self.Chi0 / denominator / 4.0
        
        vertex_sum = np.zeros(target_shape*2, dtype=complex)
        for v in self.Vertices:
             vertex_sum += np.tensordot(v, v, axes=0)
             
        self.chi_k = chi_k
        self.vertex_sum = vertex_sum
        
        # Static mesh is usually small, so we keep gamma_ph_static
        gamma_ph_static = Gf(mesh=kmesh, target_shape=target_shape*2)
        gamma_ph_static.data[:] = self.g**2 * chi_k.data[:, 0, 0, 0, 0, None, None, None, None] * vertex_sum
        
        print(f"Just loaded static factorized interaction at {datetime.datetime.now()}")
        self.gamma_ph_static = gamma_ph_static
        return gamma_ph_static

    def plot_quasiparticle_weight(self, orbital_resolved=False, figsize=(8, 7)):
        """Plot the quasiparticle spectral weight Z = 1/(1 - Im Σ(iω₁)/ω₁).
        
        For a multi-band system, Z is a matrix: Z = [I - Im Σ(iω₁)/ω₁]⁻¹.
        
        Parameters:
        -----------
        orbital_resolved : bool
            If True, plot Z for each orbital (trace over spin).
            If False, plot the average Z (trace over all orbitals and spins).
        figsize : tuple
            Figure size (width, height).
        """
        if not hasattr(self, 'sigma_wk'):
            raise ValueError("No self-energy available. Run solver() first.")

        # Extract the self-energy at the first Matsubara frequency
        # self.sigma_wk(0, all) returns the Gf at the first positive frequency for all k
        # Shape: (nk, nk, norb, norb)
        sigma_im = self.sigma_wk(0, all).data.reshape(self.nk, self.nk, self.norb, self.norb).imag

        # Calculate the renormalization factor matrix Z = [I - Im Σ(iω₁)/ω₁]⁻¹
        # where ω₁ = π/β is the first Matsubara frequency
        beta = 11604. / self.T
        omega1 = pi / beta  # first fermionic Matsubara frequency (positive imaginary part)
        
        # Identity matrix for (norb, norb)
        identity = np.eye(self.norb)
        
        # Calculate Z matrix for each k-point
        # D = Im Σ / ω₁
        # Z = inv(I - D)
        D = sigma_im / omega1
        
        # We need to invert (I - D) for each k point.
        # D has shape (nk, nk, norb, norb).
        # We can reshape to (nk*nk, norb, norb) for easier iteration or broadcasting if supported.
        nk_sq = self.nk * self.nk
        D_reshaped = D.reshape(nk_sq, self.norb, self.norb)
        Z_reshaped = np.zeros_like(D_reshaped)
        
        for i in range(nk_sq):
            Z_reshaped[i] = np.linalg.inv(identity - D_reshaped[i])
            
        Z_matrix = Z_reshaped.reshape(self.nk, self.nk, self.norb, self.norb)

        if orbital_resolved:
            # Assumes norb is even and orbitals are paired with spins (e.g. 6 orbitals = 3 spatial * 2 spin)
            # We want to trace over spin for each spatial orbital.
            # Spatial orbitals: norb/2
            n_spatial = self.norb // 2
            
            fig, axes = plt.subplots(1, n_spatial, figsize=(5*n_spatial, 5))
            if n_spatial == 1:
                axes = [axes]
                
            for iorb in range(n_spatial):
                # Trace over the spin block for this orbital
                # The block assumes indices are [orb_up, orb_down] or [orb1_up, orb1_down, orb2_up, ...]
                # Based on L0sigma definition: kron(I_3, sigma), so indices are:
                # 0: orb1_up, 1: orb1_down, 2: orb2_up, 3: orb2_down, 4: orb3_up, 5: orb3_down.
                # Actually, L0sigma uses kron(eye(3), sigma), so:
                # 0,1 -> orb1; 2,3 -> orb2; 4,5 -> orb3
                
                # Slice indices
                s_idx = 2 * iorb
                e_idx = 2 * iorb + 2
                
                # Trace over the 2x2 block
                z_orb = np.trace(Z_matrix[:, :, s_idx:e_idx, s_idx:e_idx], axis1=2, axis2=3) / 2.0
                
                ax = axes[iorb]
                im = ax.imshow(z_orb, cmap='coolwarm_r', origin='lower')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'Orbital {iorb+1} Z')
                ax.set_xlabel('kx')
                if iorb == 0:
                    ax.set_ylabel('ky')
            
            plt.suptitle("Orbital-resolved Quasiparticle Weight Z")
            return fig, axes
            
        else:
            # Average over all bands (trace over full matrix / norb)
            avg_Z = np.trace(Z_matrix, axis1=2, axis2=3) / self.norb
    
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(avg_Z, cmap='coolwarm_r', origin='lower')
            cbar = plt.colorbar(im, ax=ax)
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_title("Average Quasiparticle Weight Z")
            cbar.set_label('Z factor')
    
            return fig, ax

    def plot_delta(self, oddfreq=False):
        """Plot the gap Δ(k) using the multi-orbital plotting function."""
        if hasattr(self, 'delta_wk') and self.delta_wk is not None:
            Delta = prep_for_plot_Delta(self.delta_wk, self.n_w, self.nk, self.norb, oddfreq=oddfreq)
        elif hasattr(self, 'vs_dynamic'):
            Delta = prep_for_plot_Delta(self.vs_dynamic[:, 0], self.n_w, self.nk, self.norb, oddfreq=oddfreq)
        else:
            raise ValueError("No gap function available. Run solver first.")
        
        plot_Gamma(Delta, self.nk)

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

        Gamma = get_Gamma(get_SigmaLScomponents(Delta_flat, isDelta=True))
        plot_Gamma(Gamma, self.nk)

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
        return fig, ax

    def plot_fermi_surface(self, figsize=(8, 7), nkmesh=None):
        """Plot the Fermi surface from the Green's function.
        Calculates Trace[G(w0,k) * G(w-1,-k)] summed over orbitals.
        
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

        # Extract G at lowest Matsubara freq
        # gw0: first positive freq (i*pi/beta)
        # gwm1: first negative freq (-i*pi/beta)
        
        if nkmesh is None or nkmesh == self.nk:
            # Use existing coarse mesh data
            gw0 = self.g_wk(0, all).data.reshape(self.nk, self.nk, self.norb, self.norb)
            gwm1 = self.g_wk(-1, all).data.reshape(self.nk, self.nk, self.norb, self.norb)
            
            # Create inversion index for -k
            inv_idx = np.concatenate(([0], np.arange(self.nk - 1, 0, -1))).astype(int)
            
            # gwm1(-k)
            gwm1_minusk = gwm1[inv_idx, :][:, inv_idx]
            
            # Compute trace of product
            trace_product = np.einsum('xyij,xyji->xy', gw0, gwm1_minusk).real
            xlabel = 'kx (index)'
            ylabel = 'ky (index)'
            extent = None
            
        else:
            # Interpolate on fine mesh
            print(f"Interpolating Fermi surface on {nkmesh}x{nkmesh} grid...")
            k_grid = np.linspace(0, 2*np.pi, nkmesh, endpoint=False)
            trace_product = np.zeros((nkmesh, nkmesh))
            
            # Use slice then interpolate with list argument
            gw0_slice = self.g_wk(0, all)
            gw_neg_slice = self.g_wk(-1, all)
            
            for i, kx in enumerate(k_grid):
                for j, ky in enumerate(k_grid):
                    # -k = (2pi - kx, 2pi - ky) modulo 2pi
                    # We pass -kx, -ky directly as list/array of floats
                    
                    k_vec = [float(kx), float(ky), 0.0]
                    mk_vec = [float(-kx), float(-ky), 0.0]
                    
                    G_k = gw0_slice(k_vec)
                    G_minusk = gw_neg_slice(mk_vec)
                    
                    # Trace(G(k) * G(-k))
                    # G_k is matrix (norb, norb)
                    prod = G_k @ G_minusk
                    trace_product[i, j] = np.trace(prod).real
                    
            xlabel = 'kx'
            ylabel = 'ky'
            extent = [0, 2*np.pi, 0, 2*np.pi]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(trace_product, cmap='coolwarm', origin='lower', extent=extent)
        cbar = plt.colorbar(im, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Fermi Surface: Tr[G(ω₀,k) G(-ω₀,-k)]")
        cbar.set_label('Trace Product')

        return fig, ax

# implements bandstructure from Palle, Schmalian, intended for a BCC lattice, so some terms are turned off to make it work on a square lattice
class PSSr2RuO4(SBSSr2RuO4):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override parameters to match EliashbergFunction.py's EliashbergSolverSRO
        self.t1 = 27.8 / 1000
        self.t2 = 257.8 / 1000
        self.t3 = -22.4 / 1000
        self.t4 = 13.6 / 1000 * 0 # turned off because this BCC term breaks the translational symmetry of the square lattice
        self.t5 = 3.2 / 1000
        self.t6 = -35.5 / 1000
        self.t7 = 0 / 1000
        self.t8 = -4.7 / 1000
        self.t9 = 0 / 1000
        self.t10 = 0 / 1000
        self.t11 = -2.4 / 1000
        self.bt1 = 356.8 / 1000
        self.bt2 = 126.3 / 1000
        self.bt3 = -1.0 / 1000 * 0 # turned off because this BCC term breaks the translational symmetry of the square lattice
        self.bt4 = 17.0 / 1000
        self.bt5 = 22.3 / 1000
        self.bt6 = 0 / 1000
        self.bt7 = 0 / 1000
        self.Mu1D = 286.9 / 1000
        self.Mu2D = 351.9 / 1000
        self.int1 = -2.0 / 1000
        self.int2 = 7.8 / 1000 * 0 # turned off because this BCC term breaks the translational symmetry of the square lattice
        self.int3 = 0 / 1000
        self.int4 = 0 / 1000
        self.jnt = 2.7 / 1000
        self.Eta1 = 59.2 / 1000
        self.Eta2 = 59.2 / 1000
        self.bandwidth = 3.2534755286387123

        # Susceptibility parameters
        self.Chi0  = 73.69*(7.5/1000)/2.58**2
        self.gamma = 20.

    def calculate_hk(self, k1, k2, k3):
        epsilon_1D = -self.Mu1D \
                    - 2 * self.t1 * np.cos(k1) \
                    - 2 * self.t2 * np.cos(k2) \
                    - 4 * self.t3 * np.cos(k1) * np.cos(k2) \
                    - 8 * self.t4 * np.cos(k1 / 2) * np.cos(k2 / 2) * np.cos(k3 / 2) \
                    - 2 * self.t5 * np.cos(2 * k1) \
                    - 2 * self.t6 * np.cos(2 * k2) \
                    - 4 * self.t7 * np.cos(2 * k1) * np.cos(k2) \
                    - 4 * self.t8 * np.cos(k1) * np.cos(2 * k2) \
                    - 4 * self.t9 * np.cos(2 * k1) * np.cos(2 * k2) \
                    - 2 * self.t10 * np.cos(3 * k1) \
                    - 2 * self.t11 * np.cos(3 * k2)
        epsilon_1D2 = -self.Mu1D \
                    - 2 * self.t1 * np.cos(k2) \
                    - 2 * self.t2 * np.cos(k1) \
                    - 4 * self.t3 * np.cos(k2) * np.cos(k1) \
                    - 8 * self.t4 * np.cos(k2 / 2) * np.cos(k1 / 2) * np.cos(k3 / 2) \
                    - 2 * self.t5 * np.cos(2 * k2) \
                    - 2 * self.t6 * np.cos(2 * k1) \
                    - 4 * self.t7 * np.cos(2 * k2) * np.cos(k1) \
                    - 4 * self.t8 * np.cos(k2) * np.cos(2 * k1) \
                    - 4 * self.t9 * np.cos(2 * k2) * np.cos(2 * k1) \
                    - 2 * self.t10 * np.cos(3 * k2) \
                    - 2 * self.t11 * np.cos(3 * k1)
        epsilon_2D = -self.Mu2D \
                    - 2 * self.bt1 * (np.cos(k1) + np.cos(k2)) \
                    - 4 * self.bt2 * np.cos(k1) * np.cos(k2) \
                    - 8 * self.bt3 * np.cos(k1 / 2) * np.cos(k2 / 2) * np.cos(k3 / 2) \
                    - 2 * self.bt4 * (np.cos(2 * k1) + np.cos(2 * k2)) \
                    - 4 * self.bt5 * (np.cos(2 * k1) * np.cos(k2) + np.cos(k1) * np.cos(2 * k2)) \
                    - 4 * self.bt6 * np.cos(2 * k1) * np.cos(2 * k2) \
                    - 2 * self.bt7 * (np.cos(3 * k1) + np.cos(3 * k2))
        epsilon_i = 4 * self.int1 * np.sin(k1) * np.sin(k2) \
                    + 8 * self.int2 * np.sin(k1 / 2) * np.sin(k2 / 2) * np.cos(k3 / 2) \
                    + 8 * self.int3 * (np.cos(k1) + np.cos(k2)) * np.sin(k1) * np.sin(k2) \
                    + 4 * self.int4 * np.sin(2 * k1) * np.sin(2 * k2)
        epsilon_j = 8 * self.jnt * np.sin(k1 / 2) * np.cos(k2 / 2) * np.sin(k3 / 2)
        epsilon_j2 = 8 * self.jnt * np.cos(k1 / 2) * np.sin(k2 / 2) * np.sin(k3 / 2)
        
        epsilon_matrix = np.array([
            [epsilon_1D, epsilon_i, epsilon_j],
            [epsilon_i, epsilon_1D2, epsilon_j2],
            [epsilon_j, epsilon_j2, epsilon_2D]
        ])
        
        hk = np.kron(epsilon_matrix, sigma[0]) + (
            self.Eta1 * np.kron(-1*GellMannMatrix[8], sigma[1]) +
            self.Eta1 * np.kron(GellMannMatrix[6], sigma[2]) +
            self.Eta2 * np.kron(-1*GellMannMatrix[2], sigma[3])
        )

        return hk


if __name__ == '__main__':
    print("=== SRO1 (SBSSr2RuO4) ===")
    SRO1 = SBSSr2RuO4(T=100, n_w=512, g=2., tol=1e-6, fixed_density=True, filling=0.65)
    print("\nSolving Dyson Equation...")
    SRO1.dyson_solver()
    print("\nSolving Linearized Gap Equation (Dynamic)...")
    SRO1.solve_linearized_gap_dynamic()
    
    print("\n--- SRO1 Minimal Story ---")
    print(f"Run Parameters: T={SRO1.T}, n_w={SRO1.n_w}, g={SRO1.g}, filling={SRO1.filling}")
    print(f"Leading Eigenvalues: {SRO1.En_dynamic}")
    print("\nLeading Eigenvector Symmetry Analysis:")
    SRO1.check_symmetries(verbose=True)

    print("\n\n=== SRO2 (PSSr2RuO4) ===")
    SRO2 = PSSr2RuO4(T=100, n_w=512, g=2., tol=1e-6, fixed_density=True, filling=0.65)
    print("\nSolving Dyson Equation...")
    SRO2.dyson_solver()
    print("\nSolving Linearized Gap Equation (Dynamic)...")
    SRO2.solve_linearized_gap_dynamic()
    
    print("\n--- SRO2 Minimal Story ---")
    print(f"Run Parameters: T={SRO2.T}, n_w={SRO2.n_w}, g={SRO2.g}, filling={SRO2.filling}")
    print(f"Leading Eigenvalues: {SRO2.En_dynamic}")
    print("\nLeading Eigenvector Symmetry Analysis:")
    SRO2.check_symmetries(verbose=True)
