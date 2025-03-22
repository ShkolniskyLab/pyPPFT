import numpy as np
import pytest
import warnings


############################################
# Discrete pseduo-polar and Radon transforms
############################################

def ppft2(im):
    """
    Fast algorithm for computing the pseudo-polar Fourier transform.
    The computation requires O(n² log n) operations.

    Parameters:
        im (numpy.ndarray): The image whose pseudo-polar Fourier transform should be computed.
                            Must be of dyadic square size (2^k x 2^k).

    Returns:        
        res1 (shape: (2n+1, n+1)): Contains the values of PPI1(k, l).
        res2 (shape: (2n+1, n+1)): Contains the values of PPI2(k, l).

    Description:
        res1(k, l): Pseudo-polar Fourier transform corresponding to radius k and angle 
          arctan(2l/n). The index l varies from -n/2 to n/2, covering angles from -π/4 to π/4.
        res2(k, l): Pseudo-polar Fourier transform corresponding to radius k and angle 
          π/2 - arctan(2l/n). The index l varies from -n/2 to n/2, covering angles from 3π/4 to π/4.

    To obtain a continuous change in angle from res1 to res2, res2 must be flipped such that 
    it corresponds to angles from π/4 to 3π/4, ensuring continuity with the angles in res1.

    See the final thesis PDF for more information.
    """
    im = np.flipud(im)

    if im.ndim != 2:
        raise ValueError("Input must be a 2D image")

    n = im.shape[0]
    if im.shape[1] != n:
        raise ValueError("Input image must be square")

    if n % 2 != 0:
        raise ValueError("Input image must have even side")

    m = 2 * n + 1
    alpha = 2 * (n + 1) / (n * m)
    res1 = np.zeros((m, n + 1), dtype=np.complex128)
    res2 = np.zeros((m, n + 1), dtype=np.complex128)

    # Part I: Computation of res1
    EI = np.vstack([np.zeros((n//2, n)), im, np.zeros((n//2+1, n))])
    FEI = cfftd(EI, axis=0)  # Column-wise FFT

    for k in range(-n, n+1):
        u = FEI[to_unaliased_idx(k, m), :]
        w = cfrft(np.append(u, 0), k * alpha)  # Apply fractional FFT
        res1[to_unaliased_idx(k, m), :] = np.flip(w)

    # Part II: Computation of res2
    EI = np.hstack([np.zeros((n, n//2)), im, np.zeros((n, n//2+1))])
    FEI = cfftd(EI, axis=1)  # Row-wise FFT

    for k in range(-n, n+1):
        v = FEI[:, to_unaliased_idx(k, m)]
        w = cfrft(np.append(v, 0), k * alpha)
        res2[to_unaliased_idx(k, m), :] = np.flip(w)

    return res1, res2

def ippft2(pp1, pp2, tol=1e-6, maxiter=10, verbose=False):
    """
    Inverse pseudo-polar Fourier transform.
    The inverse transform is computed using conjugate gradient method.
    
    Parameters:
    -----------
    pp1, pp2 : ndarray
        Pseudo-polar sectors as returned from the function ppft.
    tol : float, optional
        Error tolerance used by the conjugate gradient method. Default 1.e-2.
    maxiter : int, optional
        Maximum number of iterations. Default 10.
    verbose : int, optional
        Display verbose CG information. 0 will suppress verbose information.
        Any non-zero value will display verbose CG information.
    
    Returns:
    --------
    Y : ndarray
        The inverted matrix.
    flag : int
        Convergence flag. See CG for more information.
    residual : float
        Residual error at the end of the inversion.
    iter : int
        The iteration number at which tol was achieved. Relevant only if
        flag=0.
    """    

    temp = precond_adj_ppft2(pp1, pp2)

    # Create zero matrix of the same size as temp for initial guess
    initial_guess = np.zeros_like(temp)

    def PtP(X): return precond_adj_ppft2(*ppft2(X))
    Y, flag, residual, iter, _ = CG(
        PtP, temp, [], tol, maxiter, initial_guess, verbose)

    if flag:
        warnings.warn(
            f'Inversion did not converge. Residual error {residual:.5e}')

    return Y, flag, residual, iter


def radon2(im):
    """
    Fast algorithm for computing the discrete Radon transform.
    The computation requires O(n^2logn) operations.
    
    Parameters:
    -----------
    im : ndarray
        The image whose discrete Radon transform should be computed.
        Must be real (no imaginary components) and of a dyadic square size (2^k x 2^k).
    
    Returns:
    --------
    res1, res2 : ndarray
        Arrays of size (2n+1)x(n+1) that contain the discrete Radon transform of the input image im.
        
        res1 contains the values which correspond to rays of radius k=-n...n and angles
        theta=arctan(2l/n) l=-n/2...n/2 (from -pi/4 to pi/4).
        
        res2 contains the values which correspond to rays of radius k=-n...n and angles
        theta=pi/2-arctan(2l/n) l=-n/2...n/2 (from 3pi/4 to pi/4).
    
    Notes:
    ------
    Due to small round-off errors, the output may contain small imaginary
    components. If the input image is real, the function truncates any
    imaginary components from the output arrays (since the discrete Radon
    transform of a real image is real).
    """
    # Call the optimized PPFT function 
    res1, res2 = ppft2(im)
    
    # Inverse FFT along columns (axis 0)
    res1 = icfftd(res1, axis=0)
    res2 = icfftd(res2, axis=0)
    
    # Check for unexpected imaginary components
    if np.any(np.abs(np.imag(res1)) > 1e-5):
        warnings.warn(f"res1 contains imaginary components of maximal value {np.max(np.abs(np.imag(res1)))}")
        
    if np.any(np.abs(np.imag(res2)) > 1e-5):
        warnings.warn(f"res2 contains imaginary components of maximal value {np.max(np.abs(np.imag(res2)))}")
    
    # Remove the imaginary component if the input image is real
    if np.isrealobj(im):
        res1 = np.real(res1)
        res2 = np.real(res2)
    
    return res1, res2



def iradon2(res1, res2, tol=1e-6, maxiter=10, verbose=True):
    """
    2-D inverse discrete Radon transform.
    The inverse transform is computed using the conjugate gradient method.
    
    Parameters:
    -----------
    res1, res2 : ndarray
        Discrete Radon sectors as returned from the function Radon.
    tol : float, optional
        Error tolerance used by the conjugate gradient method. Default 1.e-2.
    maxiter : int, optional
        Maximum number of iterations. Default 10.
    verbose : int, optional
        Display verbose CG information. 0 will suppress verbose information.
        Any non-zero value will display verbose CG information.
    
    Returns:
    --------
    Y : ndarray
        The inverted matrix.
    flag : int
        Convergence flag. See CG for more information.
    residual : float
        Residual error at the end of the inversion.
    iter : int
        The iteration number at which tol was achieved. Relevant only if
        flag=0.
    """
    # Apply centered FFT along the first dimension (columns)
    temp1 = cfftd(res1, axis=0)
    temp2 = cfftd(res2, axis=0)
    
    # Call the inverse pseudo-polar Fourier transform
    Y, flag, residual, iter = ippft2(temp1, temp2, tol, maxiter, verbose)
    
    return Y, flag, residual, iter


###############
# FFT functions
###############

def cfft(x):
    """
    Aliased FFT of the sequence x.
    The FFT is computed using O(n log n) operations.
    
    Parameters:
    x : array-like
        The sequence whose FFT should be computed. Can be of odd or even length.
        Must be a 1-D vector.
    
    Returns:
    np.ndarray
        The aliased FFT of the sequence x.
    """
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))

def icfft(x):
    """
    Aliased inverse Fourier transform (IFFT) of the sequence x.
    The IFFT is computed using O(n log n) operations.
    
    Parameters:
    x : array-like
        The sequence whose IFFT should be computed. Can be of odd or even length.
    
    Returns:
    np.ndarray
        The aliased IFFT of the sequence x.
    """
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))


def cfftd(x, axis=0):
    """
    Computes the 1-D aliased FFT of a multi-dimensional image x along dimension d.
    
    Parameters:
        x (numpy.ndarray): The image whose FFT should be computed. Can be of odd or even length.
        d (int): The dimension along which to perform the FFT.
    
    Returns:
        numpy.ndarray: The 1-D aliased FFT of the image x along dimension d.
    """
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def icfftd(x, axis=0):
    """
    Computes the 1-D aliased inverse FFT of a multi-dimensional image x along dimension d.

    Parameters:
        x (numpy.ndarray): The image whose inverse FFT should be computed. Can be of odd or even length.
        d (int): The dimension along which to perform the inverse FFT.

    Returns:
        numpy.ndarray: The 1-D aliased inverse FFT of the image x along dimension d.
    """
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def cfft2(x):
    """
    Aliased 2D FFT of the image x.
    The FFT is computed using O(n^2 log n) operations.
    
    Parameters:
    x : array-like
        The image whose 2D FFT should be computed. Can be of odd or even length.
    
    Returns:
    np.ndarray
        The aliased 2D FFT of the image x.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def icfft2(x):
    """
    Aliased 2D Inverse FFT of the sequence x.
    The inverse FFT is computed using O(n^2 log n) operations.
    
    Parameters:
    x : array-like
        The frequency image whose 2D inverse FFT should be computed. Can be of odd or even length.
    
    Returns:
    np.ndarray
        The aliased 2D inverse FFT of the sequence x.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def cfftn(x):
    """
    Aliased n-dimensional FFT of the array x.
    The inverse FFT is computed using O((n^d) log n) operations, where d is the dimension of the image.
    
    Parameters:
    x : array-like
        The frequency image whose FFT should be computed. Can be of odd or even length in each dimension.
    
    Returns:
    np.ndarray
        The aliased n-dimensional FFT of the array x.
    """
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x)))

def icfftn(x):
    """
    Aliased n-dimensional Inverse FFT of the array x.
    The inverse FFT is computed using O((n^d) log n) operations, where d is the dimension of the image.
    
    Parameters:
    x : array-like
        The frequency image whose inverse FFT should be computed. Can be of odd or even length in each dimension.
    
    Returns:
    np.ndarray
        The aliased n-dimensional inverse FFT of the array x.
    """
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x)))


def cfrft(x, alpha):
    """
    Computes the aliased fractional Fourier transform (FRFT) of the sequence x
    using an O(n log n) algorithm.

    The fractional Fourier transform w of the sequence x (with parameter alpha) is defined as:

        w(k) = sum_{u=-n/2}^{n/2-1} x(u) * exp(-2 * pi * i * k * u * alpha / N),  
               where -n/2 <= k <= n/2-1, and N = len(x).

    Parameters:
        x (array-like): The sequence whose FRFT should be computed. Can be of odd or even length.
                        Must be a 1-D row vector.
        alpha (float): The alpha parameter of the fractional Fourier transform.

    Returns:
        numpy.ndarray: The aliased FRFT with parameter alpha of the sequence x.
    """

    m = len(x)
    j = np.arange(low_idx(m), hi_idx(m) + 1)
    j2 = np.arange(low_idx(2 * m), hi_idx(2 * m) + 1)
    E = 1j * np.pi * alpha

    y = x * np.exp(-E * j**2 / m)
    y = np.concatenate([np.zeros(m), y, np.zeros(m)])

    z = np.zeros(3 * m, dtype=np.complex128)
    l = to_unaliased_idx(-m, 3 * m)
    z[l:l + len(j2)] = np.exp(E * j2**2 / m)

    Y = cfft(y)
    Z = cfft(z)
    W = Y * Z
    w = icfft(W)
    w = w[to_unaliased_idx(low_idx(m), 3 * m)
                           :to_unaliased_idx(hi_idx(m), 3 * m) + 1]
    w = w * np.exp(-E * j**2 / m)

    return w

########################################
# Reference functions (used for testing)
########################################
def cdft(x):
    """
    Aliased DFT of the sequence x.
    The DFT is computed directly using O(n^2) operations.

    Parameters:
    x : array-like
        The sequence whose DFT should be computed. Can be of odd or even length.
    
    Returns:
    np.ndarray
        The aliased DFT of the sequence x.
    """
    m = len(x)
    y = np.zeros(m, dtype=np.complex128)
    for k in range(low_idx(m), hi_idx(m) + 1):
        acc = 0
        for j in range(low_idx(m), hi_idx(m) + 1):
            acc += x[to_unaliased_idx(j, m)] * np.exp(-2j * np.pi * j * k / m)
        y[to_unaliased_idx(k, m)] = acc
    return y


def icdft(x):
    """
    Aliased inverse discrete Fourier transform (IDFT) of the sequence x.
    The IDFT is computed directly using O(n^2) operations.

    Parameters:
    x : array-like
        The sequence whose IDFT should be computed. Can be of odd or even length.
    
    Returns:
    np.ndarray
        The aliased IDFT of the sequence x.
    """
    m = len(x)
    y = np.zeros(m, dtype=np.complex128)
    for k in range(low_idx(m), hi_idx(m) + 1):
        acc = 0
        for j in range(low_idx(m), hi_idx(m) + 1):
            acc += x[to_unaliased_idx(j, m)] * np.exp(2j * np.pi * j * k / m)
        y[to_unaliased_idx(k, m)] = acc / m
    return y


def cdft2(x):
    """
    Aliased 2D DFT of the image x.
    The DFT is computed directly using O(n^4) operations.
    
    Parameters:
    x : array-like
        The sequence whose DFT should be computed. Can be of odd or even length.
    
    Returns:
    np.ndarray
        The aliased 2D DFT of the image x.
    """
    x = np.array(x)
    m, n = x.shape
    y = np.zeros_like(x, dtype=np.complex128)
    for xi1 in range(low_idx(n), hi_idx(n) + 1):
        for xi2 in range(low_idx(m), hi_idx(m) + 1):
            acc = 0
            for u in range(low_idx(n), hi_idx(n) + 1):
                for v in range(low_idx(m), hi_idx(m) + 1):
                    acc += x[to_unaliased_idx(v, m), to_unaliased_idx(u, n)] * \
                        np.exp(-2j * np.pi * (u * xi1 / n + v * xi2 / m))
            y[to_unaliased_idx(xi2, m), to_unaliased_idx(xi1, n)] = acc
    return y

def frft(x, alpha):
    """
    Computes the aliased fractional Fourier transform (FRFT) of the sequence x 
    using an O(n²) direct computation.

    The fractional Fourier transform y of the sequence x (with parameter alpha) is defined as:

        y(k) = sum_{u=-n/2}^{n/2-1} x(u) * exp(-2 * pi * i * k * u * alpha / N),  
               where -n/2 <= k <= n/2-1, and N = len(x).

    The value of the fractional Fourier transform y for index k (-n/2 <= k <= n/2-1) 
    is stored in the array y at index to_unaliased_coord(k, N), which is between 0 and N-1.

    Parameters:
        x (array-like): The sequence whose FRFT should be computed. Can be of odd or even length.
        alpha (float): The alpha parameter of the fractional Fourier transform.

    Returns:
        numpy.ndarray: The aliased FRFT with parameter alpha of the sequence x.
                      The result is always a row vector.
    """    
    m = len(x)
    y = np.zeros(m, dtype=np.complex128)

    for k in range(low_idx(m), hi_idx(m) + 1):
        acc = 0
        for j in range(low_idx(m), hi_idx(m) + 1):
            acc += x[to_unaliased_idx(j, m)] * \
                np.exp(-2 * np.pi * 1j * j * k * alpha / m)
        y[to_unaliased_idx(k, m)] = acc

    return y

def slow_ppft2(im):
    im = np.flipud(im)
    s = im.shape

    if len(s) != 2:
        raise ValueError("Input must be a 2D image")

    if s[0] != s[1]:
        raise ValueError("Input image must be square")

    if s[0] % 2 != 0:
        raise ValueError("Input image must have even side")

    n = s[0]
    ppRows = 2 * n + 1
    ppCols = n + 1
    res1 = np.zeros((ppRows, ppCols), dtype=np.complex128)
    res2 = np.zeros((ppRows, ppCols), dtype=np.complex128)

    for k in range(-n, n+1):
        for l in range(-n//2, n//2 + 1):
            res1[to_unaliased_idx(k, ppRows), to_unaliased_idx(
                l, ppCols)] = _trig_poly(im, -2*l*k/n, k)
            res2[to_unaliased_idx(k, ppRows), to_unaliased_idx(
                l, ppCols)] = _trig_poly(im, k, -2*l*k/n)

    return res1, res2


def _trig_poly(im, xi1, xi2):
    n = im.shape[0]
    m = 2 * n + 1
    acc = 0

    for u in range(low_idx(n), hi_idx(n) + 1):
        for v in range(low_idx(n), hi_idx(n) + 1):
            acc += im[to_unaliased_idx(v, n), to_unaliased_idx(u, n)] * \
                np.exp(-2j * np.pi * (xi1 * u + xi2 * v) / m)

    return acc


def slow_radon2(im):
    """
    Computes the pseudo-Radon transform directly.
    The computation requires O(n^3) operations.

    Parameters:
        im (numpy.ndarray): The image whose discrete Radon transform should be computed.
                            Must be real and of dyadic square size (2^k x 2^k).

    Returns:
        tuple (numpy.ndarray, numpy.ndarray):
            - res1: Contains the Radon values of basically horizontal lines.
            - res2: Contains the Radon values of basically vertical lines.
    
    Notes:
        - The first argument of res1 and res2 corresponds to pseudo-radius.
        - The second argument corresponds to pseudo-angle.
    """

    im = np.flipud(im)

    if im.ndim != 2:
        raise ValueError("Input must be a 2D image")

    if im.shape[0] != im.shape[1]:
        raise ValueError("Input image must be square")

    if im.shape[0] % 2 != 0:
        raise ValueError("Input image must have even side")

    n = im.shape[0]
    pp_rows = 2 * n + 1
    pp_cols = n + 1

    res1 = np.zeros((pp_rows, pp_cols))
    res2 = np.zeros((pp_rows, pp_cols))

    # Computation of res1
    for t in range(-n, n + 1):
        for l in range(-n // 2, n // 2 + 1):
            slope = 2 * l / n
            acc = 0

            for u in range(-n // 2, n // 2):
                acc += _I1(im, n, u, slope * u + t)

            res1[to_unaliased_idx(t, pp_rows), to_unaliased_idx(l, pp_cols)] = acc

    # Computation of res2
    for t in range(-n, n + 1):
        for l in range(-n // 2, n // 2 + 1):
            slope = 2 * l / n
            acc = 0

            for v in range(-n // 2, n // 2):
                acc += _I2(im, n, slope * v + t, v)

            res2[to_unaliased_idx(t, pp_rows), to_unaliased_idx(l, pp_cols)] = acc

    return res1, res2


def _I1(im, n, u, y):
    """
    Trigonometric interpolation of I along the columns (y-axis).
    """
    m = 2 * n + 1
    acc = 0

    for v in range(-n // 2, n // 2):
        acc += im[to_unaliased_idx(v, n), to_unaliased_idx(u, n)] * _dirichlet(y - v, m)

    return acc


def _I2(im, n, x, v):
    """
    Trigonometric interpolation of I along the rows (x-axis).
    """
    m = 2 * n + 1
    acc = 0

    for u in range(-n // 2, n // 2):
        acc += im[to_unaliased_idx(v, n), to_unaliased_idx(u, n)] * _dirichlet(x - u, m)

    return acc


def _dirichlet(t, m):
    """
    Compute the value of the Dirichlet kernel of length `m` at point(s) `t`.

    Parameters:
        t (float or array-like): The input value(s) where the Dirichlet kernel is evaluated.
        m (int): The length of the Dirichlet kernel.

    Returns:
        numpy.ndarray: The Dirichlet kernel values at `t`.
    """
    t = np.asarray(t)  # Ensure t is an array for element-wise operations
    y = np.zeros_like(t, dtype=np.float64)

    mask = np.abs(t) < np.finfo(float).eps  # Check for values close to zero
    y[mask] = 1
    y[~mask] = np.sin(np.pi * t[~mask]) / (m * np.sin(np.pi * t[~mask] / m))

    return y

####################
# Auxiliry functions
####################

def precond_adj_ppft2(pp1, pp2):
    """
    Computes the preconditioned adjoint of the pseudo-polar Fourier transform (PPFT).

    Parameters:
        pp1 (numpy.ndarray): The first pseudo-polar section. Must be of size (2n+1, n+1) as obtained from ppft2.
        pp2 (numpy.ndarray): The second pseudo-polar section. Must be of size (2n+1, n+1) as obtained from ppf2.

    Returns:
        numpy.ndarray: The image im that is the preconditioned adjoint of the PPFT.

    Notes:
        - See the differences between this function and `optimizedadjppft.m` to understand how the preconditioner is defined.
        - See `precond.m` for an explicit form of the preconditioner.
    """
    
    if pp1.shape != pp2.shape:
        raise ValueError("pp1 and pp2 must have the same size")

    s1 = pp1.shape
    if s1[0] % 2 == 0 or s1[1] % 2 == 0:
        raise ValueError("pp1 and pp2 must be of size (2n+1)x(n+1)")

    n = (s1[0] - 1) // 2
    if s1[1] - 1 != n:
        raise ValueError("Input parameter must be of size (2n+1)x(n+1)")

    m = 2 * n + 1
    alpha = 2 * (n + 1) / (n * m)

    # Compute adjoint of PP1
    tmp = np.zeros((2 * n + 1, n), dtype=np.complex128)
    for k in range(-n, n + 1):
        mult = 1 / (m ** 2) if k == 0 else abs(k * alpha)
        u = np.flip(pp1[to_unaliased_idx(k, 2 * n + 1), :])
        v = mult * cfrft(u, -k * alpha)
        tmp[to_unaliased_idx(k, 2 * n + 1), :] = v[:n]

    tmp = m * icfftd(tmp, axis=0)  # Inverse FFT along columns
    adjpp1 = np.flipud(tmp[n//2:3*n//2, :])

    # Compute adjoint of PP2
    tmp = np.zeros((2 * n + 1, n), dtype=np.complex128)
    for k in range(-n, n + 1):
        mult = 1 / (m ** 2) if k == 0 else abs(k * alpha)
        u = np.flip(pp2[to_unaliased_idx(k, 2 * n + 1), :])
        v = mult * cfrft(u, -k * alpha)
        tmp[to_unaliased_idx(k, 2 * n + 1), :] = v[:n]

    tmp = m * icfftd(tmp, axis=0)  # Inverse FFT along columns
    tmp = tmp.T
    adjpp2 = np.flipud(tmp[:, n//2:3*n//2])

    # Combine both adjoints
    im = (adjpp1 + adjpp2) / (m ** 2)

    return im

def CG(PtP, X, params=None, tol=1e-9, maxiter=10, initial_guess=0, verbose=0, RefY=None):
    """
    Solve the system X=PtP(Y,params) using the conjugate gradient method.
    The operator PtP must be hermitian and positive defined.
    The first parameter to PtP must be the variable Y.
    
    Parameters:
    -----------
    PtP : callable
        Name of the operator to invert
    X : ndarray
        The transformed matrix. The matrix at the range space of the operator PtP, 
        whose source needs to be found.
    params : list, optional
        Additional parameters to the operator PtP.
    tol : float, optional
        Error tolerance of the CG method. Default 1.e-9.
    maxiter : int, optional
        Maximum number of iterations. Default 10.
    initial_guess : ndarray or int, optional
        Initial guess of the solution. Default is 0.
    verbose : int, optional
        By default, if more than one output argument is expected, then all output 
        messages are suppressed. Set this flag to any value other than 0 to
        always display output messages.
    RefY : ndarray, optional
        The untransformed matrix Y. Used only for checking absolute error.
        If not specified, absolute error is not computed.
    
    Returns:
    --------
    Y : ndarray
        The result matrix of the CG method. This is the estimate of the CG
        method to the solution of the system X=PtP(Y).
    flag : int
        A flag that describes the convergence of the CG method.
        0 CG converged to the desired tolerance tol within maxiter iterations.
        1 CG did not converge to tol within maxiter iterations.
    relres : float
        Residual error at the end of the CG method. Computed using max norm.
    iter : int
        The iteration number at which tol was achieved. Relevant only if
        flag=0.
    absres : float
        The absolute error at the end of the CG method. Relevant only if RefY
        was given as parameter. Computed using max norm.
    """
    # Check the input and initialize flags and default parameters
    # Flag is 1 if the reference untransformed matrix RefY is given and 0 otherwise
    ref_given = RefY is not None

    if params is None:
        params = []

    # Initialize convergence flag. If the routine will detect that the CG method converged, this flag
    # will be set to 0 to represent convergence. By default it assumes that the CG did not converge.
    flag = 1

    # Set flag to suppress output if more than one output is expected.
    suppress_output = False
    if not verbose:
        suppress_output = True

    # iter holds the iteration in which CG converged to tol.
    iter_converged = -1

    # Initialization
    xk = initial_guess
    temp = PtP(xk, *params)
    gk = temp - X
    pk = -gk
    dk = np.sum(np.abs(gk.flatten())**2)

    # Conjugate gradient iteration
    j = 2
    done = False

    perr = 0
    xerr = 0

    while (j <= maxiter) and (not done):
        perr = np.sum(np.abs(pk.flatten())**2)
        if ref_given:  # If reference matrix is given compute absolute error
            xerr = np.max(np.abs(RefY - xk))

        if (not suppress_output) and flag:
            print(f'Iteration {j-1}:  Gradient norm={perr:.7e}', end='')
            print(f'\t Residual error={dk:.7e}', end='')
            if ref_given:
                print(f'\t Absolute error={xerr:.7e}', end='')
            print()

        if perr <= tol:
            iter_converged = j - 1  # CG converged at previous iteration
            flag = 0
            done = True

        if perr > tol:
            hk = PtP(pk, *params)
            # In numpy, dot product of flattened arrays
            tk = dk / np.dot(pk.flatten(), hk.flatten()
                             )  # line search parameter
            xk = xk + tk * pk       # update approximate solution
            gk = gk + tk * hk       # update gradient
            temp = np.sum(np.abs(gk.flatten())**2)
            bk = temp / dk
            dk = temp
            pk = -gk + bk * pk       # update search direction

        j = j + 1

    relres = perr
    absres = None
    if ref_given:
        absres = xerr

    return xk, flag, relres, iter_converged, absres if ref_given else None


#####################################
# Coordinate transformation functions
#####################################

def low_idx(n):
    """
    Return the minimal index for an aliased sequence of length n.
    
    Parameters:
    -----------
    n : int
        The length of the indexed sequence.
    
    Returns:
    --------
    int
        The minimal index for the aliased sequence.
    
    Examples:
    ---------
    For n=5, the indices of the aliased sequence are -2 -1 0 1 2.
    Hence, lowIdx(5) = -2.
    
    For n=4, the indices of the aliased sequence are -2 -1 0 1.
    Hence, lowIdx(4) = -2.
    """
    return int(-np.fix(n/2))


def hi_idx(n):
    """
    Returns the maximal index for an aliased sequence of length n.
    
    Parameters:
    -----------
    n : int
        The length of the indexed sequence.
    
    Returns:
    --------
    int
        The maximal index for the aliased sequence.
    
    Examples:
    ---------
    For n=5, the indices of the aliased sequence are -2 -1 0 1 2.
    Hence, hiIdx(5) = 2.
    
    For n=4, the indices of the aliased sequence are -2 -1 0 1.
    Hence, hiIdx(4) = 1.
    """
    return int(np.fix((n-0.5)/2))


def to_unaliased_idx(idx, n):
    """
    Converts an index from the range -n/2...n/2-1 to an index in the range 
    0...n-1. Both odd and even values of n are handled.
    
    Parameters:
    -----------
    idx : int
        An index from the range -n/2...n/2-1.
    n : int
        The range of indices.
    
    Returns:
    --------
    int
        The index "idx" scaled to the range 0...n-1.
    
    Examples:
    ---------
    For n = 5 and idx = -1:
    toUnaliasedIdx(-1, 5) will return 2:
        -2 -1  0  1  2
            ^
        the index of -1 is 1 if scaled to 1...n.
    """
    return int(idx + np.floor(n/2))


def to_unaliased_coord(idxs, N):
    """
    Converts indices from the range -n/2...n/2-1 to indices in the range 
    0...n-1. Both odd and even values of n are handled.

    This function accepts a vector of aliased indices (aCoord) and a vector 
    of ranges (N) from which the indices are taken. It converts each aliased 
    index aCoord[k] into an unaliased index uCoord[k] in the range 
    0...N[k]-1. If the vector of ranges (N) is a scalar, then the function 
    assumes that all indices are taken from the same range N.
    This allows calling the function on a vector of indices:
        to_unaliased_coord([-1, 1, 2], 5)
    instead of:
        to_unaliased_coord([-1, 1, 2], [5, 5, 5])

    Parameters:
        aCoord (array-like): Vector of aliased indices. Must be a 1-D row vector.
        N (array-like or int): Vector that contains the range of each index. 
                               Must be a 1-D row vector or a scalar. If N is a 
                               scalar, it is used for all coordinates.
    Returns:
        numpy.ndarray: Vector of unaliased indices.

    Notes:
        If N is not a scalar, the vectors aCoord and N must have the same length.
    """
    uCoord = np.zeros_like(idxs, dtype=np.float64)
    if np.isscalar(N):
        N = np.ones_like(idxs)*N

    for k in range(len(idxs)):
        uCoord[k] = to_unaliased_idx(idxs[k], N[k])

    return uCoord

################
# Test functions
################

@pytest.mark.parametrize("n", [5, 21, 32, 44])
def test_fft(n):
    eps = 1e-12
    print(f'Testing FFT routines with n={n}')
    x = np.random.rand(n)

    # Basic 1D tests - compare against direct computation
    cdftx = cdft(x)
    cfftx = cfft(x)
    frftx = frft(x, 1)
    cfrftx = cfrft(x, 1)
    icfftx = icfft(x)
    icdftx = icdft(x)
    assert np.linalg.norm(cdftx - cfftx) / \
        np.linalg.norm(cdftx) < eps, "cfft NOT OK"
    assert np.linalg.norm(cdftx - frftx) / \
        np.linalg.norm(cdftx) < eps, "frft NOT OK"
    assert np.linalg.norm(frftx - cfrftx) / \
        np.linalg.norm(frftx) < eps, "cfrft NOT OK"
    assert np.linalg.norm(icdftx - icfftx) / \
        np.linalg.norm(icdftx) < eps, "icfft NOT OK"

    # Compare foward inverse
    x2 = icfft(cfft(x))
    assert np.linalg.norm(
        x2 - x)/np.linalg.norm(x) < eps, "cfft forward backward NOT OK"

    # Test cfrft
    frftx = frft(x, 0.5)
    cfrftx = cfrft(x, 0.5)
    assert np.linalg.norm(frftx - cfrftx) / \
        np.linalg.norm(frftx) < eps, "cfrft NOT OK"

    frftx = frft(x, 0.3)
    cfrftx = cfrft(x, 0.3)
    assert np.linalg.norm(frftx - cfrftx) / \
        np.linalg.norm(frftx) < eps, "cfrft NOT OK"

    # 2D tests
    x = np.random.rand(n, n)
    cdft2x = cdft2(x)
    cfft2x = cfft2(x)
    assert np.linalg.norm(cdft2x - cfft2x) / \
        np.linalg.norm(cdft2x) < eps, "cfft NOT OK"

    y = cfft2(x)
    icfft2x = icfft2(y)
    assert np.linalg.norm(x - icfft2x)/np.linalg.norm(x) < eps, "frft NOT OK"


@pytest.mark.parametrize("n", [4, 20, 32])
def test_ppft2(n):
    eps = 1e-12
    print(f'Testing ppft2 (forward PPFT) with n={n}')
    im = np.random.rand(n, n)

    pp1_ref, pp2_ref = slow_ppft2(im)
    pp1, pp2 = ppft2(im)
    err1 = np.linalg.norm(pp1 - pp1_ref)/np.linalg.norm(pp1_ref)
    err2 = np.linalg.norm(pp2 - pp2_ref)/np.linalg.norm(pp2_ref)
    assert err1 < eps, "ppft2 NOT OK"
    assert err2 < eps, "ppft2 NOT OK"


@pytest.mark.parametrize("n", [4, 20, 100])
def test_ippft2(n):
    eps = 1e-6
    print(f'Testing ippft2 (inverse PPFT) with n={n}')
    im = np.random.rand(n, n)

    pp1, pp2 = ppft2(im)
    imr, *_ = ippft2(pp1, pp2, tol=1.0e-12, maxiter=50, verbose=False)
    err = np.linalg.norm(imr - im)/np.linalg.norm(im)
    assert  err < eps, "ippft2 NOT OK"

@pytest.mark.parametrize("n", [4, 20, 32])
def test_radon2(n):
    eps = 1e-12
    print(f'Testing radon2 (forward radon) with n={n}')
    im = np.random.rand(n, n)

    res1_ref, res2_ref = slow_radon2(im)
    res1, res2 = radon2(im)
    err1 = np.linalg.norm(res1 - res1_ref)/np.linalg.norm(res1_ref) 
    err2 = np.linalg.norm(res2 - res2_ref)/np.linalg.norm(res2_ref) 
    assert err1 < eps, "radon2 NOT OK"
    assert err2 < eps, "radon2 NOT OK"
    

@pytest.mark.parametrize("n", [4, 20, 100])
def test_iradon2(n):
    eps = 1e-6
    print(f'Testing iradon2 (inverse radon) with n={n}')
    im = np.random.rand(n,n)
    [res1,res2]=radon2(im)
    imr, *_ = iradon2(res1, res2, tol=1.0e-12, maxiter=50, verbose=False)
    err = np.linalg.norm(imr - im)/np.linalg.norm(im)
    assert  err < eps, f"iradon2 NOT OK"
    
def generate_symmetric_matrix_with_condition(n, condition_number, seed=None):
    """
    Generate a random symmetric matrix with a specified condition number.
    
    Parameters:
    -----------
    n : int
        Size of the square matrix (n x n)
    condition_number : float
        Desired condition number of the matrix
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    A : ndarray
        Random symmetric matrix with the specified condition number
    actual_condition : float
        The actual condition number of the generated matrix (should be close to the requested value)
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Create a random orthogonal matrix Q via QR decomposition
    # This will be used to control the eigenvectors
    random_matrix = np.random.randn(n, n)
    Q, _ = np.linalg.qr(random_matrix)

    # Step 2: Create eigenvalues with the desired condition number
    # The condition number is the ratio of largest to smallest eigenvalue
    # We'll create eigenvalues linearly spaced between 1 and condition_number
    eigenvalues = np.linspace(1, condition_number, n)

    # Step 3: Form the matrix A = Q * diag(eigenvalues) * Q^T
    # This creates a symmetric matrix with our desired eigenvalues
    diagonal_matrix = np.diag(eigenvalues)
    A = Q @ diagonal_matrix @ Q.T

    # Ensure the matrix is perfectly symmetric (eliminate floating-point errors)
    A = (A + A.T) / 2

    # Calculate the actual condition number
    actual_condition = np.linalg.cond(A)

    return A, actual_condition


if __name__ == "__main__":
    pytest.main([__file__])
