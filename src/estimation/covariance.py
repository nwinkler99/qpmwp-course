############################################################################
### QPMwP - COVARIANCE
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# # Standard library imports
from typing import Union, Optional

# Third party imports
import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf, MinCovDet

# TODO:

# [ ] Add covariance functions:
#    [ ] cov_linear_shrinkage
#    [ ] cov_nonlinear_shrinkage
#    [ ] cov_factor_model
#    [ ] cov_robust
#    [ ] cov_ewma (expoential weighted moving average)
#    [ ] cov_garch
#    [ ] cov_dcc (dynamic conditional correlation)
#    [ ] cov_pc_garch (principal components garch)
#    [ ] cov_ic_garch (independent components analysis)
#    [ ] cov_constant_correlation


# [ ] Add helper methods:
#    [x] is_pos_def
#    [ ] is_pos_semidef
#    [ ] is_symmetric
#    [ ] is_correlation_matrix
#    [ ] is_diagonal
#    [ ] make_symmetric
#    [x] make_pos_def
#    [ ] make_correlation_matrix (from covariance matrix)
#    [ ] make_covariance_matrix (from correlation matrix)





class CovarianceSpecification(dict):

    def __init__(self,
                 method='pearson',
                 check_positive_definite=False,
                 **kwargs):
        super().__init__(
            method=method,
            check_positive_definite=check_positive_definite,
        )
        self.update(kwargs)


class Covariance:

    def __init__(self,
                 spec: Optional[CovarianceSpecification] = None,
                 **kwargs):
        # allow passing span for EWMA via kwargs
        self.spec = CovarianceSpecification() if spec is None else spec
        self.spec.update(kwargs)
        self._matrix: Union[pd.DataFrame, np.ndarray, None] = None

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        if isinstance(value, CovarianceSpecification):
            self._spec = value
        else:
            raise ValueError(
                'Input value must be of type CovarianceSpecification.'
            )
        return None

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if isinstance(value, (pd.DataFrame, np.ndarray)):
            self._matrix = value
        else:
            raise ValueError(
                'Input value must be a pandas DataFrame or a numpy array.'
            )
        return None

    def estimate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        inplace: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray, None]:

        estimation_method = self.spec['method']
        # retrieve span if set, else default
        span = self.spec.get('span', 252)
        bandwidth = self.spec.get('bandwidth', None)
        clip = self.spec.get('clip', None)
        print(f'Covariance estimation method: {estimation_method}')

        if clip:
            X = X.clip(lower=clip[0], upper=clip[1])

        if estimation_method == 'pearson':
            cov_matrix = cov_pearson(X=X)

        elif estimation_method == 'ledoit':
            cov_matrix = cov_ledoit(X=X)

        elif estimation_method == 'ledoit':
            cov_matrix = cov_ledoit(X=X)

        elif estimation_method == 'mcd':
            cov_matrix = cov_mcd(X=X)

        elif estimation_method == 'ewma_ledoit':
            cov_matrix = cov_ewma_ledoit(X=X, span=span)

        elif estimation_method == 'ewma_ledoit_mix':
            span1 = self.spec.get('span1', 0.5*252)
            span2 = self.spec.get('span2', 3*252)
            short_weight = self.spec.get('short_weight', 0.5)
            long_weight = self.spec.get('long_weight', 0.5)
            cov_matrix = short_weight*cov_ewma_ledoit(X=X, span=span1) + long_weight*cov_ewma_ledoit(X=X, span=span2)

        elif estimation_method == 'nls':
            cov_matrix = cov_nonlinear_shrink(X=X, bandwidth=bandwidth)

        else:
            raise ValueError(
                'Estimation method not recognized.'
            )

        if self.spec.get('check_positive_definite'):
            if not is_pos_def(cov_matrix):
                cov_matrix = make_pos_def(cov_matrix)

        if inplace:
            self.matrix = cov_matrix
            return None
        else:
            return cov_matrix







# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------
from sklearn.impute import SimpleImputer


def _impute(X, strategy='mean'):
    """Return numpy array with NaNs imputed by column mean/median."""
    imputer = SimpleImputer(strategy=strategy)               # :contentReference[oaicite:0]{index=0}
    return imputer.fit_transform(X)

def cov_ledoit(X, strategy='mean'):
    arr = X.values if isinstance(X, pd.DataFrame) else X
    arr_filled = _impute(arr, strategy=strategy)
    lw = LedoitWolf()
    lw.fit(arr_filled)
    cov = lw.covariance_
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(cov, index=X.columns, columns=X.columns)
    return cov

def cov_mcd(X, strategy='mean'):
    arr = X.values if isinstance(X, pd.DataFrame) else X
    arr_filled = _impute(arr, strategy=strategy)
    mcd = MinCovDet()
    mcd.fit(arr_filled)
    cov = mcd.covariance_
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(cov, index=X.columns, columns=X.columns)
    return cov

def cov_pearson(X:  Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    if isinstance(X, pd.DataFrame):
        covmat = X.cov()
    else:
        covmat = np.cov(X, rowvar=False)
    return covmat

def cov_ewma_ledoit(
    X: pd.DataFrame,
    span: int = 252,
    impute_strategy: str = 'mean'
) -> pd.DataFrame:
    """
    EWMA‑Kovarianz + Ledoit–Wolf‑Shrinkage.

    Args:
      X           : DataFrame (Index=time, Columns=assets) of returns.
      span        : EWMA‑Fensterlänge in Tagen.
      impute_strategy: 'mean'|'zero' for NaN‑Imputation.

    Returns:
      shrunk covariance DataFrame.
    """
    # 1) Fehlende Werte füllen
    arr = X.values
    if impute_strategy == 'mean':
        col_means = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_means, inds[1])
    elif impute_strategy == 'zero':
        arr = np.nan_to_num(arr, 0.0)
    
    # 2) EWMA‑Gewichte
    T, p = arr.shape
    λ = (span - 1) / span
    # weights: λ^(T-1), …, λ^0
    w = λ ** np.arange(T-1, -1, -1)
    w = w / w.sum()
    # gewichtetes Mittel
    mean_w = w @ arr
    A = arr - mean_w  # zentrieren
    # EWMA‑Kovarianz
    S_ewma = (A.T * w) @ A

    # 3) Ledoit–Wolf auf S_ewma anwenden
    # wir ermitteln δ mit sklearn, indem wir X mit sqrt(w)-Gewichten neu skalieren:
    Xw = A * np.sqrt(w)[:, None]
    lw = LedoitWolf().fit(Xw)
    δ = lw.shrinkage_
    mu = np.trace(S_ewma) / p

    Σ_shrunk = (1 - δ) * S_ewma + δ * mu * np.eye(p)

    # zurück zu DataFrame
    return pd.DataFrame(Σ_shrunk, index=X.columns, columns=X.columns)

from scipy.stats import gaussian_kde

def cov_nonlinear_shrink(
    X: pd.DataFrame,
    bandwidth: float = None,
    impute_strategy: str = 'mean'
) -> pd.DataFrame:
    """
    Non‑linear shrinkage of the covariance:
      1) fill NaNs
      2) sample covariance S
      3) eigen‑decompose S = V Λ V'
      4) smooth Λ via a kernel smoother
      5) rebuild Σ_nl = V Λ_shrunk V'

    Args:
      X         : DataFrame (T×N) of returns
      bandwidth : kernel bandwidth (if None, Scott’s rule)
      impute_strategy: 'mean' or 'zero' for NaN imputation

    Returns:
      Σ_nl DataFrame (N×N)
    """
    # 1) impute
    arr = X.values.copy()
    if impute_strategy == 'mean':
        col_means = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_means, inds[1])
    else:
        arr = np.nan_to_num(arr, 0.0)

    # 2) sample covariance
    # (unbiased: rowvar=False divides by T-1)
    S = np.cov(arr, rowvar=False)

    # 3) eigen‑decomposition
    eigvals, eigvecs = np.linalg.eigh(S)
    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # 4) kernel‑smooth the eigenvalues
    # treat eigvals as data, estimate density, then smoothed value = conditional expectation
    # we use gaussian_kde to get a smooth pdf; then for each λ_i compute weighted average
    kde = gaussian_kde(eigvals, bw_method=bandwidth)
    # weights_ij = K(λ_j - λ_i)
    # for efficiency we can approximate: λ_shrunk_i = ∫ x K(x-λ_i) f̂(x) dx / ∫ K(x-λ_i) f̂(x) dx
    grid = np.linspace(eigvals.min(), eigvals.max(), 200)
    pdf_grid = kde(grid)
    λ_shrunk = np.empty_like(eigvals)
    for i, λ in enumerate(eigvals):
        # kernel weights on the grid centered at λ
        w = np.exp(-0.5 * ((grid - λ) / kde.factor)**2)
        λ_shrunk[i] = (w * grid * pdf_grid).sum() / (w * pdf_grid).sum()

    # ensure positivity
    λ_shrunk = np.clip(λ_shrunk, 1e-8, None)

    # 5) reconstruct
    Σ_nl = (eigvecs * λ_shrunk) @ eigvecs.T

    return pd.DataFrame(Σ_nl, index=X.columns, columns=X.columns)



def is_pos_def(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def make_pos_def(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_pos_def(A3):
        return A3

    k = 1
    while not is_pos_def(A3):
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

