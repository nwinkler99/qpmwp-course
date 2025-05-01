from typing import Optional, Union
from scipy.optimize import minimize_scalar
import numpy as np
import pandas as pd
import copy
from optimization.optimization import Optimization, Constraints, Covariance, ExpectedReturn, Objective, OptimizationData
from estimation.black_litterman import generate_views_from_scores, bl_posterior_mean
from abc import ABC, abstractmethod 

class Objective():

    '''
    A class to handle the objective function of an optimization problem.

    Parameters:
    kwargs: Keyword arguments to initialize the coefficients dictionary. E.g. P, q, constant.
    '''

    def __init__(self, **kwargs):
        self.coefficients = kwargs

    @property
    def coefficients(self) -> dict:
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value: dict) -> None:
        if isinstance(value, dict):
            self._coefficients = value
        else:
            raise ValueError('Input value must be a dictionary.')
        return None




class OptimizationParameter(dict):

    '''
    A class to handle optimization parameters.

    Parameters:
    kwargs: Additional keyword arguments to initialize the dictionary.
    '''

    def __init__(self, **kwargs):
        super().__init__(
            solver_name = 'cvxopt',
        )
        self.update(kwargs)



class Optimization(ABC):

    '''
    Abstract base class for optimization problems.

    Parameters:
    params (OptimizationParameter): Optimization parameters.
    kwargs: Additional keyword arguments.
    '''

    def __init__(self,
                 params: Optional[OptimizationParameter] = None,
                 constraints: Optional[Constraints] = None,
                 **kwargs):
        self.params = OptimizationParameter() if params is None else params
        self.params.update(**kwargs)
        self.constraints = Constraints() if constraints is None else constraints
        self.objective: Objective = Objective()
        self.results = {}

    @abstractmethod
    def set_objective(self, optimization_data: OptimizationData) -> None:
        raise NotImplementedError(
            "Method 'set_objective' must be implemented in derived class."
        )

    @abstractmethod
    def solve(self) -> None:

        # TODO:
        # Check consistency of constraints
        # self.check_constraints()

        # Get the coefficients of the objective function
        obj_coeff = self.objective.coefficients
        if 'P' not in obj_coeff.keys() or 'q' not in obj_coeff.keys():
            raise ValueError("Objective must contain 'P' and 'q'.")

        # Ensure that P and q are numpy arrays
        obj_coeff['P'] = to_numpy(obj_coeff['P'])
        obj_coeff['q'] = to_numpy(obj_coeff['q'])

        self.solve_qpsolvers()
        return None

    def solve_qpsolvers(self) -> None:

        self.model_qpsolvers()
        self.model.solve()

        solution = self.model.results['solution']
        status = solution.found
        ids = self.constraints.ids
        # weights = pd.Series(solution.x[:len(ids)] if status else [None] * len(ids),
        #                     index=ids)
        weights = pd.Series(solution.x[:len(ids)], index=ids)

        self.results.update({
            'weights': weights.to_dict(),
            'status': status,
        })

        return None

    def model_qpsolvers(self) -> None:

        # constraints
        constraints = self.constraints
        GhAb = constraints.to_GhAb()
        lb = constraints.box['lower'].to_numpy() if constraints.box['box_type'] != 'NA' else None
        ub = constraints.box['upper'].to_numpy() if constraints.box['box_type'] != 'NA' else None

        # Create the optimization model as a QuadraticProgram
        self.model = QuadraticProgram(
            P=self.objective.coefficients['P'],
            q=self.objective.coefficients['q'],
            G=GhAb['G'],
            h=GhAb['h'],
            A=GhAb['A'],
            b=GhAb['b'],
            lb=lb,
            ub=ub,
            solver_settings=self.params)

        # Deal with turnover constraint or penalty (cannot have both)
        turnover_penalty = self.params.get('turnover_penalty')

        ## Turnover constraint
        tocon = self.constraints.l1.get('turnover')
        if tocon is not None and (turnover_penalty is None or turnover_penalty == 0):
            x_init = np.array(list(tocon['x0'].values()))
            self.model.linearize_turnover_constraint(x_init=x_init,
                                                     to_budget=tocon['rhs'])

        ## Turnover penalty
        if turnover_penalty is not None and turnover_penalty > 0:
            x_init = pd.Series(self.params.get('x_init')).to_numpy()
            self.model.linearize_turnover_objective(x_init=x_init,
                                                    turnover_penalty=turnover_penalty)

        return None



class EmptyOptimization(Optimization):
    '''
    Placeholder class for an optimization.
    This class is intended to be a placeholder and should not be used directly.
    '''

    def set_objective(self, optimization_data: OptimizationData) -> None:
        raise NotImplementedError(
            'EmptyOptimization is a placeholder and does not implement set_objective.'
        )

    def solve(self) -> None:
        raise NotImplementedError(
            'EmptyOptimization is a placeholder and does not implement solve.'
        )


class MaxSharpe(Optimization):

    def __init__(self,
                 constraints: Optional[Constraints] = None,
                 covariance: Optional[Covariance] = None,
                 expected_return: Optional[ExpectedReturn] = None,
                 turnover_penalty: Optional[float] = None,
                 risk_aversion: float = 1.0,
                 iterations: int = 10, 
                 **kwargs) -> None:
        super().__init__(
            constraints=constraints,
            **kwargs,
        )
        self.covariance = Covariance() if covariance is None else covariance
        self.expected_return = ExpectedReturn() if expected_return is None else expected_return
        self.params['turnover_penalty'] = turnover_penalty
        self.params['risk_aversion'] = risk_aversion
        self.iterations = iterations

    def set_objective(self, optimization_data: OptimizationData) -> None:
        self.data = optimization_data
        X = optimization_data['return_series']
        self.cov = self.covariance.estimate(X=X, inplace=False)
        self.mu  = self.expected_return.estimate(X=X, inplace=False)
        self.objective = Objective(
            q = -1.0 * self.mu,
            P = 2.0 * self.cov,
        )
        self.base_P = copy.deepcopy(2.0 * self.cov)

    def solve(self) -> None:
        parent_solve = super(MaxSharpe, self).solve

        def _neg_sharpe(lam: float) -> float:
            # scale P and re-solve
            self.params['risk_aversion'] = lam
            self.objective.coefficients["P"] = self.base_P * lam
            parent_solve()
            w = np.array(list(self.results["weights"].values()))
            sr = (self.mu @ w) / np.sqrt(w @ self.cov @ w)
            return -sr

        # one-line search over [1e-2,1e2]
        res = minimize_scalar(
            _neg_sharpe,
            bounds=(1e-2, 1e2),
            method="bounded",
            options={"maxiter": self.iterations, "xatol": 1e-3}
        )

        # unpack results
        best_lambda = res.x
        best_sharpe = -res.fun
        # re-solve once at the optimum to populate weights
        self.objective.P = self.base_P * best_lambda
        parent_solve()

        self.results = {
            "weights":       self.results["weights"],
            "best_sharpe":   best_sharpe,
            "risk_aversion": best_lambda,
            "status":        True,
        }
        return None
    

class BlackLittermanMS(Optimization):

    def __init__(self,
                 fields: list[str],
                 covariance: Optional[Covariance] = None,
                 risk_aversion: float = 1,
                 tau_psi: float = 1,
                 tau_omega: float = 1,
                 view_method: str = 'absolute',
                 scalefactor: int = 1,
                 iterations: int = 5,
                 **kwargs) -> None:
        super().__init__(
            fields=fields,
            risk_aversion=risk_aversion,
            tau_psi=tau_psi,
            tau_omega=tau_omega,
            view_method=view_method,
            scalefactor=scalefactor,
            **kwargs
        )
        self.covariance = Covariance() if covariance is None else covariance
        self.iterations = iterations
    def set_objective(self, optimization_data: OptimizationData) -> None:
        '''
        Sets the objective function for the optimization problem.
        
        Parameters:
        training_data: Training data which must contain 
            return series (to compute the covariances) and scores.
        '''
        self.data = optimization_data
        # Retrieve configuration parameters from the params attribute
        fields = self.params.get('fields')
        risk_aversion = self.params.get('risk_aversion')
        tau_psi = self.params.get('tau_psi')
        tau_omega = self.params.get('tau_omega')
        view_method = self.params.get('view_method')
        scalefactor = self.params.get('scalefactor')

        # Calculate the covariance matrix
        self.covariance.estimate(
            X=optimization_data['return_series'],
            inplace=True,
        )

        # Extract benchmark weights
        cap_weights = optimization_data['cap_weights']

        # # Alternatively, calculate minimum tracking error portfolio
        # optim = LeastSquares(
        #     constraints = self.constraints,
        #     solver_name = self.params.get('solver_name'),
        # )
        # optim.set_objective(optimization_data=optimization_data)
        # optim.solve()
        # cap_weights = pd.Series(optim.results['weights'])

        # Implied expected return of benchmark
        mu_implied = risk_aversion * self.covariance.matrix @ cap_weights

        # Extract scores
        scores = optimization_data['scores'][fields]

        # Construct the views
        P_tmp = {}
        q_tmp = {}
        for col in scores.columns:
            P_tmp[col], q_tmp[col] = generate_views_from_scores(
                scores=scores[col],
                mu_implied=mu_implied,
                method=view_method,
                scalefactor=scalefactor,
            )

        P = pd.concat(P_tmp, axis=0)
        q = pd.concat(q_tmp, axis=0)

        # Define the uncertainty of the views
        Omega = pd.DataFrame(
            np.diag([tau_omega] * len(q)),
            index=q.index,
            columns=q.index
        )
        Psi = self.covariance.matrix * tau_psi

        # Compute the posterior expected return vector
        mu_posterior = bl_posterior_mean(
            mu_prior=mu_implied,
            P=P,
            q=q,
            Psi=Psi,
            Omega=Omega,
        )

        self.objective = Objective(
            q = mu_posterior * (-1),
            P = self.covariance.matrix * risk_aversion * 2,
            mu_implied = mu_implied,
            mu_posterior = mu_posterior,
        )


    def solve(self) -> None:
        parent_solve = super(MaxSharpe, self).solve

        def _neg_sharpe(lam: float) -> float:
            # scale P and re-solve
            self.params['risk_aversion'] = lam
            self.set_objective(optimization_data=self.data)
            parent_solve()
            w = np.array(list(self.results["weights"].values()))
            sr = (self.mu @ w) / np.sqrt(w @ self.cov @ w)
            return -sr

        # one-line search over [1e-2,1e2]
        res = minimize_scalar(
            _neg_sharpe,
            bounds=(1e-2, 1e2),
            method="bounded",
            options={"maxiter": self.iterations, "xatol": 1e-3}
        )

        # unpack results
        best_lambda = res.x
        best_sharpe = -res.fun
        # re-solve once at the optimum to populate weights
        self.objective.P = self.base_P * best_lambda
        parent_solve()

        self.results = {
            "weights":       self.results["weights"],
            "best_sharpe":   best_sharpe,
            "risk_aversion": best_lambda,
            "status":        True,
        }
        return None


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
        print(f'Covariance estimation method: {estimation_method}')
        if estimation_method == 'pearson':
            cov_matrix = cov_pearson(X=X)

        elif estimation_method == 'ledoit':
            cov_matrix = cov_ledoit(X=X)  # Placeholder for ledait_shrinkage

        elif estimation_method == 'mcd':
            cov_matrix = cov_mcd(X=X)  # Placeholder for ledait_shrinkage

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

