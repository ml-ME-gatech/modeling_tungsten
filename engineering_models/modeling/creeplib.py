import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, PredictionResults
from statsmodels.tools import add_constant
from abc import ABC,abstractmethod
from scipy.stats.distributions import t as t_dist
import sys
sys.path.append('../')
from common_util import ProjectPaths
from scipy.optimize import minimize_scalar
from numpy.linalg import lstsq
from typing import Tuple
paths = ProjectPaths()

class LarsonMiller(ABC):

    r""" 
    Base class for Larson Miller model. The equation is:
    
    $$
    Z = T*(C + log(t))
    S_t = f(Z;\theta)
    $$
    
    where $f$ is an arbitary function and $\theta$ are function parameters,and $C$ is the so-called
    Larson-Miller Parameter.
    """

    def __init__(self, C: float = None,
                       results: OLS = None):
        self.C = C
        self.SE_C = None
        self.results = results
        self.prediction_results = None

    @staticmethod
    def lmp(t: np.ndarray,T: np.ndarray,C: float) -> np.ndarray:
        return T*(C + np.log(t))
    
    def lmp_confint(self,t: np.ndarray,T: np.ndarray,alpha: float = 0.05) -> np.ndarray:
        """
        Compute the confidence intervals for the Larson Miller parameter.

        Parameters
        ----------
        t : np.ndarray
            The time to failure.
        T : np.ndarray
            The temperature.
        C : float
            The Larson Miller parameter.
        alpha : float, optional
            The significance level. Default is 0.05.

        Returns
        -------
        np.ndarray
            The confidence intervals for the Larson Miller parameter.
        """
        t = t_dist.ppf(1 - alpha / 2, len(t) - 1)
        return np.array([self.lmp(t,T,self.C- t * self.SE_C),self.lmp(t,T,self.C + t * self.SE_C)]).T
    
    def get_lmp(self,t: np.ndarray,
                     T: np.ndarray,
                     conf_int: bool = False,
                     alpha: float = 0.05) -> np.ndarray:
        r"""
        Compute the Larson Miller parameter.

        Parameters
        ----------
        t : np.ndarray
            The time to failure.
        T : np.ndarray
            The temperature.

        Returns
        -------
        np.ndarray
            The Larson Miller parameter.
        """
        if conf_int:
            return self.lmp(t,T,self.C),self.lmp_confint(t,T,alpha=alpha)
        return self.lmp(t,T,self.C)
    
    @abstractmethod
    def predict_from_lmp(self, Z: np.ndarray,
                               conf_int: float = False,
                               alpha: float = None) -> np.ndarray: 

        
        pass


    @abstractmethod
    def fit(self,t: np.ndarray,
                 T: np.ndarray,
                 S_t: np.ndarray,
                 **kwargs):

        pass

    @abstractmethod
    def param_confint(self,alpha: float = 0.05) -> np.ndarray:
        """
        Compute the confidence intervals for the parameters.

        Parameters
        ----------
        alpha : float, optional
            The significance level. Default is 0.05.

        Returns
        -------
        np.ndarray
            The confidence intervals for the parameters.
        """
        pass

    @abstractmethod
    def get_prediction(self,t: np.ndarray,T: np.ndarray) -> PredictionResults:      
        pass


class LarsonMillerSLR(LarsonMiller):
    r"""
    Fit the larson miller model the equation: 

    $$
    Z = T (C + log(t))
    S_t = \beta_0 + \beta_1 Z = \beta_0 + \beta_1 C T + \beta_1 T log(t)
    $$

    Approximate the variance in $\hat{C}$ using the delta method, that is: 

    $$
    \mathbb{V}\text{ar} \approx & \frac{1}{\hat{\beta}^2_1} \sigma_{\beta_1 C}^2 + 
                                & \frac{\hat{C}^2}{\hat{\beta}_1^2} \sigma_{\beta_1}^2 - 
                                & 2 \frac{\hat{C}}{\hat{\beta}_1^2} \sigma_{ab}
    $$
    """

    def fit(self,t: np.ndarray,
            T: np.ndarray,
            S_t: np.ndarray,
            **kwargs) -> None:
        r"""
        Fit the Larson Miller model using ordinary least squares regression.

        Parameters
        ----------
        t : np.ndarray
            The time to failure.
        T : np.ndarray
            The temperature.
        S_t : np.ndarray
            The stress at time t.
        C : float, optional
            The Larson Miller parameter. If not provided, it will be estimated from the data.
        deg : int, optional
            The degree of the polynomial to fit. Default is 1 (linear).
        kwargs : additional arguments for OLS.

        Returns
        -------
        OLSResults
            The fitted model results.
        """
        X = add_constant(
            pd.DataFrame(np.array([T.values,np.log(t.values)*T.values]).T, 
            columns = ['Temperature','log(t)*T'])
            )

        self.results = OLS(S_t.values,X).fit(**kwargs)
        self.C = self.results.params['Temperature']/self.results.params['log(t)*T']
        cov = self.results.cov_params().to_numpy()
        self.SE_C = np.sqrt(
            (1/self.results.params['log(t)*T']**2) * cov[1,1] + 
            (self.C**2/self.results.params['log(t)*T']**2) * cov[2,2] - 
            (2*self.C/self.results.params['log(t)*T']**2) * cov[1,2]
        )
        self.mean_T2 = np.mean(T**2)
        return self

    def param_confint(self, alpha = 0.05):
        params = self.results.conf_int(alpha)
        t = t_dist.ppf(1 - alpha / 2, self.results.df_resid)
        C = np.array([self.C - t * self.SE_C, self.C + t * self.SE_C])
        params.loc['beta_1',:] = self.results.params['log(t)*T']
        params.loc['C',:] = C 
        params.drop(index = ['Temperature','log(t)*T'], inplace = True)
        return params



    def lmp_confint(self,t: np.ndarray,T: np.ndarray,alpha: float = 0.05) -> np.ndarray:
        """
        Compute the confidence intervals for the Larson Miller parameter.

        Parameters
        ----------
        t : np.ndarray
            The time to failure.
        T : np.ndarray
            The temperature.
        C : float
            The Larson Miller parameter.
        alpha : float, optional
            The significance level. Default is 0.05.

        Returns
        -------
        np.ndarray
            The confidence intervals for the Larson Miller parameter.
        """
        t = t_dist.ppf(1 - alpha / 2, len(t) - 1)
        return np.array([self.lmp(t,T,self.C- t * self.SE_C),self.lmp(t,T,self.C + t * self.SE_C)]).T
        
    def predict_from_lmp(self, Z: np.ndarray,
                               conf_int: float = False,
                               forecast: bool = False,
                               alpha: float = 0.05) -> np.ndarray: 

        beta_0,beta_1 = self.results.params['const'],self.results.params['log(t)*T']
        S_t = beta_0 + beta_1 * Z
        if conf_int:
            var_St = self.results.bse['const']**2 + \
                (Z**2 * self.results.bse['log(t)*T']**2) + \
                (2 * Z * self.results.cov_params().loc['const','log(t)*T']) + \
                self.mean_T2* self.SE_C**2*self.results.bse['log(t)*T']**2 
            t = t_dist.ppf(1 - alpha / 2, self.results.df_resid)
            if forecast:
                var_St += self.results.mse_resid
            
            return S_t, np.array([S_t - t * np.sqrt(var_St), S_t + t * np.sqrt(var_St)])
        else:
            return S_t
        

    def get_prediction(self, t, T):
        self.prediction_results = self.results.get_prediction(
            add_constant(pd.DataFrame([np.array([T,np.log(t)*T])].T, columns = ['Temperature','log(t)*T']))
        )
        return self.prediction_results
    

class LarsonMillerPowerLaw(LarsonMiller):
    r"""
    Larson-Miller power-law model:

    .. math::
        Z = T \,\bigl(C + \ln(t)\bigr),

        \ln(S_t) = \ln(A_1) + \beta_1 \ln(Z).

    The model is fit by searching over C (scalar) to minimize SSE in the log-log
    regression, i.e. we solve

    .. math::
        \min_C \sum_{i=1}^n \Bigl[\ln(S_{t_i}) - \bigl(\alpha + \beta \ln(Z_i)\bigr)\Bigr]^2,

    where :math:`\alpha = \ln(A_1)` and :math:`\beta = \beta_1`.

    Once the best C is found, we do a final OLS in log-space to get the fitted
    parameters.  A nonparametric bootstrap is then used to estimate parameter
    variances and provide confidence/prediction intervals.

    Parameters
    ----------
    C : float, optional
        Initial guess for the Larson-Miller constant parameter. If None, a default
        is chosen (0.0).
    alpha : float, optional
        If you want to seed the alpha parameter (log(A_1)).
    beta : float, optional
        If you want to seed the beta parameter.
    A_1 : float, optional
        If you want to seed A_1.  If given, overrides alpha if both are present.
    results : OLS, optional
        For consistency with the base class (unused here).

    Attributes
    ----------
    C : float
        Fitted (or user-set) Larson-Miller constant.
    SE_C : float
        An approximate SE for C if you want it, though we rely mainly on bootstrap
        for intervals in this class.
    alpha_ : float
        Fitted intercept in log-space, i.e. alpha_ = log(A_1).
    beta_ : float
        Fitted power exponent in log-space, i.e. beta_ = beta_1.
    A_1_ : float
        Fitted coefficient = exp(alpha_).
    bootstrap_params_ : np.ndarray
        Shape (n_boot, 3). Each row is (alpha, beta, C) from one bootstrap sample.
    """

    def __init__(self,
                 C: float = None,
                 alpha: float = None,
                 beta: float = None,
                 A_1: float = None,
                 results=None):
        super().__init__(C=C, results=results)
        # If user gave alpha/beta/A_1 as direct seeds:
        self.alpha_ = alpha
        self.beta_ = beta
        self.A_1_ = A_1 if A_1 is not None else (np.exp(alpha) if alpha is not None else None)
        self.bootstrap_params_ = None
        self.n_boot = 0
        self.SE_C = None
        self.fitC_ = True  # If True, we re-optimize C in .fit(); otherwise, we keep it fixed.

    @staticmethod
    def _get_sse_for_C(
        C_candidate: float,
        t: np.ndarray,
        T: np.ndarray,
        S_t: np.ndarray
    ) -> float:
        """
        Given a candidate C, compute SSE of log(S_t) ~ alpha + beta*log(Z).
        """
        # Compute Z = T*(C + log(t))
        Z = T * (C_candidate + np.log(t))
        
        # Z must be positive for log(Z) to be valid. If invalid, return large penalty
        if np.any(Z <= 0):
            return 1.0e15  # a large number to discourage this C

        # OLS in log-log space: log(S_t) = alpha + beta log(Z).
        y = np.log(S_t)
        X = add_constant(np.log(Z))  # X[:,0] = 1.0, X[:,1] = log(Z)

        # Fit a quick OLS
        model = OLS(y, X).fit()
        sse = np.sum(model.resid**2)
        return sse

    def fit(
        self,
        t: np.ndarray | pd.Series,
        T: np.ndarray | pd.Series,
        S_t: np.ndarray |  pd.Series,
        n_boot: int = 1000,
        random_state: int = 0,
        method_bounds: Tuple[float, float] = (-10, 100),
        **kwargs
    ) -> "LarsonMillerPowerLaw":
        r"""
        Fit the power-law Larson-Miller model, searching over C to minimize SSE
        in log-space, then storing the final regression.  Perform a bootstrap
        (with replacement) to get parameter distributions.

        Parameters
        ----------
        t : array-like
            Times to failure (must be positive).
        T : array-like
            Temperatures (must be positive).
        S_t : array-like
            Stresses at time t (must be positive).
        n_boot : int, optional
            Number of bootstrap samples. Default = 1000.
        random_state : int, optional
            For reproducible bootstrap sampling.
        method_bounds : (float, float)
            Bounds for the scalar search on C. Adjust as needed.
        kwargs : dict
            Extra arguments (unused, for potential extension).

        Returns
        -------
        self : LarsonMillerPowerLaw
            Fitted model instance.
        """
        t = np.asarray(t)
        T = np.asarray(T)
        S_t = np.asarray(S_t)

        # 1) Possibly find best C via scalar minimization of SSE, if self.fitC_ is True.
        if self.fitC_ and (self.C is None):
            res = minimize_scalar(
                self._get_sse_for_C,
                bounds=method_bounds,
                args=(t, T, S_t),
                method="bounded"
            )
            self.C = res.x
        elif not self.fitC_:
            # If user gave a specific C, or we want to keep it fixed
            pass
        else:
            # If user gave a non-None C but still wants to optimize, you might want
            # to override with your own logic or skip. For now, we do the same as above:
            res = minimize_scalar(
                self._get_sse_for_C,
                bounds=method_bounds,
                args=(t, T, S_t),
                method="bounded"
            )
            self.C = res.x

        # 2) Once we have C, do the final OLS in log-space to get alpha, beta.
        Z = T * (self.C + np.log(t))
        if np.any(Z <= 0):
            raise ValueError(
                "Best-fit C yields non-positive Z for at least one observation. "
                "Try different bounds or check your data."
            )

        y = np.log(S_t)
        X = add_constant(np.log(Z))  # shape (n, 2). col0=1, col1=log(Z)
        final_model = OLS(y, X).fit()
        self.results = final_model

        # Retrieve alpha, beta
        self.alpha_ = final_model.params[0]   # intercept = ln(A_1)
        self.beta_ = final_model.params[1]    # exponent
        self.A_1_ = np.exp(self.alpha_)

        # 3) Bootstrap to get distributions of (alpha, beta, C).
        rng = np.random.default_rng(seed=random_state)
        n = len(t)
        self.n_boot = n_boot
        boot_params = np.zeros((n_boot, 3))  # (alpha, beta, C)

        for b in range(n_boot):
            # Sample w/ replacement
            indices = rng.integers(0, n, size=n)
            t_b = t[indices]
            T_b = T[indices]
            S_b = S_t[indices]

            # Re-optimize C on the bootstrap sample if we are letting it vary:
            if self.fitC_:
                res_b = minimize_scalar(
                    self._get_sse_for_C,
                    bounds=method_bounds,
                    args=(t_b, T_b, S_b),
                    method="bounded"
                )
                C_b = res_b.x
            else:
                C_b = self.C

            # Fit alpha,beta with that C_b
            Z_b = T_b * (C_b + np.log(t_b))
            if np.any(Z_b <= 0):
                boot_params[b, :] = np.nan
                continue

            y_b = np.log(S_b)
            X_b = add_constant(np.log(Z_b))
            model_b = OLS(y_b, X_b).fit()
            alpha_b = model_b.params[0]
            beta_b = model_b.params[1]

            boot_params[b, 0] = alpha_b
            boot_params[b, 1] = beta_b
            boot_params[b, 2] = C_b

        self.SE_C = np.nan
        if np.sum(np.isfinite(boot_params[:,2])) > 1:
            self.SE_C = np.nanstd(boot_params[:, 2])  # if wanted

        self.bootstrap_params_ = boot_params

        return self

    def set_params_from_values(
        self,
        alpha: float,
        beta: float,
        C: float,
        bootstrap_params: np.ndarray = None
    ):
        """
        Directly set the fitted parameters for this model, bypassing internal .fit().
        Optionally also set the (alpha, beta, C) bootstrap samples array.

        Parameters
        ----------
        alpha : float
            ln(A_1)
        beta : float
            exponent
        C : float
            Larson-Miller constant
        bootstrap_params : np.ndarray, shape (n_boot, 3), optional
            Each row = (alpha_b, beta_b, C_b)
        """
        self.alpha_ = alpha
        self.beta_ = beta
        self.C = C
        self.A_1_ = np.exp(alpha)
        self.bootstrap_params_ = bootstrap_params
        self.SE_C = np.std(bootstrap_params[:, 2]) if bootstrap_params is not None else None

    def param_confint(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Compute confidence intervals for (A_1, beta_1, C) using the bootstrap
        percentile method.
        """
        if self.bootstrap_params_ is None:
            raise RuntimeError("Must call fit() or set_params_from_values() with bootstrap samples before param_confint().")

        valid_rows = np.isfinite(self.bootstrap_params_).all(axis=1)
        boot_params = self.bootstrap_params_[valid_rows]
        if len(boot_params) < 10:
            raise RuntimeError("Too few valid bootstrap samples to compute CI.")

        alpha_lower = alpha / 2
        alpha_upper = 1.0 - alpha / 2

        alpha_samples = boot_params[:, 0]
        beta_samples = boot_params[:, 1]
        C_samples = boot_params[:, 2]

        A1_samples = np.exp(alpha_samples)  # A_1 = e^(alpha)

        def pct_bounds(x):
            return np.quantile(x, [alpha_lower, alpha_upper])

        A1_ci = pct_bounds(A1_samples)
        beta_ci = pct_bounds(beta_samples)
        C_ci = pct_bounds(C_samples)

        ci_df = pd.DataFrame(
            index=["A_1", "beta_1", "C"],
            columns=["lower", "upper"],
            data=np.vstack([A1_ci, beta_ci, C_ci])
        )
        return ci_df

    def params(self):
        """Return the point estimates (A_1, beta_1, C)."""
        return pd.Series([self.A_1_, self.beta_, self.C],
                         index=["A_1", "beta_1", "C"])

    def predict_from_lmp(
        self,
        Z: np.ndarray,
        conf_int: bool = False,
        alpha: float = 0.05,
        return_samples = False,
    ):
        """
        Predict S(t) = A_1 * Z^beta at given Z.
        If conf_int=True, compute percentile intervals from .bootstrap_params_.
        """
        S_pred = self.A_1_ * (Z ** self.beta_)
        if (not conf_int) or (self.bootstrap_params_ is None):
            return S_pred

        valid_rows = np.isfinite(self.bootstrap_params_).all(axis=1)
        boot_params = self.bootstrap_params_[valid_rows]
        S_pred_samples = []
        for (alpha_b, beta_b, _) in boot_params:
            S_b = np.exp(alpha_b) * (Z ** beta_b)
            S_pred_samples.append(S_b)
        S_pred_samples = np.vstack(S_pred_samples)  # shape (n_valid, len(Z))

        alpha_lower = alpha/2
        alpha_upper = 1 - alpha/2
        lb = np.quantile(S_pred_samples, alpha_lower, axis=0)
        ub = np.quantile(S_pred_samples, alpha_upper, axis=0)
        if return_samples and not conf_int:
            return S_pred, S_pred_samples
        elif return_samples and conf_int:   
            return S_pred, np.array([lb, ub]),S_pred_samples
        else:
            return S_pred,np.array([lb,ub])

    def get_prediction(self, t: np.ndarray, T: np.ndarray,conf_int = False, alpha = 0.05):
        """
        Compute prediction at the given t, T with the fitted C, returning a dict.
        """
        Z = self.get_lmp(t, T)
        return self.predict_from_lmp(Z,conf_int=conf_int, alpha=alpha)    
    

class LarsonMillerPowerLawMulti(LarsonMiller):
    """
    Larson-Miller model across multiple data sources, all sharing one scalar C,
    but each with its own alpha_i = ln(A_{1,i}) and beta_i.

    The model for data source i is:
        log(S_i) ~ alpha_i + beta_i*log(Z_i),
    where
        Z_i = T_i*(C + ln(t_i))   (same C for all i).
    """

    def __init__(self, 
                 C_fixed: float = None, 
                 C_bootstrap: np.ndarray = None):
        """
        Parameters
        ----------
        C_fixed : float, optional
            If provided, we skip the internal search for C
            and treat this as the final, fixed C in the model.
        C_bootstrap : np.ndarray, optional
            If provided, it should be an array of length n_boot (that you
            will specify at .fit(..., n_boot=...)). Then, during the bootstrap
            loop, we use these pre-specified C values (one per iteration).
            If None, we do the usual behavior (i.e. re-optimize C each iteration
            or use the same fixed C for all iterations).
        """
        self.C_fixed = C_fixed
        self.C_bootstrap = C_bootstrap

        self.C_ = None       # final best/fixed C for the main fit
        self.C = C_fixed
        self.alpha_ = None   # array of alpha_i across sources
        self.beta_ = None    # array of beta_i across sources

        # For bootstrap results:
        # shape (n_boot, 1 + 2*n_sources).
        # Column 0 => C_b, columns 1.. => alpha_i, beta_i pairs
        self.bootstrap_params_ = None
        self.n_boot = 0

        # We'll store the sample sizes for weighting or reference:
        self.data_sizes_ = None
    
    @staticmethod
    def _ols_loglog(y: np.ndarray, x: np.ndarray):
        """
        Quick OLS in log–log space to get (alpha, beta, SSE).

        Model:  log(y) = alpha + beta * log(x)
        Returns alpha, beta, SSE.
        """
        # log transform
        logy = np.log(y)
        logx = np.log(x)

        # X matrix for OLS
        Xmat = np.column_stack([
            np.ones_like(logx),
            logx
        ])
        # Solve for [alpha, beta] in a least-squares sense
        # We can use np.linalg.lstsq or statsmodels if we want:
        coeffs, residuals, _,_= lstsq(Xmat, logy, rcond=None)
        alpha, beta = coeffs
        # If lstsq doesn't return residuals, we can compute them explicitly:
        if len(residuals) == 0:
            # residuals not returned if rank < 2
            resid = logy - Xmat @ coeffs
            sse = np.sum(resid**2)
        else:
            sse = residuals[0]
        return alpha, beta, sse

    @classmethod
    def _get_sse_for_C(cls, C_candidate, t_list, T_list, S_list,weight = False):
        """
        Given a candidate C, sum the SSE (in log–log space) across all data sources.
        For each data source i, we do a separate OLS: log(S_i) ~ alpha_i + beta_i*log(Z_i).
        Then sum the SSE_i for i=1..m.
        """
        total_sse = 0.0
        # Loop over data sources:
        tl = 0
        for t_i, T_i, S_i in zip(t_list, T_list, S_list):
            # Compute Z_i = T_i * (C + ln(t_i))
            # Must ensure positivity inside the ln(t_i) and inside (C + ln(t_i)).
            if np.any(t_i <= 0):
                return 1e15  # penalize invalid
            ln_t_i = np.log(t_i)

            Z_i = T_i * (C_candidate + ln_t_i)

            # If any Z_i <= 0, we must penalize:
            if np.any(Z_i <= 0):
                return 1e15

            # Now do the OLS in log–log space for this source i:
            alpha_i, beta_i, sse_i = cls._ols_loglog(S_i, Z_i)
            total_sse += sse_i*len(S_i) if weight else sse_i
            tl += len(S_i)

        return total_sse/tl if weight else total_sse

    def fit(self, t_list, T_list, S_list,
            method_bounds=(-10, 100),
            n_boot=1000,
            random_state=0,
            weight = False):
        """
        Fit the multi-source Larson-Miller power-law model.

        If C_fixed is None, we do a bounded search for the best C.
        Otherwise, we skip the search and just use C_fixed.

        Then for bootstrap:
          - If C_bootstrap is provided, we use those values of C
            (one per iteration).
          - If C_bootstrap is not provided and C_fixed is not None,
            we re-use that same C for all bootstrap samples
            (no re-optimization).
          - If C_bootstrap is None and C_fixed is None,
            we re-optimize C in each bootstrap iteration.

        Parameters
        ----------
        t_list, T_list, S_list : list of ndarrays
            Data from each source. Must be positive for logs.
        method_bounds : (float, float)
            Bounds for the scalar search on C if we do it.
        n_boot : int
            Number of bootstrap iterations.
        random_state : int
            RNG seed for reproducibility.

        Returns
        -------
        self
        """
        n_sources = len(t_list)
        self.data_sizes_ = [len(t_i) for t_i in t_list]

        # --- 1) Determine final best/fixed C_ for the main fit ---
        if self.C_fixed is not None:
            # Skip searching
            self.C_ = float(self.C_fixed)
        else:
            # Do a bounded search to find best C
            res = minimize_scalar(
                self._get_sse_for_C,
                bounds=method_bounds,
                args=(t_list, T_list, S_list,weight),
                method="bounded"
            )
            self.C_ = res.x
        
        self.C = self.C_  # for consistency with base class

        # --- 2) OLS for each data source with the final C_ ---
        alpha_arr = np.zeros(n_sources)
        beta_arr = np.zeros(n_sources)
        for i, (t_i, T_i, S_i) in enumerate(zip(t_list, T_list, S_list)):
            Z_i = T_i * (self.C_ + np.log(t_i))
            if np.any(Z_i <= 0):
                raise ValueError(f"Final C={self.C_:.4f} yields non-positive Z for source {i}.")
            alpha_i, beta_i, _ = self._ols_loglog(S_i, Z_i)
            alpha_arr[i] = alpha_i
            beta_arr[i]  = beta_i
        self.alpha_ = alpha_arr
        self.beta_  = beta_arr

        # --- 3) Bootstrap ---
        self.n_boot = n_boot
        boot_params = np.zeros((n_boot, 1 + 2*n_sources), dtype=float)
        rng = np.random.default_rng(seed=random_state)

        # If user provided a separate C_bootstrap, check length
        use_custom_C_boot = (self.C_bootstrap is not None)
        if use_custom_C_boot:
            if len(self.C_bootstrap) != n_boot:
                raise ValueError(
                    f"Length of C_bootstrap ({len(self.C_bootstrap)}) != n_boot ({n_boot})."
                )

        for b in range(n_boot):
            # 3A) Resample data for each source
            t_b_list = []
            T_b_list = []
            S_b_list = []
            for t_i, T_i, S_i in zip(t_list, T_list, S_list):
                n_i = len(t_i)
                idx = rng.integers(0, n_i, size=n_i)
                t_b_list.append(t_i[idx])
                T_b_list.append(T_i[idx])
                S_b_list.append(S_i[idx])

            # 3B) Determine C_b for iteration b
            if use_custom_C_boot:
                # We use the user-provided array directly
                C_b = self.C_bootstrap[b]
            else:
                # either re-optimizing each time, or re-using the fixed C
                if self.C_fixed is not None:
                    # we have a single fixed C, so no re-optimization
                    C_b = self.C_
                else:
                    # we do the usual re-optimization
                    res_b = minimize_scalar(
                        self._get_sse_for_C,
                        bounds=method_bounds,
                        args=(t_b_list, T_b_list, S_b_list),
                        method="bounded"
                    )
                    C_b = res_b.x

            # 3C) OLS to get alpha_b_i, beta_b_i for each source
            alpha_b_arr = np.zeros(n_sources)
            beta_b_arr = np.zeros(n_sources)
            invalid = False
            for i2, (tb_i, Tb_i, Sb_i) in enumerate(zip(t_b_list, T_b_list, S_b_list)):
                Zb_i = Tb_i * (C_b + np.log(tb_i))
                if np.any(Zb_i <= 0):
                    invalid = True
                    break
                a_b, be_b, _ = self._ols_loglog(Sb_i, Zb_i)
                alpha_b_arr[i2] = a_b
                beta_b_arr[i2]  = be_b

            if invalid:
                # Store NaNs for this bootstrap row if we got non-positive Z
                boot_params[b, :] = np.nan
            else:
                boot_params[b, 0] = C_b
                col = 1
                for i2 in range(n_sources):
                    boot_params[b, col]   = alpha_b_arr[i2]
                    boot_params[b, col+1] = beta_b_arr[i2]
                    col += 2

        self.bootstrap_params_ = boot_params
        return self
    
    def param_confint(self, alpha=0.05):
        """
        Compute bootstrap percentile confidence intervals for the parameters.

        Returns a DataFrame with:
            Row index = ["C"] + ["A_{1,i}", "beta_i"] for i in each source
            Columns = ["lower", "upper"]

        If too few valid bootstrap samples, raises an error.
        """
        if self.bootstrap_params_ is None:
            raise RuntimeError("Call .fit(...) before .param_confint()")

        # drop rows with any NaN
        valid_rows = np.all(np.isfinite(self.bootstrap_params_), axis=1)
        bp = self.bootstrap_params_[valid_rows]
        if len(bp) < 20:
            raise RuntimeError(f"Too few valid bootstrap samples ({len(bp)})")

        alpha_lower = alpha/2
        alpha_upper = 1 - alpha/2

        # The first column of bp is C
        # Then for i in 0..(n_sources-1), we have (alpha_i, beta_i) in columns 1+2*i : 1+2*i+2
        C_samples = bp[:, 0]
        C_ci = np.quantile(C_samples, [alpha_lower, alpha_upper])

        # We'll build up a table of results
        # The user can easily slice out "A_{1,i}" vs. "beta_i".
        param_names = ["C"]
        param_data = [C_ci]

        n_sources = (bp.shape[1] - 1) // 2  # how many alpha,beta pairs
        # gather alpha_i, beta_i
        col = 1
        for i in range(n_sources):
            alpha_i_samples = bp[:, col]
            beta_i_samples = bp[:, col+1]
            col += 2

            # A_{1,i} = exp(alpha_i)
            A1_i_samples = np.exp(alpha_i_samples)
            A1_i_ci = np.quantile(A1_i_samples, [alpha_lower, alpha_upper])
            beta_i_ci = np.quantile(beta_i_samples, [alpha_lower, alpha_upper])

            param_names.append(f"A_1_source{i}")
            param_data.append(A1_i_ci)

            param_names.append(f"beta_source{i}")
            param_data.append(beta_i_ci)

        ci_df = pd.DataFrame(index=param_names, columns=["lower", "upper"], data=param_data)
        return ci_df

    def params(self):
        """
        Return the point estimates (C, A_{1,i}, beta_i) as a dict or DataFrame.
        """
        d = {"C": self.C_}
        for i, (alpha_i, beta_i) in enumerate(zip(self.alpha_, self.beta_)):
            d[f"A_1_source{i}"] = np.exp(alpha_i)
            d[f"beta_source{i}"] = beta_i
        return pd.Series(d)

    def predict_from_lmp(
        self,
        Z: np.ndarray,
        source_i: int,
        conf_int: bool = False,
        alpha: float = 0.05,
        return_samples: bool = False
    ):
        r"""
        Predict :math:`S_t = A_{1,i} * Z^{\beta_i}` at given Z, for a single data source i.

        If ``conf_int=True``, uses the bootstrap distribution of (alpha_i_b, beta_i_b),
        ignoring the bootstrap's C_b (since here we treat Z as "already known").

        Parameters
        ----------
        Z : np.ndarray
            Input array of Z-values. shape = (n_points,).
        source_i : int
            Which data source's alpha_i, beta_i to use (0-based).
        conf_int : bool, optional
            If True, also return percentile-based (lb, ub).
        alpha : float, optional
            Significance level (default=0.05 => 95% intervals).
        return_samples : bool, optional
            If True, also return the entire array of bootstrap sample predictions.

        Returns
        -------
        If conf_int=False and return_samples=False:
            S_pred : np.ndarray
        If conf_int=True and return_samples=False:
            (S_pred, (lb, ub))
        If conf_int=False and return_samples=True:
            (S_pred, samples_2d)
        If conf_int=True and return_samples=True:
            (S_pred, (lb, ub), samples_2d)
        """
        Z = np.asarray(Z)
        alpha_i = self.alpha_[source_i]
        beta_i  = self.beta_[source_i]

        # 1) Point prediction from final fit
        S_pred = np.exp(alpha_i) * (Z ** beta_i)

        # If no conf_int and no samples, done
        if (not conf_int) and (not return_samples):
            return S_pred

        # 2) Construct bootstrap predictions
        valid_rows = np.all(np.isfinite(self.bootstrap_params_), axis=1)
        bp = self.bootstrap_params_[valid_rows]  # shape (n_valid, 1+2*n_sources)
        alpha_col = 1 + 2*source_i
        beta_col  = 2 + 2*source_i

        n_valid = bp.shape[0]
        samples_2d = np.zeros((n_valid, len(Z)), dtype=float)

        for j in range(n_valid):
            alpha_b = bp[j, alpha_col]
            beta_b  = bp[j, beta_col]
            # we ignore the bootstrap's C_b for computing Z, because Z is already given.
            S_pred_b = np.exp(alpha_b) * (Z ** beta_b)
            samples_2d[j, :] = S_pred_b

        # Build the return
        results = [S_pred]
        if conf_int:
            alpha_lower = alpha / 2
            alpha_upper = 1.0 - alpha / 2
            lb = np.nanquantile(samples_2d, alpha_lower, axis=0)
            ub = np.nanquantile(samples_2d, alpha_upper, axis=0)
            results.append((lb, ub))

        if return_samples:
            results.append(samples_2d)

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def predict_from_lmp_aggregate(
        self,
        Z: np.ndarray,
        conf_int: bool = False,
        alpha: float = 0.05,
        return_samples: bool = False
    ):
        r"""
        Provide an overall weighted mean and confidence interval across ALL data 
        sources at once, given a user-supplied Z. We do not re-compute Z from 
        time/temperature or from the bootstrap's C_b.

        Weighted point estimate:
            S_agg = ( sum_{i} [n_i * S_pred_i] ) / ( sum_{i} n_i )

        Weighted intervals:
          1) For each source i, compute bootstrap predictions ignoring the bootstrap's C_b,
             i.e. S_pred_b_i = exp(alpha_b) * Z^beta_b.
          2) Replicate those predictions n_i times each (so bigger n_i => bigger weight).
          3) Concatenate into one big array.
          4) Compute percentile-based intervals.

        Parameters
        ----------
        Z : np.ndarray
            Array of Z-values for which to predict S.
        conf_int : bool, optional
            If True, also return percentile intervals from the weighted bootstrap.
        alpha : float, optional
            Significance level (0.05 => 95% intervals).
        return_samples : bool, optional
            If True, also return the big weighted array of bootstrap predictions.

        Returns
        -------
        If conf_int=False and return_samples=False:
            S_agg : np.ndarray
        If conf_int=True and return_samples=False:
            (S_agg, (lb, ub))
        If conf_int=False and return_samples=True:
            (S_agg, samples_2d)
        If conf_int=True and return_samples=True:
            (S_agg, (lb, ub), samples_2d)
        """
        if self.data_sizes_ is None:
            raise RuntimeError("No data_sizes_ found. Did you call .fit(...) first?")
        Z = np.asarray(Z)
        n_sources = len(self.alpha_)

        # Weighted point estimate:
        total_n = float(sum(self.data_sizes_))
        S_num = np.zeros_like(Z, dtype=float)
        for i in range(n_sources):
            A_1i = np.exp(self.alpha_[i])
            beta_i = self.beta_[i]
            S_i = A_1i * (Z ** beta_i)  # point pred for source i
            S_num += self.data_sizes_[i] * S_i
        S_agg = S_num / total_n

        # If no intervals or samples, done
        if (not conf_int) and (not return_samples):
            return S_agg

        # Weighted distribution:
        big_samples_list = []
        valid_rows = np.all(np.isfinite(self.bootstrap_params_), axis=1)
        bp = self.bootstrap_params_[valid_rows]  # shape (n_valid, 1+2*n_sources)

        for i in range(n_sources):
            alpha_col = 1 + 2*i
            beta_col  = 2 + 2*i
            n_valid = bp.shape[0]
            samples_2d = np.zeros((n_valid, len(Z)), dtype=float)
            for j in range(n_valid):
                alpha_b = bp[j, alpha_col]
                beta_b  = bp[j, beta_col]
                S_pred_b = np.exp(alpha_b) * (Z ** beta_b)
                samples_2d[j, :] = S_pred_b

            # replicate by n_i
            repeated = np.repeat(samples_2d, repeats=self.data_sizes_[i], axis=0)
            big_samples_list.append(repeated)

        if len(big_samples_list) == 0:
            # edge case: no valid bootstrap => just return S_agg
            if conf_int or return_samples:
                # can't build intervals
                raise RuntimeError("No valid bootstrap rows to build intervals.")
            return S_agg

        big_samples = np.concatenate(big_samples_list, axis=0)

        results = [S_agg]
        if conf_int:
            alpha_lower = alpha / 2
            alpha_upper = 1.0 - alpha / 2
            lb = np.nanquantile(big_samples, alpha_lower, axis=0)
            ub = np.nanquantile(big_samples, alpha_upper, axis=0)
            results.append((lb, ub))

        if return_samples:
            results.append(big_samples)

        return tuple(results) if len(results) > 1 else results[0]
    
    def fit_models(self, t_list, T_list, S_list,
                    method_bounds=(-10, 100),
                    n_boot=1000, random_state=0):
            """
            1) Calls .fit(...) on the multi-source data.
            2) Splits the final point estimates and bootstrap samples into a list
            of LarsonMillerPowerLaw models, one per source, each with the same C
            but its own alpha_i, beta_i.

            Returns
            -------
            models : list of LarsonMillerPowerLaw
                A separate single-source model for each data source.
            """
            # 1) Fit the multi-source model
            self.fit(t_list, T_list, S_list,
                    method_bounds=method_bounds,
                    n_boot=n_boot,
                    random_state=random_state)

            from copy import deepcopy

            n_sources = len(t_list)
            # 2) Construct a single-source model for each source i
            models = []
            for i in range(n_sources):
                # Point estimates:
                alpha_i = self.alpha_[i]
                beta_i  = self.beta_[i]
                # We'll keep the single, shared C_ for all models
                A_1_i = np.exp(alpha_i)

                # Build a (n_boot, 3) bootstrap array for source i:
                #    col0 => alpha_b, col1 => beta_b, col2 => C_b
                # from the multi array shape (n_boot, 1 + 2*n_sources):
                #   col 0 => C_b
                #   col 1+2*i => alpha_i_b
                #   col 2+2*i => beta_i_b
                bp_local = np.full((self.n_boot, 3), np.nan, dtype=float)
                valid_rows = np.all(np.isfinite(self.bootstrap_params_), axis=1)
                multi_bp = self.bootstrap_params_[valid_rows]  # shape (n_valid, 1 + 2*n_sources)

                if len(multi_bp) > 0:
                    alpha_col = 1 + 2*i
                    beta_col  = 2 + 2*i
                    alpha_b_samples = multi_bp[:, alpha_col]
                    beta_b_samples  = multi_bp[:, beta_col]
                    C_b_samples     = multi_bp[:, 0]

                    # reassemble
                    n_valid_b = multi_bp.shape[0]
                    bp_local = np.zeros((n_valid_b, 3), dtype=float)
                    bp_local[:, 0] = alpha_b_samples
                    bp_local[:, 1] = beta_b_samples
                    bp_local[:, 2] = C_b_samples

                # Make a new single-source model
                mod_i = LarsonMillerPowerLaw()
                # We'll skip calling .fit() inside that single model; we just set the results:
                mod_i.set_params_from_values(
                    alpha=alpha_i,
                    beta=beta_i,
                    C=self.C_,
                    bootstrap_params=bp_local
                )
                models.append(mod_i)

            return models
    
    def get_prediction(
        self,
        t: np.ndarray,
        T: np.ndarray,
        source_i: int,
        conf_int: bool = False,
        alpha: float = 0.05,
        return_samples: bool = False
    ):
        """
        Predict S(t) for a single data source i with the single, shared C_ 
        but alpha_i, beta_i from that source.

        If conf_int=True, returns percentile-based intervals from that source's
        bootstrap distribution. If return_samples=True, returns the entire
        bootstrap array of predictions.

        Returns
        -------
        If conf_int=False and return_samples=False:
            S_pred : np.ndarray
        If conf_int=True and return_samples=False:
            (S_pred, (lb, ub))
        If conf_int=False and return_samples=True:
            (S_pred, samples_2d)
        If conf_int=True and return_samples=True:
            (S_pred, (lb, ub), samples_2d)
        """
        t = np.asarray(t)
        T = np.asarray(T)
        if len(t) != len(T):
            raise ValueError("t and T must have the same shape/length.")

        alpha_i = self.alpha_[source_i]
        beta_i = self.beta_[source_i]
        Z = T * (self.C_ + np.log(t))
        if np.any(Z <= 0):
            raise ValueError("Non-positive values in Z encountered.")

        # Point prediction
        S_pred = np.exp(alpha_i) * (Z ** beta_i)

        # If no intervals or samples, just return the point predictions
        if (not conf_int) and (not return_samples):
            return S_pred

        # Otherwise, build the bootstrap distribution for that source
        valid_rows = np.all(np.isfinite(self.bootstrap_params_), axis=1)
        bp = self.bootstrap_params_[valid_rows]
        # The columns: col 0 => C_b, col (1+2*i) => alpha_i_b, col (2+2*i) => beta_i_b
        alpha_col = 1 + 2*source_i
        beta_col  = 2 + 2*source_i

        n_valid = bp.shape[0]
        samples_2d = np.zeros((n_valid, len(t)), dtype=float)

        for j in range(n_valid):
            alpha_b = bp[j, alpha_col]
            beta_b  = bp[j, beta_col]
            C_b     = bp[j, 0]
            Z_b     = T * (C_b + np.log(t))
            if np.any(Z_b <= 0):
                samples_2d[j, :] = np.nan
            else:
                samples_2d[j, :] = np.exp(alpha_b) * (Z_b ** beta_b)

        results = [S_pred]
        if conf_int:
            alpha_lower = alpha / 2
            alpha_upper = 1.0 - alpha / 2
            lb = np.nanquantile(samples_2d, alpha_lower, axis=0)
            ub = np.nanquantile(samples_2d, alpha_upper, axis=0)
            results.append((lb, ub))

        if return_samples:
            results.append(samples_2d)

        return tuple(results) if len(results) > 1 else results[0]

    def get_prediction_aggregate(
        self,
        t: np.ndarray,
        T: np.ndarray,
        conf_int: bool = False,
        alpha: float = 0.05,
        return_samples: bool = False
    ):
        """
        Provide an "overall" weighted mean and confidence interval from ALL data 
        sources at once, weighting by the number of observations n_i in each source.

        Weighted point estimate:
            S_agg(t) = sum_i [n_i * S_i(t)] / sum_i n_i

        Weighted confidence intervals:
            1) For each source i, get the single-source bootstrap predictions 
                and replicate them n_i times (np.repeat).
            2) Concatenate across all sources => big array.
            3) Compute percentile-based intervals from that big, repeated array.

        Parameters
        ----------
        t, T : np.ndarray
            Conditions at which to predict.
        conf_int : bool, optional
            Whether to compute percentile intervals from the repeated 
            (weighted) bootstrap samples.
        alpha : float, optional
            Significance level for intervals (default 0.05 => 95% intervals).
        return_samples : bool, optional
            Whether to also return the big weighted bootstrap predictions array.

        Returns
        -------
        If conf_int=False and return_samples=False:
            S_agg : np.ndarray (shape = (len(t),))
        If conf_int=True and return_samples=False:
            (S_agg, (lb, ub))
        If conf_int=False and return_samples=True:
            (S_agg, samples_2d)
        If conf_int=True and return_samples=True:
            (S_agg, (lb, ub), samples_2d)
        """
        t = np.asarray(t)
        T = np.asarray(T)
        n_sources = len(self.alpha_)
        if self.data_sizes_ is None or len(self.data_sizes_) != n_sources:
            raise RuntimeError("No valid data_sizes_ found. Make sure .fit(...) was called first.")

        # 1) Weighted point estimate
        total_n = sum(self.data_sizes_)
        S_num = np.zeros_like(t, dtype=float)  # accumulate numerator
        for i in range(n_sources):
            Z = self.lmp(t, T, self.C_)
            S_i = self.predict_from_lmp(Z, i, conf_int=False, return_samples=False)
            S_num += self.data_sizes_[i] * S_i
        S_agg = S_num / total_n

        # If no intervals or samples, done
        if (not conf_int) and (not return_samples):
            return S_agg

        # 2) Weighted distribution: replicate each source's samples by n_i
        big_samples_list = []
        number_samples = [int(self.bootstrap_params_.shape[0]*float(size)/float(max(self.data_sizes_)))
                           for size in self.data_sizes_]
        
        for i in range(n_sources):
            # single-source bootstrap
            # conf_int=False => we won't get (lb,ub)
            # return_samples=True => we do get samples_2d
            C_samples = np.random.choice(self.bootstrap_params_[:, 0], number_samples[i])
            repeated = np.empty([number_samples[i],self.bootstrap_params_.shape[0], t.shape[0]], dtype=float)
            for j in range(C_samples.shape[0]):
                Z = self.lmp(t,T, C_samples[j])
                _,repeated[j,...]  =self.predict_from_lmp(Z, i,
                                                    conf_int=False,
                                                    return_samples=True)
                
            # replicate by n_i
            big_samples_list.append(repeated.reshape([-1, t.shape[0]]))

        big_samples = np.concatenate(big_samples_list, axis=0)  # shape = (sum_i(n_i * valid_boot_i), len(t))

        results = [S_agg]
        if conf_int:
            alpha_lower = alpha / 2
            alpha_upper = 1.0 - alpha / 2
            lb = np.nanquantile(big_samples, alpha_lower, axis=0)
            ub = np.nanquantile(big_samples, alpha_upper, axis=0)
            results.append((lb, ub))

        if return_samples:
            results.append(big_samples)

        return tuple(results) if len(results) > 1 else results[0]