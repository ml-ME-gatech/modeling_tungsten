import jax.numpy as jnp
import numpyro
from numpyro.distributions import Normal, TruncatedNormal, LogNormal, Distribution, Exponential, HalfCauchy, InverseGamma
import numpy as np
from typing import Callable, Dict, List, Tuple, Any
from numpyro.infer import Predictive
from statsmodels.regression.linear_model import  OLSResults
from jax import vmap
from numpyro.distributions import MultivariateNormal, Uniform
from functools import wraps, cached_property
import os
import importlib
import dill

def jmak_model(t: np.ndarray,T:np.ndarray,a1: float,B1: float,a2: float,B2: float,n: float) -> np.ndarray:
    """
    So we can just plug in a point estimate easily
    """
    b = np.exp(a1 + B1/T)
    t_inc = np.exp(a2 + B2/T)
    return 1.0 - np.exp(-b**n*(np.where(t > t_inc, t,t_inc) - t_inc)**n)

def glm_model(t: np.ndarray,T:np.ndarray,a1: float,B1: float,a2: float,B2: float,nu: float) -> np.ndarray:
    B = np.exp(a1 + B1/T)
    M = np.exp(a2 + B2/T)
    return (1.+ np.exp(-B*(t - M)))**(-1./nu)

def glm_numpyro_model(t: jnp.ndarray,
                      T: jnp.ndarray,
                      X: jnp.ndarray,
                      std: jnp.ndarray,
                      pmean: jnp.ndarray,
                      pstd: jnp.ndarray) -> None:
    """
    numpyro/jax numpy compatible model for the glm
    """

    # priors
    a1 = numpyro.sample('a1',Normal(pmean[0],pstd[0]))
    a2 = numpyro.sample('a2',Normal(pmean[2],pstd[2]))
    B1 = numpyro.sample('B1',Normal(pmean[1],pstd[1]))
    B2 = numpyro.sample('B2',Normal(pmean[3],pstd[3]))
    nu = numpyro.sample('nu',LogNormal(pmean[4]))

    sigma = numpyro.sample('sigma',HalfCauchy(1.0))

    # arrhenius equations
    B = jnp.exp(a1 + B1/T)
    M = jnp.exp(a2 + B2/T)

    #model prediction
    Xhat = (1.+ jnp.exp(-B*(t - M)))**(-1./nu)
    
    # likelihood
    with numpyro.plate("data", t.shape[0]):
        numpyro.sample('obs',TruncatedNormal(Xhat,scale = jnp.sqrt(sigma**2 + std**2),
                                             low = 0.0,high = 1.0),obs = X)

def jmak_numpyro_model(t: jnp.ndarray,
                      T: jnp.ndarray,
                      X: jnp.ndarray,
                      std: jnp.ndarray,
                      pmean: jnp.ndarray,
                      pstd: jnp.ndarray) -> None:
    """
    numpyro/jax numpy compatible model for the JMAK model
    """

    # priors
    a1 = numpyro.sample('a1',Normal(pmean[0],pstd[0]))
    a2 = numpyro.sample('a2',Normal(pmean[2],pstd[2]))
    B1 = numpyro.sample('B1',Normal(pmean[1],pstd[1]))
    B2 = numpyro.sample('B2',Normal(pmean[3],pstd[3]))
    n = numpyro.sample('n',LogNormal(pmean[4]/2))

    sigma = numpyro.sample('sigma',HalfCauchy(1.0))

    # arrhenius equations
    b = jnp.exp(a1 + B1/T)
    t_inc = jnp.exp(a2 + B2/T)

    #model prediction
    Xhat = 1.0 - jnp.exp(-b**n*(jnp.where(t > t_inc, t,t_inc) - t_inc)**n)
    
    # likelihood
    with numpyro.plate("data", t.shape[0]):
        numpyro.sample('obs',TruncatedNormal(Xhat,scale = jnp.sqrt(sigma**2 + std**2),
                                             low = 0.0,high = 1.0),obs = X)


def gl_likelihood(params: jnp.ndarray,
                    t: jnp.ndarray,
                    T: jnp.ndarray,
                    X: jnp.ndarray,
                    std: jnp.ndarray) -> None:
    """
    jax numpy compatible likelihood for the GL model
    """

    # priors
    a1,B1,a2,B2,nu,log_sigma = params
    
    # arrhenius equations
    B = jnp.exp(a1 + B1/T)
    M = jnp.exp(a2 + B2/T)

    #model prediction
    Xhat = (1.+ jnp.exp(-B*(t - M)))**(-1./nu)
    
    # likelihood
    likelihood = TruncatedNormal(Xhat,scale = jnp.sqrt(jnp.exp(log_sigma)**2 + std**2),low = 0.0,high = 1.0)

    return likelihood.log_prob(X).sum() 

def jmak_likelihood(params: jnp.ndarray,
                    t: jnp.ndarray,
                    T: jnp.ndarray,
                    X: jnp.ndarray,
                    std: jnp.ndarray) -> None:
    """
    jax numpy compatible likelihood for the JMAK model
    """

    # priors
    a1,B1,a2,B2,n,log_sigma = params
    
    # arrhenius equations
    b = jnp.exp(a1 + B1/T)
    t_inc = jnp.exp(a2 + B2/T)

    #model prediction
    Xhat = 1.0 - jnp.exp(-b**n*(jnp.where(t > t_inc, t,t_inc) - t_inc)**n)
    
    # likelihood
    likelihood = TruncatedNormal(Xhat,scale = jnp.sqrt(jnp.exp(log_sigma)**2 + std**2),low = 0.0,high = 1.0)

    return likelihood.log_prob(X).sum() 

def build_predictive_seperate(model_fn: Callable, posterior_samples: Dict, *, return_name="obs", **fixed_kwargs):
    """
    Wraps a NumPyro model so that you only need to pass rng_key, t, T, std.

    Args:
        model_fn:           your numpyro model, e.g. jmak_numpyro_model
        posterior_samples:  dict of arrays from your MCMC/SVI or 
                            {'a1': ..., 'B1': ..., 'n': ..., …}
        return_name:        which site to return (default "obs")
        **fixed_kwargs:     kwargs that the model always needs
                            (e.g. pmean=pmean, pstd=pstd)

    Returns:
        A function `predict(rng_key, *, t, T, std)` which returns
        an array of shape (n_draws, total_obs) of posterior‑predictive draws.
    """
    # instantiate a Predictive object once
    predictive = Predictive(model_fn,
                            posterior_samples=posterior_samples,
                            return_sites=[return_name])

    def predict(rng_key, *, 
                t: jnp.ndarray, 
                T: jnp.ndarray, 
                std: jnp.ndarray = None):
        
        if std is None:
            std = jnp.zeros_like(t)

        # ensure everything is a jax array
        kwargs = {
            "t":      jnp.asarray(t),
            "T":      jnp.asarray(T),
            "std":    jnp.asarray(std),
            # we never pass X as data, so obs=None gives draws
            "X":      None,
            **fixed_kwargs
        }
        out = predictive(rng_key, **kwargs)
        return out[return_name]

    return predict

def normal_prior_lm(name: str,
                    params: Tuple[jnp.ndarray| np.ndarray],
                    latent_transform: Callable[[jnp.ndarray],jnp.ndarray]) -> Tuple[str,Distribution,Callable]:

    """
    make a linear latent model with a normal prior on the parameters, specified by 
    the mean and covariance.

    Args:
    ---------
    name: str
        the name of the latent model
    params: Tuple[jnp.ndarray| np.ndarray]
        the mean and covariance of the normal prior
    latent_transform: Callable[[jnp.ndarray],jnp.ndarray]
        the transformation to apply to the latent variables before applying the linear model

    Returns:
    ---------
    Tuple[str,Distribution,Callable]
    The name, distribution, and latent model function, with a None placeholder for sampled parameters
    """
    def linear_latent_model(x: jnp.ndarray,p: jnp.ndarray):
        x_ = latent_transform(x)
        y = jnp.sum(x_ * p,axis = -1)
        return y

    return [name,MultivariateNormal(params[0],params[1]),linear_latent_model,None]

def uniform_prior_lm(name: str,
                    params: Tuple[jnp.ndarray| np.ndarray],
                    latent_transform: Callable[[jnp.ndarray],jnp.ndarray]) -> Tuple[str,Distribution,Callable]:

    """
    Effectively the same as normal_prior_lm, but with a uniform prior on the parameters
    """
    def linear_latent_model(x: jnp.ndarray,p: jnp.ndarray):
        x_ = latent_transform(x)
        y = jnp.sum(x_ * p,axis = -1)
        return y

    a = params[0] - 3.0*jnp.sqrt(params[1])
    b = params[0] + 3.0*jnp.sqrt(params[1])
    return [name,Uniform(a,b),linear_latent_model,None]

def statsmodels_lm_to_bayesian(name: str,
                               ols: OLSResults,
                               latent_transform: Callable[[jnp.ndarray],jnp.ndarray],
                               prior = 'normal') -> Tuple[str,
                                                        Distribution,
                                                        Callable]:

    """
    utility to convert a statsmodels OLS model to a bayesian model with a normal or uniform prior
    """
    try:
        return {'normal':normal_prior_lm(name,(ols.params,ols.cov_params()),latent_transform),
        'uniform':uniform_prior_lm(name,(ols.params,5*np.diag(ols.cov_params())),latent_transform)}[prior]
    except KeyError:
        raise ValueError('Invalid prior type. Must be one of "normal" or "uniform"')

def alm(latent_model_tuple: Tuple[jnp.ndarray,Callable], 
                       x: jnp.ndarray) -> jnp.ndarray:
    
    """
    apply the latent model
    """
    _,_,latent_model,latent_params = latent_model_tuple
    return latent_model(x,latent_params)


def correct_tform_shape(func):
    @wraps(func)
    def wrapped(x: jnp.ndarray) -> jnp.ndarray:
        squeezed = False
        if x.ndim == 1:
            x = x[jnp.newaxis, ...]   # same as x.reshape((1, -1))
            squeezed = True
        out = func(x)
        if squeezed:
            out = out.squeeze(0)
        return out
    return wrapped

#affine transformations for the latent variables
@correct_tform_shape
def log_kbar_tform(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([jnp.ones([x.shape[0],1]),x[...,0:1]],axis = 1)

@correct_tform_shape
def log_tbar_tform(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([jnp.ones([x.shape[0],1]),x[...,1:2]],axis = 1)

def jmak_single_model_prediction(t: jnp.ndarray,
                            T: jnp.ndarray,
                            std: jnp.ndarray,
                            latent_models: Dict[str,Tuple[jnp.ndarray,Callable]],
                            latent_var : jnp.ndarray,
                            sigma: float) -> Tuple[jnp.ndarray,jnp.ndarray]:
    
    #apply the latent models to the sampled latent variables and latent model parameters
    a1 = alm(latent_models['a1'],latent_var)
    B1 = alm(latent_models['B1'],latent_var)
    n = alm(latent_models['n'],latent_var)
    a2 = alm(latent_models['a2'],latent_var)
    B2 = alm(latent_models['B2'],latent_var)
    #compute the arrhenius parameters from the model parameters
    b = jnp.exp(a1 + B1/T)
    t_inc = jnp.exp(a2 + B2/T)
    
    #compute the model prediction
    Xhat = 1.0 - jnp.exp(-b**n*(jnp.where(t > t_inc, t,t_inc) - t_inc)**n)

    return Xhat,jnp.sqrt(std**2 + sigma**2)

def gl_single_model_prediction(t: jnp.ndarray,
                            T: jnp.ndarray,
                            std: jnp.ndarray,
                            latent_models: Dict[str,Tuple[jnp.ndarray,Callable]],
                            latent_var : jnp.ndarray,
                            sigma: float) -> Tuple[jnp.ndarray,jnp.ndarray]:
    
    #apply the latent models to the sampled latent variables and latent model parameters
    a1 = alm(latent_models['a1'],latent_var)
    B1 = alm(latent_models['B1'],latent_var)
    nu = alm(latent_models['nu'],latent_var)
    a2 = alm(latent_models['a2'],latent_var)
    B2 = alm(latent_models['B2'],latent_var)

    #compute the arrhenius parameters from the model parameters
    B = jnp.exp(a1 + B1/T)
    M = jnp.exp(a2 + B2/T)
    
    #compute the model prediction
    Xhat = (1.+ jnp.exp(-B*(t - M)))**(-1./nu)

    return Xhat,jnp.sqrt(std**2 + sigma**2)

def numpyro_hierarchical_model(single_model_prediction: Callable,
                                    t: List[jnp.ndarray],
                                    T: List[jnp.ndarray],
                                    X: List[jnp.ndarray],
                                    std: List[jnp.ndarray],
                                    latent_models: Dict[str,Tuple[jnp.ndarray,Callable]],
                                    latent_means: jnp.ndarray,
                                    latent_std: jnp.ndarray) -> None:
                      
    """
    numpyro/jax numpy compatible model for the JMAK model
    """
    _shape = sum([t_.shape[0] for t_ in t])
    if X is not None:
        X_ = jnp.concatenate(X)
    else:
        X_ = None

    # priors for "latent" variables
    latent_dist = Normal(latent_means,latent_std)
    latent_var = numpyro.sample('latent_variables',latent_dist)

    #set-up the latent models
    for p,(name,dist,_,_)  in latent_models.items():
        latent_models[p][-1] = numpyro.sample(name,dist)

    #priors for the model error
    sigma = numpyro.sample('sigma',HalfCauchy(1.0).expand([latent_means.shape[0]]))

    Xhat_list =[]
    std_list = []
    #compute the model predictions and stds for each data set
    for i,(t_,T_,std_) in enumerate(zip(t,T,std)):
        Xhat,std_dev = single_model_prediction(t_,T_,std_,latent_models,latent_var[i:i+1,...],sigma[i])
        Xhat_list.append(Xhat)
        std_list.append(std_dev)
    
    Xhat = jnp.concatenate(Xhat_list)
    tt_std = jnp.concatenate(std_list)

    #sample the joint likelihood
    with numpyro.plate('data',_shape):
        numpyro.sample('obs',TruncatedNormal(loc = Xhat,scale = tt_std,low = 0.0,high = 1.),obs = X_)

def _param_map(params: jnp.ndarray) -> Tuple[jnp.ndarray]:

    a1 = params[:2]
    B1 = params[2:4]
    n = params[4:6]
    a2 = params[6:8]
    B2 = params[8:10]
    latent_var = params[10:-5].reshape((5,2))
    sigma = params[-5:]
    return a1,B1,n,a2,B2,latent_var,sigma
                    
def numpyro_hierarchical_posterior(single_model_prediction: Callable,
                                    ep_param: str,
                                    latent_models: Dict[str,Tuple[jnp.ndarray,Callable]],
                                    params: jnp.ndarray,
                                    t: List[jnp.ndarray],
                                    T: List[jnp.ndarray],
                                    X: List[jnp.ndarray],
                                    std: List[jnp.ndarray]) -> float:
    
    a1,B1,n,a2,B2,latent_var,log_error = _param_map(params)
    latent_models['a1'][-1] = a1
    latent_models['B1'][-1] = B1
    latent_models[ep_param][-1] = n
    latent_models['a2'][-1] = a2
    latent_models['B2'][-1] = B2
    X_ = jnp.concatenate(X)
    #compute the model predictions and stds for each data set
    Xhat_list =[]
    std_list = []
    #compute the model predictions and stds for each data set
    for i,(t_,T_,std_) in enumerate(zip(t,T,std)):
        Xhat,std_dev = single_model_prediction(t_,T_,std_,latent_models,latent_var[i:i+1,...],jnp.exp(log_error[i]))
        Xhat_list.append(Xhat)
        std_list.append(std_dev)
    
    Xhat = jnp.concatenate(Xhat_list)
    tt_std = jnp.concatenate(std_list)

    #sample the joint likelihood
    return TruncatedNormal(Xhat,tt_std,low = 0.0,high = 1.0).log_prob(X_).sum()


def predict_single_group(predict_fn: Callable,
                         rng_key: jnp.ndarray,
                         t: jnp.ndarray,
                         T: jnp.ndarray,
                         std: jnp.ndarray = None,
                         *,
                         n_groups: int,
                         group_index: int = 0):
    """
    Use a single t/T/std to do a hierarchical posterior predict, then pick out
    just one group's predictions.

    Args:
      predict_fn:    the function returned by build_predictive_hierarchical(...)
      rng_key:       a JAX PRNGKey
      t, T:          1‑d arrays of your new time / temperature points
      std:           optional measurement‑error array (defaults to zeros)
      n_groups:      how many groups were in your original hierarchical fit
      group_index:   which group's predictions to return (0‑based)

    Returns:
      Array of shape (n_draws, len(t)) with the posterior‑predictive draws
      for that single new dataset.
    """
    # 1) coerce to JAX arrays & default std→0
    t   = jnp.asarray(t)
    T   = jnp.asarray(T)
    std = jnp.zeros_like(t) if std is None else jnp.asarray(std)

    # 1) broadcast to lists
    t_list   = [t] * n_groups
    T_list   = [T] * n_groups
    std_list = [std] * n_groups

    # 2) call the hierarchical predict (gets back shape (n_draws, n_groups*len(t)))
    all_preds = predict_fn(rng_key, t=t_list, T=T_list, std=std_list)

    # 3) slice out just the group_index
    N = t.shape[0]
    start = group_index * N
    stop  = (group_index + 1) * N
    return all_preds[:, start:stop]

def build_predictive_hierarchical(posterior_samples: Dict[str, jnp.ndarray],
                                  single_model_prediction: Callable,
                                  latent_models: Dict[str, Tuple[jnp.ndarray,Callable]],
                                  latent_means: jnp.ndarray,
                                  latent_std: jnp.ndarray,
                                  return_name: str = "obs"):
    """
    Wraps your numpyro_hierarchical_model so that you only ever pass
    rng_key, t, T, std — everything else (priors, latent_models, etc.)
    is baked in.

    Args:
      posterior_samples:  Dict of sample arrays from your HMC/SVI run,
                          keys must match the sites in numpyro_hierarchical_model
      single_model_prediction:  your jmak/gl prediction fn
      latent_models:       the dict you built of (name, dist, fn, None)
      latent_means:        array of prior means for the per‐batch latent vars
      latent_std:          array of prior stds for the per‐batch latent vars
      return_name:         which site to return from Predictive (defaults to "obs")

    Returns:
      A function `predict(rng_key, *, t, T, std)` -> ndarray of shape
      (n_draws, total_obs) containing posterior‐predictive samples.
    """

    # build the Predictive once
    predictive = Predictive(
        lambda t, T, std: numpyro_hierarchical_model(
            single_model_prediction=single_model_prediction,
            t=t,
            T=T,
            X=None,                    # no observed data => generative
            std=std,
            latent_models=latent_models,
            latent_means=latent_means,
            latent_std=latent_std,
        ),
        posterior_samples=posterior_samples,
        return_sites=[return_name]
    )

    def predict(rng_key, *,
                t: List[jnp.ndarray],
                T: List[jnp.ndarray],
                std: List[jnp.ndarray] = None):
        # default std → zero if you don’t have one
        if std is None:
            std = [jnp.zeros_like(tt) for tt in t]
        
        # make sure everything is a jnp array
        kwargs = {
            "t":   [jnp.asarray(tt) for tt in t],
            "T":   [jnp.asarray(Ti) for Ti in T],
            "std": [jnp.asarray(si) for si in std],
        }

        out = predictive(rng_key, **kwargs)
        return out[return_name]

    return predict

def build_predictive_hierarchical_single(data_index: int,
                                         posterior_samples: Dict[str, jnp.ndarray],
                                        single_model_prediction: Callable,
                                        latent_models: Dict[str, Tuple[jnp.ndarray,Callable]],
                                        latent_means: jnp.ndarray,
                                        latent_std: jnp.ndarray,
                                        return_name: str = "obs"):
    
    predictive = build_predictive_hierarchical(
        posterior_samples=posterior_samples,
        single_model_prediction=single_model_prediction,
        latent_models=latent_models,
        latent_means=latent_means,
        latent_std=latent_std,
        return_name=return_name)

    def predict(rng_key, *,
                t: jnp.ndarray,
                T: jnp.ndarray,
                std: jnp.ndarray = None):


        return predict_single_group(predictive,
                                    rng_key,
                                    t,
                                    T,
                                    std = std,
                                    n_groups=latent_means.shape[0],
                                    group_index= data_index) 
    
    return predict


def predict_heirarchical_point(
    model_func: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Tuple], jnp.ndarray, float],
        Tuple[jnp.ndarray, jnp.ndarray],
    ],
    map_params: Dict[str, np.ndarray],
    latent_models: Dict[str, Tuple],
    t: np.ndarray,
    T: np.ndarray,
    std: np.ndarray = None,
    group_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience: for a single dataset (group_index), plug in your MAP params
    and return (Xhat, sigma_total).

    Args:
      model_func:     jmak_single_model_prediction or gl_single_model_prediction
      map_params:     inf_summary['ml'], with keys:
                       'a1','B1','n' or 'nu','a2','B2','latent_variables','sigma'
      latent_models:  dict from statsmodels_lm_to_bayesian(...)
                      (each value is [name, dist, fn, None])
      t:              time‑vector, shape (M,)
      T:              temperature‑vector, shape (M,)
      std:            measurement‑error vector, shape (M,). Defaults to zeros.
      group_index:    which row of latent_variables to use (0‑based).

    Returns:
      Xhat:   ndarray (M,) of model prediction
      σ_tot:  ndarray (M,) of total predictive std (model σ + measurement std)
    """
    M = t.shape[0]
    if std is None:
        std = np.zeros_like(t)

    # 1) overwrite latent_models with your point‐estimates
    for key in ['a1','B1','a2','B2']:
        # last slot in each tuple is the actual parameter array
        latent_models[key][-1] = jnp.asarray(map_params[key])

    # your exponent‐param might be 'n' or 'nu'
    ep_key = 'n' if 'n' in map_params else 'nu'
    latent_models[ep_key][-1] = jnp.asarray(map_params[ep_key])

    # 2) grab the latent_variables for this group
    #    shape (n_groups, dim), so pick [group_index]
    lv = jnp.asarray(map_params['latent_variables'][group_index : group_index+1, :])

    # 3) call the model
    Xhat, sigma_total = model_func(
        jnp.asarray(t),
        jnp.asarray(T),
        jnp.asarray(std),
        latent_models,
        lv,
        jnp.asarray(map_params['sigma'][group_index]),
    )

    # back to NumPy for plotting etc.
    return np.array(Xhat), np.array(sigma_total)

class RxKineticNumpyro:
    """
    Convenience wrapper around a NumPyro model + posterior_samples + prior args.

    __init__(
        posterior_samples: Dict or path-to-dill,
        pargs: (pmean, pstd) sequence or path-to-file,
        numpyro_model: your model function (e.g. jmak_numpyro_model)
    )
    """

    def __init__(
        self,
        posterior_samples: Dict | str,
        pargs: Tuple[jnp.ndarray] | str,
        numpyro_model: Callable,
        model_func: Callable,
        *,
        return_name: str = "obs",
        ml_est: Dict[str, float] = None,
    ):
        # store for dump/load
        self.numpyro_model = numpyro_model
        self.model_func = model_func
        self.return_name = return_name
        self.ml_est = ml_est

        # --- load posterior_samples ---
        if isinstance(posterior_samples, str):
            if not os.path.exists(posterior_samples):
                raise FileNotFoundError(f"Cannot find posterior_samples file: {posterior_samples}")
            with open(posterior_samples, "rb") as f:
                self.posterior_samples = dill.load(f)
        elif isinstance(posterior_samples, Dict):
            self.posterior_samples = posterior_samples
        else:
            raise ValueError("posterior_samples must be a dict or path to a dill file")

        # --- load or unpack pmean, pstd ---
        if isinstance(pargs, str):
            if not os.path.exists(pargs):
                raise FileNotFoundError(f"Cannot find pargs file: {pargs}")
            # try dill first
            try:
                with open(pargs, "rb") as f:
                    loaded = dill.load(f)
            except Exception:
                # fallback to numpy .npz
                loaded = np.load(pargs)
            if isinstance(loaded, Dict) and "pmean" in loaded and "pstd" in loaded:
                self.pmean = jnp.asarray(loaded["pmean"])
                self.pstd  = jnp.asarray(loaded["pstd"])
            else:
                raise ValueError("Loaded pargs must be a dict with keys 'pmean' and 'pstd'")
        elif (
            isinstance(pargs, Tuple)
            and len(pargs) == 2
            and hasattr(pargs[0], "shape")
            and hasattr(pargs[1], "shape")
        ):
            self.pmean, self.pstd = jnp.asarray(pargs[0]), jnp.asarray(pargs[1])
        else:
            raise ValueError("pargs must be (pmean, pstd) or path to file containing them")

        # --- build the Predictive object under the hood ---
        self.predictive = build_predictive_seperate(
            self.numpyro_model,
            self.posterior_samples,
            return_name=return_name,
            pmean=self.pmean,
            pstd=self.pstd
        )
        self.model_func = model_func
        self.ml_est = ml_est    

    def sample_predictive(
        self,
        rng_key: jnp.ndarray,
        t: jnp.ndarray,
        T: jnp.ndarray,
        std: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Draw posterior‑predictive samples for a new curve (t, T, std).

        Returns an array of shape (n_draws, len(t)).
        """
        if std is None:
            std = jnp.zeros_like(t)
        
        return self.predictive(rng_key, t=t, T=T, std=std)
    
    def point_prediction(
        self,
        t: jnp.ndarray,
        T: jnp.ndarray,
        point_estimate: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Draw posterior‑predictive samples for a new curve (t, T, std).

        Returns an array of shape (len(t),).
        """
        return self.model_func(t, T, *point_estimate)
    
    def ml_prediction(
        self,
        t: jnp.ndarray,
        T: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Draw posterior‑predictive samples for a new curve (t, T, std).

        Returns an array of shape (n_draws, len(t)).
        """

        return self.point_prediction(t, T, [self.ml_est[key] for key in self._ml_order])
    
    def predictive_confidence_interval(self,
                                       rng_key: jnp.ndarray,
                                       t: jnp.ndarray,
                                       T: jnp.ndarray,
                                       std: jnp.ndarray = None,
                                       alpha = 0.95) -> Tuple[jnp.ndarray,jnp.ndarray]:
        """
        Draw posterior‑predictive samples for a new curve (t, T, std).

        Returns an array of shape (n_draws, len(t)).
        """
        if std is None:
            std = jnp.zeros_like(t)
        
        predictions = self.predictive(rng_key, t=t, T=T, std=std)
        return numpyro.diagnostics.hpdi(predictions, alpha)


    def dump(self, filepath: str):
        """
        Serialize all essential state to a single file.
        """
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        state = {
            'posterior_samples': self.posterior_samples,
            'pmean': np.array(self.pmean),
            'pstd': np.array(self.pstd),
            'numpyro_model': f"{self.numpyro_model.__module__}.{self.numpyro_model.__name__}",
            'model_func': f"{self.model_func.__module__}.{self.model_func.__name__}",
            'return_name': self.return_name,
            'ml_est': self.ml_est,
        }
        with open(filepath, 'wb') as f:
            dill.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'RxKineticNumpyro':
        """
        Reconstruct an instance from a file produced by dump().
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find state file: {filepath}")
        with open(filepath, 'rb') as f:
            state = dill.load(f)
        # helper to import by string
        def _import(path: str):
            mod, name = path.rsplit('.', 1)
            return getattr(importlib.import_module(mod), name)
        numpyro_model = _import(state['numpyro_model'])
        model_func     = _import(state['model_func'])
        pargs = (state['pmean'], state['pstd'])
        return cls(
            state['posterior_samples'],
            pargs,
            numpyro_model,
            model_func,
            return_name=state.get('return_name', 'obs'),
            ml_est=state.get('ml_est', None)
        )
    

class JMAKNumpyro(RxKineticNumpyro):
    """
    Convenience wrapper around a NumPyro model + posterior_samples + prior args.

    __init__(
        posterior_samples: Dict or path-to-dill,
        pargs: (pmean, pstd) sequence or path-to-file,
        numpyro_model: your model function (e.g. jmak_numpyro_model)
    )
    """
    def __init__(
        self,
        posterior_samples: Dict | str,
        pargs: Tuple[jnp.ndarray] | str,
        *,
        return_name: str = "obs",
        ml_est: Dict[str, float] = None,
    ):
        
        super().__init__(
            posterior_samples,
            pargs,
            jmak_numpyro_model,
            jmak_model,
            return_name = return_name,
            ml_est=ml_est
        )

        self._ml_order = ['a1', 'B1', 'a2', 'B2', 'n']

    @classmethod
    def load(cls, filepath: str) -> 'JMAKNumpyro':
        """
        Reconstruct an instance from a file produced by dump().
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find state file: {filepath}")
        with open(filepath, 'rb') as f:
            state = dill.load(f)
        # helper to import by string
        def _import(path: str):
            mod, name = path.rsplit('.', 1)
            return getattr(importlib.import_module(mod), name)
        
        pargs = (state['pmean'], state['pstd'])
        return cls(
            state['posterior_samples'],
            pargs,
            return_name=state.get('return_name', 'obs'),
            ml_est=state.get('ml_est', None)
        )

class GLNumpyro(RxKineticNumpyro):

    def __init__(
        self,
        posterior_samples: Dict | str,
        pargs: Tuple[jnp.ndarray] | str,
        *,
        return_name: str = "obs",
        ml_est: Dict[str, float] = None,
    ):
        
        super().__init__(
            posterior_samples,
            pargs,
            glm_numpyro_model,
            glm_model,
            return_name=return_name,
            ml_est=ml_est
        )
        self._ml_order = ['a1', 'B1', 'a2', 'B2', 'nu']

    @classmethod
    def load(cls, filepath: str) -> 'GLNumpyro':
        """
        Reconstruct an instance from a file produced by dump().
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find state file: {filepath}")
        with open(filepath, 'rb') as f:
            state = dill.load(f)
        # helper to import by string
        def _import(path: str):
            mod, name = path.rsplit('.', 1)
            return getattr(importlib.import_module(mod), name)
        
        pargs = (state['pmean'], state['pstd'])
        return cls(
            state['posterior_samples'],
            pargs,
            return_name=state.get('return_name', 'obs'),
            ml_est=state.get('ml_est', None)
        )
    
class RxKineticHierarchicalNumpyro:
    """
    Convenience wrapper for hierarchical NumPyro models.

    __init__(
        posterior_samples: Dict or path-to-dill,
        latent_means: jnp.ndarray or path-to-file,
        latent_std:  jnp.ndarray or path-to-file,
        numpyro_model: Callable (e.g. numpyro_hierarchical_model),
        single_model_prediction: Callable,
        latent_models: Dict of (name, dist, fn, None)
    )
    """

    def __init__(
        self,
        posterior_samples: Dict | str,
        latent_means: jnp.ndarray |  str,
        latent_std:  jnp.ndarray | str,
        single_model_prediction: Callable,
        ep_param:str,
        linear_models: Dict[str, OLSResults],
        return_name: str = "obs",
        map_est: Dict[str, float] = None,
        prior: str = 'normal'
    ):
        # store for dump/load
        self.single_model_prediction = single_model_prediction
        self.linear_models = linear_models
        self.return_name = return_name
        self.map_est = map_est  # Optional point estimates for latent parameters
        self.ep_param = ep_param
        self.prior = prior

        # --- load posterior_samples ---
        if isinstance(posterior_samples, str):
            if not os.path.exists(posterior_samples):
                raise FileNotFoundError(f"Cannot find posterior_samples file: {posterior_samples}")
            with open(posterior_samples, "rb") as f:
                self.posterior_samples = dill.load(f)
        elif isinstance(posterior_samples, Dict):
            self.posterior_samples = posterior_samples
        else:
            raise ValueError("posterior_samples must be a dict or path to a dill file")

        # --- load latent_means/std ---
        def _load_arr(x):
            if isinstance(x, str):
                if not os.path.exists(x):
                    raise FileNotFoundError(f"Cannot find file: {x}")
                arr = np.load(x) if x.endswith('.npz') else dill.load(open(x,'rb'))
                return jnp.asarray(arr)
            return jnp.asarray(x)

        self.latent_means = _load_arr(latent_means)
        self.latent_std   = _load_arr(latent_std)

        # --- build full predictive ---
        self._predictive = build_predictive_hierarchical(
            posterior_samples=self.posterior_samples,
            single_model_prediction=self.single_model_prediction,
            latent_models=self.latent_models,
            latent_means=self.latent_means,
            latent_std=self.latent_std,
            return_name=self.return_name
        )

    def sample_predictive(
        self,
        rng_key: jnp.ndarray,
        t: List[jnp.ndarray],
        T: List[jnp.ndarray],
        std: List[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Full hierarchical posterior predictive: returns (n_draws, sum n_i)"""
        return self._predictive(rng_key, t=t, T=T, std=std)

    def sample_predictive_group(
        self,
        rng_key: jnp.ndarray,
        t: jnp.ndarray,
        T: jnp.ndarray,
        std: jnp.ndarray = None,
        group_index: int = 0
    ) -> jnp.ndarray:
        """Posterior predictive for a single group: returns (n_draws, len(t))"""
        n_groups = self.latent_means.shape[0]
        return predict_single_group(
            self._predictive,
            rng_key,
            t,
            T,
            std=std,
            n_groups=n_groups,
            group_index=group_index
        )

    def predictive_confidence_interval_group(
        self,
        rng_key: jnp.ndarray,
        t: jnp.ndarray,
        T: jnp.ndarray,
        std: jnp.ndarray = None,
        group_index: int = 0,
        alpha: float = 0.95
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """HPDI for a single group"""
        preds = self.sample_predictive_group(rng_key, t, T, std, group_index)
        return numpyro.diagnostics.hpdi(preds, alpha)

    def map_prediction_group(
        self,
        t: jnp.ndarray,
        T: jnp.ndarray,
        group_index: int = 0
    ) -> np.ndarray:
        """Point‐estimate prediction using ML latent estimates for given group"""
        Xhat = predict_heirarchical_point(self.single_model_prediction,
                                          self.map_est,
            self.latent_models,
            t,
            T,
            std=None,
            group_index=group_index
        )
        return Xhat[0]

    @cached_property
    def latent_models(self) -> Dict[str, Tuple[jnp.ndarray, Callable]]:
        lm = {
            p: statsmodels_lm_to_bayesian(p,
                   self.linear_models[p],
                   log_kbar_tform,
                   prior=self.prior)
            for p in ['a1','B1', self.ep_param]
        }
        lm.update({
            p: statsmodels_lm_to_bayesian(p,
                   self.linear_models[p],
                   log_tbar_tform,
                   prior=self.prior)
            for p in ['a2','B2']
        })
        return lm
    
    def dump(self, filepath: str):
        """Serialize state to file, including latent_models"""
        state = {
            'posterior_samples': self.posterior_samples,
            'latent_means':      np.array(self.latent_means),
            'latent_std':        np.array(self.latent_std),
            'single_model_fn':   f"{self.single_model_prediction.__module__}.{self.single_model_prediction.__name__}",
            'linear_models':     self.linear_models,
            'return_name':       self.return_name,
            'map_est':           self.map_est,
            'ep_param':          self.ep_param,
            'prior':            self.prior
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            dill.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> 'RxKineticHierarchicalNumpyro':
        """Reconstruct from dump(), including latent_models"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find state file: {filepath}")
        with open(filepath, 'rb') as f:
            state = dill.load(f)
        # extract latent_models directly from state
        lm_dict = state['linear_models']
        # re-import your single_model_prediction function
        def _import(path: str):
            mod, name = path.rsplit('.', 1)
            return getattr(importlib.import_module(mod), name)
        single_fn = _import(state['single_model_fn'])
        # rebuild the wrapper
        return cls(
            state['posterior_samples'],
            state['latent_means'],
            state['latent_std'],
            single_fn,
            state.get('ep_param', 'n'),
            lm_dict,
            return_name = state.get('return_name','obs'),
            map_est = state.get('map_est', None),
            prior = state.get('prior', 'normal')
        )

class JMAKHierarchical(RxKineticHierarchicalNumpyro):
    def __init__(
        self,
        posterior_samples: Dict | str,
        latent_means: jnp.ndarray | str,
        latent_std:  jnp.ndarray | str,
        linear_models: Dict[str, OLSResults],
        return_name: str = "obs",
        map_est: Dict[str,float] = None,
        prior: str = 'normal'
    ):
        

        super().__init__(
            posterior_samples,
            latent_means,
            latent_std,
            jmak_single_model_prediction,
            'n',
            linear_models,
            return_name=return_name,
            map_est =map_est,
            prior = prior,
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'RxKineticHierarchicalNumpyro':
        """Reconstruct from dump(), including latent_models"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find state file: {filepath}")
        with open(filepath, 'rb') as f:
            state = dill.load(f)
        # extract latent_models directly from state
        lm_dict = state['linear_models']
        # rebuild the wrapper
        return cls(
            state['posterior_samples'],
            state['latent_means'],
            state['latent_std'],
            lm_dict,
            return_name = state.get('return_name','obs'),
            map_est = state.get('map_est', None),
            prior = state.get('prior', 'normal')
        )
    
class GLHierarchical(RxKineticHierarchicalNumpyro):
    def __init__(
        self,
        posterior_samples: Dict | str,
        latent_means: jnp.ndarray | str,
        latent_std:  jnp.ndarray | str,
        linear_models: Dict[str, OLSResults],
        prior: str = 'normal',
        return_name: str = "obs",
        map_est: Dict[str,float] = None
    ):
        super().__init__(
            posterior_samples,
            latent_means,
            latent_std,
            gl_single_model_prediction,
            'nu',
            linear_models,
            return_name=return_name,
            map_est=map_est,
            prior=prior
        )

    @classmethod
    def load(cls, filepath: str) -> 'GLHierarchical':
        """Reconstruct from dump(), including latent_models"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find state file: {filepath}")
        with open(filepath, 'rb') as f:
            state = dill.load(f)
        # extract latent_models directly from state
        lm_dict = state['linear_models']
        # rebuild the wrapper
        return cls(
            state['posterior_samples'],
            state['latent_means'],
            state['latent_std'],
            lm_dict,
            return_name = state.get('return_name','obs'),
            map_est = state.get('map_est', None),
            prior = state.get('prior', 'normal')
        )


def bayes_r2(model: RxKineticHierarchicalNumpyro | RxKineticNumpyro,
                    rng_key: jnp.ndarray,
                    t: jnp.ndarray,
                    T: jnp.ndarray,
                    y: jnp.ndarray,
                    std: jnp.ndarray = None,
                    **kwargs,
                    ) -> jnp.ndarray:
    """
    Compute posterior draws of Bayesian R^2 for one group.

    Returns:
        r2: array shape (n_draws,) of R^2_s.
    """

    # 1) draw posterior predictive samples (n_draws, N)
    try:
        y_pred = model.sample_predictive_group(
            rng_key, t=t, T=T, std=std, **kwargs
        )  # shape (S, N)
    except AttributeError:
        y_pred = model.sample_predictive(
            rng_key, t=t, T=T, std=std, **kwargs
        )

    # 2) compute per‐draw variances
    var_pred  = jnp.var(y_pred,     axis=1)   # shape (S,)
    var_resid = jnp.var(y - y_pred, axis=1)   # shape (S,)

    # 3) Bayes R^2 per draw
    return var_pred / (var_pred + var_resid)