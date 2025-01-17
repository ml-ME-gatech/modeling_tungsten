import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.stats import t as tdist
from abc import abstractmethod,ABC
from statsmodels.regression.linear_model import OLS, OLSResults
from typing import Tuple, Callable, Iterable
from scipy.optimize import minimize_scalar, OptimizeResult
from dataclasses import dataclass
import pickle
import pandas as pd
import warnings
from scipy.integrate import quad
from scipy.special import gamma,digamma
import math
import optax
import jax
from tqdm import tqdm   
from functools import partial
from jax import lax

_EULER_MASCHERONI = 0.57721566490153286060651209008240243104215933593992
_FILE_TO_LABEL = {'rf_data/alfonso_data/highly_rolled.csv': 'Lopez et al. (2015) - HR',
                'rf_data/alfonso_data/moderate_roll.csv': 'Lopez et al. (2015) - MR',
                'rf_data/richou_data/batch_a_data.csv': 'Richou et al. (2020) - Batch A',
                'rf_data/richou_data/batch_b_data.csv': 'Richou et al. (2020) - Batch B',
                'rf_data/yu_data/data.csv': 'Yu et al. (2017)'}


def resampled_adam(param_samples: Iterable,
                   objective_fun: Callable,
                   lr = 5e-4,
                   opt_iter = 1000):
    
    """
    from an array of initial parameter samples (ideally drawn from the posterior distribution)
    perform adam optimization on the objective function. The function will return the best optimial 
    parameters found
    """
    opt_params = []
    fun_value = np.zeros(len(param_samples))
    reduce = optax.contrib.reduce_on_plateau(factor = 0.5,
                                                patience = 5,
                                                cooldown= 0,
                                                accumulation_size= 50,
                                                min_scale = 1e-8,
                                                rtol = 1e-4)
    
    last_samples = max(1,int(9*opt_iter/10))

    def _update(optimizer, state,_):
        """
        update function for the optimizer
        """
        params,opt_state = state
        value,grads = jax.value_and_grad(objective_fun)(params)
        updates,new_opt_state = optimizer.update(grads,opt_state,params,value = value)    
        new_params = optax.apply_updates(params,updates)
        return (new_params,new_opt_state),(params,value)
    
    def _optimize(optimizer,params):
        """
        do the optimization using the lax.scan function to avoid explicit for loops
        in python. Significantly speeds up the optimization process
        """
        opt_state = optimizer.init(params)
        _, (params_hist,value_hist) = lax.scan(partial(_update, optimizer), (params, opt_state), length=opt_iter)
        return params_hist,value_hist
    
    # main iterator over the parameter samples
    iterator = tqdm(param_samples,desc = 'Optimizing')
    
    for i,psample in enumerate(iterator):   
        solver = optax.chain(optax.adam(lr),reduce,)
        param_hist,value_hist = _optimize(solver,psample)
        i_ = np.argmin(value_hist[last_samples:]) + last_samples
        fun_value[i] = value_hist[i_]
        opt_params.append(param_hist[i_])

    index= np.argmin(fun_value)
    return opt_params[index],fun_value[index]

def kbar_jmak(a1: float,B1: float,n: float,T1: float,T2: float):

    def _integrate_func(x: np.ndarray):
        return np.exp(-B1/x)
    
    return quad(_integrate_func, T1, T2)[0] * gamma(1 + 1/n)/(math.exp(a1)*(T2 - T1))

def kbar_gl(a1: float,B1: float,nu: float,T1: float,T2: float):

    def _integrate_func(x: np.ndarray):
        return np.exp(-B1/x)
    
    return quad(_integrate_func, T1, T2)[0]*(digamma(1/nu) + _EULER_MASCHERONI)/(math.exp(a1)*(T2 - T1))

def tbar(a2: float,B2: float,T1: float,T2: float):

    def _integrate_func(x: np.ndarray):
        return np.exp(B2/x)
    
    return quad(_integrate_func, T1, T2)[0]*math.exp(a2)/(T2 - T1)


def get_loglinear_arrhenius_parameter_bounds_from_file(plabel: str,
                                             file: str,
                                             alpha = 1e-3,
                                             file_to_label = _FILE_TO_LABEL) -> Tuple[np.ndarray,np.ndarray]:
    
    """
    read log-linear arrhenius model from file (approx. estimated in seperate notebook),
    and provide nonlinear optimization bounds for the parameters
    """
    
    label = file_to_label[file]
    with open(f'.model/{plabel}_{label}_first_{2}.pkl','rb') as f:
        ols_res_f = pickle.load(f).parameter_confidence_interval(alpha)
    
    with open(f'.model/{plabel}_{label}_last_{2}.pkl','rb') as f:
        ols_res_l = pickle.load(f).parameter_confidence_interval(alpha)
    
    ci = np.concatenate([ols_res_f,
                         ols_res_l],axis = 1)
    
    bounds = np.array([np.min(ci,axis = 1),np.max(ci,axis = 1)]).T
    return bounds,bounds.mean(axis = 1)
    
def read_prepare_data(file: str,
                      mult = 1.,
                      exclude_index = []) -> pd.DataFrame:
    
    """
    helper function to read data from file and make sure that the values
    are within the bounds of the model. Also make sure that standard 
    deviations are above some minimum value that I couldn't estimate
    from the plots.
    """
    df = pd.read_csv(file,index_col = 0)
    index = np.ones(df.shape[0],dtype = bool)   
    index[exclude_index] = False
    df = df.loc[index,:]
    
    df['time']*=mult
    t = df['time'].to_numpy()
    T = df['temperature'].to_numpy() + 273.15
    X = df['X'].to_numpy()
    X[X <= 0 ] = 0.0
    X[X >= 1] = 1
    with warnings.catch_warnings(action = 'ignore'):
        try:
            df.loc[df['std'] == 0,'std'] = max(df.loc[df['std'] > 0,'std'].min(),1e-3)
        except KeyError as ke:
            df['std'] = 1e-3
            print(f'No standard deviation column found, using {1e-3} for all values')

    return t,T,X,df

def jmak_fit_model_setup(file: str,
                         bounds_n: np.ndarray = np.array([[1.0,3.0]]),
                         mult = 1.,
                         exclude_index = [],
                         **kwargs):
    
    """
    neccssary setup for fitting the JMAK model
    """
    bounds_tinc,p0_tinc = get_loglinear_arrhenius_parameter_bounds_from_file('log_tinc',file,**kwargs)
    bounds_b,p0_b = get_loglinear_arrhenius_parameter_bounds_from_file('log_b',file, **kwargs)
    bounds = np.concatenate([bounds_n,bounds_b,bounds_tinc],axis = 0)
    p0 = np.concatenate([p0_b,p0_tinc])
    return *read_prepare_data(file, mult = mult,exclude_index = exclude_index),bounds,p0

def gl_fit_model_setup(file: str,
                       bounds_nu: np.ndarray = np.array([[1e-3,1.0]]),
                       mult = 1.,
                        exclude_index = [],
                       **kwargs):

    """
    neccssary setup for fitting the GL model
    """
    bounds_B,p0_B = get_loglinear_arrhenius_parameter_bounds_from_file('log_B',file,**kwargs)
    bounds_M,p0_M = get_loglinear_arrhenius_parameter_bounds_from_file('log_tinc',file, **kwargs)
    bounds = np.concatenate([bounds_nu,bounds_B,bounds_M],axis = 0)
    p0 = np.concatenate([p0_B,p0_M])
    return *read_prepare_data(file,mult = mult,exclude_index = exclude_index),bounds,p0


def setup_axis_default(ax: plt.Axes):
    
    """
    convinience function to set up the axis
    """
    ax.tick_params('both',labelsize = 11,which = 'both',direction = 'in')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    return ax

def jmak_function(t: np.ndarray,b: float,t_inc: float,n: float):
    """
    JMAK model, at a specified temperature T which the
    parameters B and M have already been evaluated at.
    """
    1.0 - np.exp(-b**n*(t - t_inc)**n)
    yhat = np.empty_like(t)
    index = t >= t_inc
    yhat[~index] = 0.
    yhat[index] = 1.0 - np.exp(-b[index]**n*(t[index]-t_inc[index])**n)
    return yhat

def generalized_logistic(t: np.ndarray,B: float,M: float,nu: float):
    """
    Generalized logistic model, at a specified temperature T which the
    parameters B and M have already been evaluated at.
    """
    return 1/(1 + np.exp(-B*(t - M)))**(1/nu)

class ArrheniusProcess(ABC):

    _p = None
    def __init__(self,params: np.ndarray = None):
        self.params = params

    @abstractmethod
    def tform(self,x: np.ndarray):
        pass

    @property
    def p(self):
        return self._p

    @abstractmethod
    def fit(self,X: np.ndarray,y: np.ndarray):
        pass

    @abstractmethod
    def log_predict(self,x: np.ndarray):
        pass
    
    @abstractmethod
    def parameter_confidence_interval(self):
        pass

    def __len__(self):
        return self._p
    
    def predict(self,x: np.ndarray):
        return np.exp(self.log_predict(x))
    
    def __call__(self, x: np.ndarray):
        return self.predict(x)
    
class LogLinearArrhenius(ArrheniusProcess):
    
    _p = 2

    def tform(self, x: np.ndarray):
        return np.hstack([np.ones_like(x[:,np.newaxis]),
                          1/x[:,np.newaxis]])
    
    def log_predict(self,x: np.ndarray):
        return self.tform(x).dot(self.params)
    
    def fit(self,x: np.ndarray,y: np.ndarray):
        self.ols_results = OLS(np.log(y),self.tform(x)).fit()
        self.params = self.ols_results.params.squeeze()
        return self
    
    def parameter_confidence_interval(self,alpha: float):
        return self.ols_results.conf_int(alpha)
    
class FudgeFactorArrhenius(LogLinearArrhenius):

    _p = 3

    def parameter_confidence_interval(self,alpha: float):
        pass

    def tform(self, x: np.ndarray):
        return np.hstack([np.ones_like(x[:,np.newaxis]),
                          1/x[:,np.newaxis]**self.params[-1]])
    
    def log_predict(self,x: np.ndarray):
        return self.tform(x).dot(self.params[:-1])
    
    def parameter_confidence_interval(self,alpha: float):
        params_ = self.params.copy()

        def _func(beta: float):
            self.params[-1] = beta
            log_yhat = self.log_predict(self.x)
            return np.linalg.norm(self.logy - log_yhat)
        
        ci = self.ols_results.conf_int(alpha)

        beta_ci = []
        for i in range(2):
            self.params[:-1] = ci[i,:].squeeze()
            opt_result = minimize_scalar(_func,bounds = self.beta_bounds,bracket = self.beta_bounds)
            if opt_result.success:
                beta_ci.append(opt_result.x)
            else:
                raise RuntimeError('failed to find confidence interval for beta')
        
        ci = np.concatenate([ci,np.array(beta_ci)[np.newaxis,:]],axis = 0)
        self.params[:] = params_[:]
        return ci

    def fit(self,x: np.ndarray,
                 y: np.ndarray,
                 beta_init: float,
                 beta_bounds: Tuple[float] = (0.1,2.0)):
        
        self.x = x.copy()

        self.params = np.zeros(3)
        self.params[-1] = beta_init
        self.logy = np.log(y)
        self.beta_bounds = beta_bounds
        
        def _func(beta: float):
            self.params[2] = beta
            X_ = self.tform(x)
            ols_results = OLS(self.logy,X_).fit()
            log_yhat = ols_results.predict(X_)
            return np.linalg.norm(self.logy - log_yhat)
        
        opt_result = minimize_scalar(_func,bracket = beta_bounds,bounds = beta_bounds)
        if opt_result.success:
            self.params[2] = opt_result.x
        else:
            raise RuntimeError('failed to fit fudge factor arrhenius model')
        
        self.ols_results = OLS(self.logy,self.tform(x)).fit()
        self.params[:-1]= self.ols_results.params.squeeze()

        return self   

@dataclass
class LogLinearArrheniusModelFunc:
    
    rxFunc: Callable = None
    n: float = None
    ap1: ArrheniusProcess = LogLinearArrhenius()
    ap2: ArrheniusProcess = LogLinearArrhenius()
    
    def _func(self,x:np.ndarray,*params):
        return self.rxFunc(x[:,0],self.ap1(x[:,1]),self.ap2(x[:,1]),params[0])
    
    def _optimize_func(self,x:np.ndarray,*params):  
        self.set_parameters(np.array(params))
        return self.rxFunc(x[:,0],self.ap1(x[:,1]),self.ap2(x[:,1]),params[0])

    def predict(self,x : np.ndarray):
        return self._func(x,*self.parameters())
    
    def parameters(self):
        return np.concatenate([[self.n],self.ap1.params,self.ap2.params])
    
    def set_parameters(self,params: np.ndarray):
        self.n = params[0]
        self.ap1.params = params[1:3]
        self.ap2.params = params[3:]
    
    def fit(self,rxFunc: Callable,
                t: np.ndarray,
                 T: np.ndarray,
                 Y: np.ndarray,
                 opt_method: Callable,
                 bounds: Tuple[np.ndarray],
                 **kwargs):
                
        self.rxFunc = rxFunc
        opt_res = opt_method(self._optimize_func,np.array([t,T]).T, Y.copy(),bounds,**kwargs)
        
        if isinstance(opt_res,OptimizeResult):
            x,flag,msg = opt_res.x.copy(),opt_res.success,opt_res.message
        else:
            x,flag,msg = opt_res
        
        if flag:
            self.set_parameters(x)
            return self
        else:
            raise ValueError(f'Optimization failed: {msg}')

def get_arrhenius_process_params(ap: LogLinearArrhenius) -> np.ndarray:
    return ap.params[0],ap.params[1]

def get_model_ap_params(model: LogLinearArrheniusModelFunc) -> np.ndarray:
    return (*get_arrhenius_process_params(model.ap1),*get_arrhenius_process_params(model.ap2))

def get_model_params(model: LogLinearArrheniusModelFunc) -> np.ndarray:
    return *get_model_ap_params(model),model.n

def hdi(samples_: np.ndarray,alpha: int) -> np.ndarray:
    """
    Compute the highest density interval at level alpha
    based upon samples from the distribution provided by "samples"
    along the last axis of the array

    Parameters
    ----------
    samples : np.ndarray
        Samples from the distribution of interest
    alpha : int

    Returns
    -------
    np.ndarray
        The HDI at level alpha

    """

    samples = samples_.reshape([-1,samples_.shape[-1]])
    samples = np.sort(samples,axis = -1)
    n = samples.shape[-1] 
    n_included = int(np.floor(alpha*n))
    n_intervals = n - n_included
    interval_width = samples[:,n_included:] - samples[:,:n_intervals]
    min_idx = np.argmin(interval_width,axis = -1)
    hdi_min = samples[np.arange(samples.shape[0],dtype = int),min_idx]
    hdi_max = samples[np.arange(samples.shape[0],dtype = int),min_idx+n_included]   

    return np.stack([hdi_min,hdi_max],axis = -1)