import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.stats import t as tdist
from abc import abstractmethod,ABC
from statsmodels.regression.linear_model import OLS, OLSResults
from typing import Tuple, Callable
from scipy.optimize import minimize_scalar, OptimizeResult
from dataclasses import dataclass

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



        