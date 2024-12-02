import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.stats import t as tdist
from abc import abstractmethod,ABC
from statsmodels.regression.linear_model import OLS, OLSResults
from typing import Tuple
from scipy.optimize import minimize_scalar

def setup_axis_default(ax: plt.Axes):
    
    """
    convinience function to set up the axis
    """
    ax.tick_params('both',labelsize = 11,which = 'both',direction = 'in')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    return ax

class SimpleLinearRegression:
    """
    class for simple linear regression results, in the style of sklearn
    """
    def __init__(self):
        pass

    def fit(self,x: np.ndarray,y: np.ndarray):
        self.y, self.x = y.copy(),x.copy()
        self.ybar = y.mean()  
        self.xbar = x.mean()
        self.S_xx = np.sum((x - self.xbar)**2)
        self.S_xy = np.sum((x - self.xbar)*(y - self.ybar))
        self.beta_1 = self.S_xy/self.S_xx
        self.beta_0 = self.ybar - self.beta_1*self.xbar
        return np.array([self.beta_0,self.beta_1])

    def __str__(self) ->str: 
        return f"y = {self.beta_0:.2f} + {self.beta_1:.2f}x"
    
    def predict(self,xnew: np.ndarray):
        return self.beta_0 + self.beta_1*xnew
    
    @property
    def n(self):
        return self.x.shape[0]

    @property
    def residuals(self):
        return self.y - (self.beta_0 + self.beta_1*self.x)
    
    @property
    def variance(self):
        return np.sum(self.residuals**2)/(self.residuals.shape[0] - 2)
    
    def beta_confidence_interval(self,alpha: float = 0.05) -> np.ndarray:
        v = tdist.ppf(1 - alpha/2,self.n - 2)
        beta_0_ci = v*(self.variance*(1/self.n + self.xbar**2/self.S_xx))**0.5
        beta_1_ci = v*(self.variance/self.S_xx)**0.5
        return np.array([[self.beta_0 - beta_0_ci,self.beta_0 + beta_0_ci],
                         [self.beta_1 - beta_1_ci,self.beta_1 + beta_1_ci]])    
    
    def predictive_confidence_interval(self,xnew: np.ndarray,alpha: float = 0.05) -> np.ndarray:
        v = tdist.ppf(1 - alpha/2,self.n - 2)
        yhat = self.predict(xnew)
        ci_yhat = v*self.variance*(1 + 1/self.n + (xnew - self.xbar)**2/self.S_xx)**0.5
        return np.array([yhat - ci_yhat,yhat + ci_yhat])  
    

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
                          1/x[:,np.newaxis]**self.params[2]])
    
    def log_predict(self,x: np.ndarray):
        return self.tform(x).dot(self.params[:-1])
    
    def parameter_confidence_interval(self,alpha: float):
        params_ = self.params.copy()

        def _func(beta: float):
            self.params[2] = beta
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
        self.params = params_
        return ci

    def fit(self,x: np.ndarray,
                 y: np.ndarray,
                 beta_init: float,
                 beta_bounds: Tuple[float] = (0.1,2.0)):
        
        self.x = x.copy()

        self.params = np.zeros(3)
        self.params[2] = beta_init
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
    
        


            




        