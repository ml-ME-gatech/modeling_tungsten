import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.stats import t as tdist

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