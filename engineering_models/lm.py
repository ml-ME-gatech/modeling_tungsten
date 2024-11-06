from abc import ABC,abstractmethod
import numpy as np
import scipy
import scipy.linalg
from typing import Callable
from functools import cached_property
import scipy.stats

class StandardizedScale:

    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self,X: np.ndarray):
        self.mean_ = X.mean(axis = 0)[np.newaxis,:]
        self.std_ = np.std(X - self.mean_,axis = 0)[np.newaxis,:]
        return self.scale(X)
    
    def scale(self,X: np.ndarray):
        return (X - self.mean_)/self.std_
    
    def unscale(self,X: np.ndarray):
        return self.std_ *X + self.mean_

class OLSFit(ABC):

    def __init__(self,alpha: np.ndarray | float = None):
        self.X = None,
        self.Y = None,
        self.coeff_ = None
        self.alpha = alpha
        self.regularize = False if self.alpha is None else True

    def regularization(self):
        XTX = self.X.T @ self.X
        XTX.ravel()[::XTX.shape[0] + 1] += self.alpha
        return XTX

    @abstractmethod
    def fit(self,X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        pass

    def scale_coeff_var(self) -> np.ndarray:
        return np.diag(self.xTx_inv)

    @cached_property
    def hat_matrix(self) -> np.ndarray:
        return self.X @ self.xTx_inv @ self.X.T
    
    @property
    def xTx_inv(self):
        return self._xTx_inv
    
    def inv_qform(self,U: np.ndarray) -> np.ndarray:
        """ 
        compute U.T @ (X.T @ X)^-1 @ U.T
        """
        return U.T @ self.xTx_inv @ U

    def confidence_interval(self,sigma: float,alpha: float) -> np.ndarray:
        t = np.squeeze(sigma*self.scale_coeff_var()*scipy.stats.t.ppf(1-alpha/2,self.X.shape[0] - self.X.shape[1]))
        return np.column_stack((self.coeff_.squeeze() - t,self.coeff_.squeeze() + t))

def md_property(func: Callable) -> Callable:

    @property
    def matrix_property(self) -> np.ndarray:
        pname = func.__name__
        if self.__getattribute__('_' + pname) is None:
            self.decomposition()
        return self.__getattribute__('_' + pname)
    
    return matrix_property

class PinvFit(OLSFit):

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

    def fit(self,X:np.ndarray,Y: np.ndarray) -> np.ndarray:
        self.X,self.Y = X,Y
        xTx = self.regularization() if self.regularize else self.X.T @ self.X
        self._xTx_inv = scipy.linalg.pinv(xTx)
        self.coeff_ = self.xTx_inv @ self.X.T @ self.Y
        return self.coeff_
    
class QRFit(OLSFit):

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)
        self._Q,self._R = None,None
    
    def decomposition(self):
        self._Q,self._R = scipy.linalg.qr(self.X,mode= 'economic')

    @md_property
    def Q(self):
        pass

    @md_property
    def R(self):
        pass
    
    def regularization(self):
        raise NotImplementedError('regularization not implemented using QR')
    
    @cached_property
    def _xTx_inv(self):
        y1 = scipy.linalg.solve_triangular(self.R.T,np.eye(self.R.shape[0]),lower = True)
        return scipy.linalg.solve_triangular(self.R,y1)

    def fit(self,X:np.ndarray,Y: np.ndarray) -> np.ndarray:
        self.X,self.Y = X,Y
        self.coeff_ = scipy.linalg.solve_triangular(self.R,self.Q.T @ self.Y)
        return self.coeff_

class CholeskyFit(OLSFit):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._U = None

    @md_property
    def U(self) -> np.ndarray:
        pass

    def decomposition(self):
        xTx = self.regularization() if self.regularize else self.X.T @ self.X
        self._U = scipy.linalg.cholesky(xTx)
    
    @cached_property
    def _xTx_inv(self):
        return scipy.linalg.cho_solve((self.U,False),np.eye(self.U.shape[0]))
    
    def fit(self,X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        self.X,self.Y = X,Y
        self.coeff_ = scipy.linalg.cho_solve((self.U,False),self.X.T @ self.Y)
        return self.coeff_
    
    
_FIT_MODELS = {'qr': QRFit,
               'pinv': PinvFit,
               "cholesky": CholeskyFit}

class LinearModel(ABC):

    def __init__(self,*args,fit_offset = True,**kwargs):
        
        self.X_ = None
        self.Y_ = None
        self.coeff_ = None
        self.offset_ = None
        self.fit_offset = fit_offset
    
    def _fit_offset(self,Y: np.ndarray) -> float:
        
        self.offset_ = Y.mean()
        return Y - self.offset_ 

    @abstractmethod
    def fit(self,X: np.ndarray,Y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray,Y: np.ndarray):
        pass

    def residuals(self) -> np.ndarray:
        return self.Y_ - self.X_ @ self.coeff_

    def sigma2(self) -> float:
        return 1./(self.X_.shape[0] - self.X_.shape[1])*np.sum(self.residuals()**2)


class OLS(LinearModel):

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)
        self.fit_method = None

    def _fit(self,X: np.ndarray,Y: np.ndarray):
        self.X_ = X.copy()
        self.Y_ = self._fit_offset(Y) if self.fit_offset else Y.copy()
        self.coeff_ = self.fit_method.fit(self.X_,self.Y_)
        return self.fit_method
    
    def fit(self,X: np.ndarray,Y: np.ndarray, method = 'qr') -> OLSFit:
        if method not in _FIT_MODELS:
            raise ValueError('method must be one of "qr", "pinv", "cholesky"')
        
        self.fit_method = _FIT_MODELS[method]()
        return self._fit(X,Y)

    def prediction_confidence_interval(self,X: np.ndarray,alpha: float):
        return self.sigma2()**0.5*np.sqrt(1. +  np.sum(X.T*(self.fit_method.xTx_inv @ X.T),axis = 0))*scipy.stats.t.ppf(alpha/2,self.X_.shape[0] - self.X_.shape[1])
    
    def coeff_confidence_interval(self,alpha: float) -> np.ndarray:
        return self.fit_method.confidence_interval(self.sigma2(),alpha)

    def predict(self,X: np.ndarray):
        return X @ self.coeff_ + self.offset_ if self.fit_offset else X @ self.coeff_

class LinearEqualityConstrainedLS(OLS):

    """
    Linearly equality constrained least squares. For now I'm just brute forcing
    the constrained portion of the solution. There may be a more effecient
    and/or elegant way to do this
    """
    def _fit(self,X: np.ndarray,Y: np.ndarray, L: np.ndarray,b: np.ndarray):
        self.X_ = X.copy()
        self.Y_ = self._fit_offset(Y) if self.fit_offset else Y.copy()
        self.coeff_ = self.fit_method.fit(self.X_,self.Y_)
        constrained_part = self.fit_method.xTx_inv @ L.T @ scipy.linalg.solve(L @ self.fit_method.xTx_inv @ L.T,b - L @ self.coeff_)
        self.coeff_ += constrained_part

        return self.fit_method
    
    def fit(self,X: np.ndarray,
                Y: np.ndarray,
                L: np.ndarray,
                b: np.ndarray,
                method = 'qr') -> OLSFit:

        if method not in _FIT_MODELS:
            raise ValueError('method must be one of "qr", "pinv", "cholesky"')
        
        self.fit_method = _FIT_MODELS[method]()
        return self._fit(X,Y,L,b)