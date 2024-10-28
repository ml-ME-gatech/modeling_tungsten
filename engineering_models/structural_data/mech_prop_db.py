import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
from pathlib import WindowsPath
import os

_PATH = WindowsPath(__file__).parent.joinpath('./')

#from ITERs documentation
_POLYNOMIALS = {'minimum_uniform_elongation_w':Polynomial(np.flip(np.array([-1.11e-5,9.39e-3,-0.1713]))),
                'yield_strength':Polynomial(np.flip(np.array([2.979e-7,-1.176e-3,1.112,1.305e2]))),
                'ultimate_tensile_strength':Polynomial(np.flip(np.array([-1.006e-7,4.172e-4,-8.485e-1,8.707e2]))),
                'minimum_uniform_elongation_wl10': Polynomial([-4.08570255e+01,1.94060425e-01,-2.16214380e-04,7.77108227e-08])}


class InterpolatedProperty:

    """
    A class to interpolate material properties from a csv file
    """
    def __init__(self,file: str,
                      poly_order = None):

        if os.path.exists(file):
            self.file = file
        else:
            self.file = _PATH.joinpath(file)
        
        self.data = None
        self.poly_order = poly_order

    
    def _read_data(self):
        self.data = pd.read_csv(self.file,index_col = 0,header = 0).astype(float)
        self.data.sort_index(inplace= True)
        self._x = self.data.index.to_numpy()
        self._y = self.data.to_numpy().squeeze()
    
    def __call__(self,T: np.ndarray,
                      return_data = False,
                      **kwargs) -> np.ndarray:

        if self.data is None:
            self._read_data()
        
        if self.poly_order is None:
            kwargs['assume_sorted'] = True

            if self._y.ndim != 1:
                output = []
                for i in range(self._y.shape[1]):
                    output.append(interp1d(self._x,self._y[:,i],**kwargs)(T)[:,None])
            
                output = np.concatenate(output,axis = 1)
            else:
                output = interp1d(self._x,self._y,**kwargs)(T)
        
        else:
            p = Polynomial.fit(self._x,self._y,self.poly_order)
            output = p(T)

        if return_data is False:
            return output
        else:
            return output,self.data


def polyprop(polynominal_function: callable) -> callable:

    def wrapped_poly(T: np.ndarray) -> np.ndarray:

        return _POLYNOMIALS[polynominal_function.__name__](T)
    
    return wrapped_poly

def true_strain_at_rupture(T: np.ndarray,
                           return_data = False,
                           poly_order = None,
                           **kwargs) -> np.ndarray:
    """
    the true strain at rupture of W 

    Taken from 

    Structural Design Critieria for ITER in-Vessel Components (SDC-IC)
    Appendix A: Material Design Limit Data
    """

    ip = InterpolatedProperty('w_iter_true_strain_at_rupture',poly_order = poly_order)
    return ip(T,return_data = return_data,**kwargs)

def youngs_modulus(T: np.ndarray,
                   return_data = False,
                   poly_order = None,
                  **kwargs) -> np.ndarray:
    """
    The elastic modulus (or "young's modulus") of WL10.

    Found in Norajitra's thesis - pruported to be available in
    the ITER Material Property Handbook
    """
    ip = InterpolatedProperty('youngs_modulus',poly_order = poly_order)
    return ip(T,return_data = return_data,**kwargs)

def minimum_uniform_elongation(T: np.ndarray,
                               return_data = False,
                                poly_order = None,
                                **kwargs) -> np.ndarray:
    """
    the minimum uniform elongation of W 

    Taken from 

    Structural Design Critieria for ITER in-Vessel Components (SDC-IC)
    Appendix A: Material Design Limit Data
    """
    kwargs['fill_value'] = -np.inf
    kwargs['bounds_error'] = False

    ip = InterpolatedProperty('w_iter_uniform_elongation',poly_order = poly_order)
    return ip(T,return_data = return_data,**kwargs)

@polyprop
def yield_strength(T: np.ndarray) -> np.ndarray:
    """
    The yield strength of WL10

    Found in Norajitra's thesis - pruported to be available in
    the ITER Material Property Handbook
    """
    pass

@polyprop
def ultimate_tensile_strength(T: np.ndarray) -> np.ndarray:
    """
    The ultimate tensile stress of WL10

    Found in Norajitra's thesis - pruported to be available in
    the ITER Material Property Handbook
    """
    pass

def ultimate_tensile_strength_w(T: np.ndarray,
                               return_data = False,
                                poly_order = None,
                                **kwargs) -> np.ndarray:
    """
    ultimate tensile stress of W from ITER MPH
    """

    kwargs['fill_value'] = -np.inf
    kwargs['bounds_error'] = False

    ip = InterpolatedProperty('w_iter_uts',poly_order = poly_order)
    return ip(T,return_data = return_data,**kwargs)
    
@polyprop
def minimum_uniform_elongation_wl10(T: np.ndarray) -> np.ndarray:
    """
    The minimum uniform elongation of WLL10

    From Davis et al (1998)
    "Assessment of tungsten for use in the ITER plasma facing
    components"
    """

    pass
        
