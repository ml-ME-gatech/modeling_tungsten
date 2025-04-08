from typing import Callable, Any
import pickle
from dataclasses import dataclass
from pathlib import PurePath
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import math
import sys
if 'win' in sys.platform.lower():
    from pathlib import WindowsPath as Path
else:
    from pathlib import PosixPath as Path

from recrystallization.common_util import LogLinearArrheniusModelFunc,LogLinearArrhenius,jmak_function,get_model_params
from recrystallization import common_util as rx_common_util
from engineering_models import common_util as em_common_util
from abc import ABC,abstractmethod
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import bisect
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

_EXT_PARENT_DIR = Path(__file__).resolve().parent.parent 
_ENGINEERING_MODEL_PATH = _EXT_PARENT_DIR.joinpath('engineering_models/.model')
_RX_INF_PATH = _EXT_PARENT_DIR.joinpath('recrystallization/.inference')

"""
Authors: Michael Lanahan

Overview
---------

This file contains a collection of convinience and utility functions and classes to facilitate use of the
engineering material and recrystallization models developed in the subfolders in this project. The code here 
is not meant to be particuarly generalizable or used as a library, but mostly as a mechanism for (me) to easily
use these results in other projects. If you feel they may be useful to you, feel free to use them as you see fit.
but no guarantees are made as to their correctness or usability.
"""

def _reconcile_rx_model_shape(t: np.ndarray | float,
                              T: np.ndarray | float) -> tuple[np.ndarray | float,np.ndarray | float]:
    """ 
    Rather than rely on numpy broadcasting rules, just get temperature and time to be the same shape

    Parameters
    ----------
    t : np.ndarray | float
        Time
    T : np.ndarray | float
        Temperature
    """
    if isinstance(t, np.ndarray) and isinstance(T, np.ndarray):
        assert t.shape == T.shape, f"Invalid shape: {t.shape} must be same as {T.shape}"
    elif isinstance(t, np.ndarray):
        T = np.full_like(t,T)
    elif isinstance(T, np.ndarray):
        t = np.full_like(T,t)
    else:
        raise TypeError('Unsupported type for t and/or T')
    return t,T

def _default_ode_kwargs(**kwargs):
    """ 
    favor precision over speed.
    """

    default_kwargs = {'method':'DOP853',
                      'rtol':1e-12,
                      'atol':1e-12}
    
    default_kwargs.update(kwargs)
    return default_kwargs

class NonisothermalJMAK:

    def __init__(self,model: LogLinearArrheniusModelFunc):
    
        self.a1,self.B1,self.a2,self.B2,self.n = get_model_params(model)
        self.model = model

    def effective_rate(self,t: float,T: Callable) -> float:
        
        def bfunc(Temp: float):
            return math.exp(self.a1 + self.B1/Temp)
        
        integral = quad(lambda x: bfunc(T(x)),0,t)[0]
        return integral/t
    
    def effective_incubation_time(self,t1: float, t2: float, T: Callable) -> float:
        
        def tinc(Temp: float):
            return math.exp(self.a2 + self.B2/Temp)
        
        def rho_func(tt: float):
            return quad(lambda x: 1./tinc(T(x)),0,tt)[0] - 1.   
        
        return bisect(rho_func,t1,t2)
    
    def get_rx_time(self,X: float, temperature: np.ndarray) -> np.ndarray:
        
        t_inc = np.exp(self.a2 + self.B2/temperature)
        b = np.exp(self.a1 + self.B1/temperature)
        return 1.0/b*np.log(1./(1. - X))**(1/self.n) + t_inc  

    def get_rx_temperature(self,X: float, time: np.ndarray,
                            temp_lim = [873.15, 2000]) -> np.ndarray:

        
        temperature = np.zeros_like(time)
        for i in range(time.shape[0]):
            def _solve_temperature(temp: float):
                t = time[i] - np.exp(self.a2 + self.B2/temp)
                if t < 0:
                    return 1e6

                w = 1./t*math.log(1./(1.-X))**(1/self.n)
                T = self.B1/(math.log(w) - self.a1)
                return T - temp

            temperature[i] = bisect(_solve_temperature,temp_lim[0],temp_lim[1])
        
        return temperature

    def nonisothermal_rx_fraction(self,t: np.ndarray,
                                       T: Callable,
                                       **solver_kwargs) -> np.ndarray:

        def b(Temp: float): 
            return math.exp(self.a1 + self.B1/Temp)
        
        start_t = self.effective_incubation_time(t[0],t[-1],T)

        def integrand(tt: float, _ ):
            TT = T(tt + start_t)
            BB = b(TT)
            return BB
        
        Y = np.zeros_like(t)
        index = t >start_t
        solver_kwargs = _default_ode_kwargs(**solver_kwargs)
        Z = solve_ivp(integrand,(t[0],t[-1]),np.array([1e-18]),
                            t_eval = (t[index] - start_t),**solver_kwargs)['y'].squeeze()**self.n  

        Y[index] = 1. - np.exp(-Z)
        return Y
    
    def isothermal_rx_fraction(self, t: np.ndarray | float, T: float | np.ndarray) -> np.ndarray:
        
        t_,T_ = _reconcile_rx_model_shape(t,T)
        return self.model.predict(np.array([t_,T_]).T).squeeze()
    
    def __call__(self, t: np.ndarray | float,
                       T: Callable | float,
                       **solver_kwargs) -> Callable:
        
        if callable(T) :
            assert(isinstance(t,np.ndarray)), "Time must be an array if temperature is a function"
            return self.nonisothermal_rx_fraction(t,T, **solver_kwargs)
        elif isinstance(T,float):
            return self.isothermal_rx_fraction(t,T)
        elif isinstance(T,np.ndarray):
            if isinstance(t,np.ndarray):
                assert t.shape == T.shape, f"Invalid shape: {t.shape} must be same as {T.shape}"
            
            return self.isothermal_rx_fraction(t,T)
        else:
            raise TypeError('Unsupported type for T')


def structural_model_to_jmak_params(latent_param: np.ndarray,
                                    latent_var: np.ndarray): 
    
    a1,B1,n = latent_var[0]*latent_param[0:3,1] + latent_param[0:3,0]
    a2,B2 = latent_var[1]*latent_param[3:,1] + latent_param[3:,0]

    return a1,B1,a2,B2,n

def read_jmak_model_inference(file_: str | PurePath,
                              estimate = 'ml') -> NonisothermalJMAK:
    """
    Helper function to read a callable JMAK model from an inference file containing parameters.

    Parameters
    ----------
    file : str | PurePath
        Path to the inference file, should probably reside in the .inference directory in recrystallization subfolder
    estimate : str
        Estimate to use for the parameters. Default is 'ml' for maximum likelihood.
    """

    file = Path(file_).resolve()

    if '.csv' == file.suffix:
        inf_summary = pd.read_csv(file,index_col = 0)
        model_params = ['a1','B1','a2','B2','n','sigma'] 
        a1,B1,a2,B2,n,_ = inf_summary.loc[model_params,estimate].to_numpy()
    elif '.pkl' == file.suffix:
        if estimate =='ml':
            estimate = 'map'
        with open(file,'rb') as f:
            inf_summary = pickle.load(f)[estimate]
            a1,B1,a2,B2,n = structural_model_to_jmak_params(inf_summary['latent_param'].reshape([5,2]),
                                                            inf_summary['latent_variables'])
    else:
        raise ValueError('Unsupported file type: must be either .csv or .pkl')
    
    b = LogLinearArrhenius(np.array([a1,B1]))
    tinc = LogLinearArrhenius(np.array([a2,B2]))
    return NonisothermalJMAK(
        LogLinearArrheniusModelFunc(jmak_function,n = n,ap1 = b,ap2 = tinc)
        )

def is_sklearn(model: Any): 
    """
    utility to check of a model is a sklearn model
    """
    return isinstance(model, Pipeline) or isinstance(model, BaseEstimator)

def _coerce_sklearn_2d_shape(X: np.ndarray) -> np.ndarray:
    """
    Utility to coerce a 1D array to a 2D array for prediction using sklearn models
    """
    if X.ndim == 1:
        return X.reshape(-1, 1)
    elif X.ndim == 2:
        return X
    else:
        raise ValueError(f"Invalid shape: {X.shape} must be 1D or 2D")
    
def _call_or_predict(model: Callable | Pipeline | BaseEstimator,
                      *args, 
                      **kwargs):
    """
    Utility to call a model or predict using an object that is either callable e.g.
    numpy.Polynomial, interpolation, ect..., or a sklearn model that has a predict method
    e.g. sklearn.linear_model.LinearRegression, sklearn.pipeline.Pipeline, ect...

    Parameters
    ----------
    model : Callable | Pipeline | BaseEstimator
        Model to call or predict with
    *args : tuple
        Arguments to pass to the model __call__ or predict methods
    **kwargs : dict
        Keyword arguments to pass to the model __call__ or predict methods
    """

    if callable(model):
        return model(*args, **kwargs)
    elif is_sklearn(model):
        args = list(args)
        X = _coerce_sklearn_2d_shape(args.pop(0))
        return model.predict(X,*args, **kwargs)
    else:
        raise ValueError(f"Invalid model type: {type(model)} must be callable or sklearn model")

class LoadableMaterialProperty:

    """
    Class for conviniently storing and loading material property
    that could be callable or could be loaded from a file. Sometimes 
    I want to pass a file, sometimes I want to pass a callable 

    Parameters
    ----------
    file : str
        Path to the file containing the material property
    material_property : Callable
        Material property
    module : Any    
        Module to use for loading the material property
    """

    def __init__(self,file: str = None,
                      material_property: Callable = None,
                      module: Any = em_common_util):    
        
        self.file = file
        self.material_property = material_property
        self.module = module
    
    def load(self):
        if self.file:
            with open(self.file,'rb') as f:
                try:
                    old_module = sys.modules['common_util']
                except KeyError:
                    old_module = None

                sys.modules['common_util'] = self.module 
                self.material_property = pickle.load(f)
                sys.modules['common_util'] = old_module
        else:
            raise ValueError("No file specified")
    
    def save(self):
        if self.file:
            with open(self.file,'wb') as f:
                pickle.dump(self.material_property,f)
        else:
            raise ValueError("No file specified")
    
    def __call__(self,*args, **kwargs):
        if self.material_property:
            return _call_or_predict(self.material_property,*args, **kwargs)
        else:
            self.load()
            if not self.material_property:
                raise ValueError("No material property specified")
            return _call_or_predict(self.material_property,*args, **kwargs)
    
    @classmethod
    def from_item(cls,item: str | PurePath | Callable,
                        module = em_common_util) -> 'LoadableMaterialProperty':
        """
        Convinience method for creating a LoadableMaterialProperty from a farily generic item
        Parameters
        ----------
        item : str | PurePath | Callable
            Item to create the LoadableMaterialProperty from
        module : Any
            Module to use for loading the material property
        """

        if isinstance(item,str) or isinstance(item,PurePath):
            return cls(file = item,module = module)
        elif isinstance(item,LoadableMaterialProperty):
            return item
        elif callable(item):
            return cls(material_property = item,module = module)
        elif item is not None:
            raise TypeError(f"Invalid type for {item}: must be str or Callable")

class RecrystallizedModel(ABC):

    def _parse_material_property_item(self,item : LoadableMaterialProperty | str | Callable,
                                           module = em_common_util) -> LoadableMaterialProperty:
        return LoadableMaterialProperty.from_item(item,module = module) 

    @abstractmethod
    def __call__(self,time: np.ndarray | float,temperature: np.ndarray) -> np.ndarray:
        pass

class RecrystallizedUltimateTensileStress(RecrystallizedModel):

    r"""
    Class for recrystallized ultimate tensile stress
    $$ 
    S_{u,rx} = S_{u}(T) - X(t,T) \Delta_{rx} S_{u}
    $$ 

    where $S_{u,rx}$ is the recrystallized ultimate tensile stress, $S_{u}(T)$ is the ultimate tensile stress at temperature $T$,
    $X(t,T)$ is the recrystallization fraction at time $t$ and temperature $T$, and $\Delta_{rx} S_{u}$ is the difference between the ultimate tensile stress and the recrystallized ultimate tensile stress.

    Parameters
    ----------
    ultimate_tensile_stress : LoadableMaterialProperty
        Ultimate tensile stress
    recrystallization_fraction : LoadableMaterialProperty
        Recrystallization fraction
    delta_uts : float
        Difference between the ultimate tensile stress and the recrystallized ultimate tensile stress
    """

    def __init__(self,ultimate_tensile_stress: LoadableMaterialProperty | str | Callable,
                      recrystallization_fraction: LoadableMaterialProperty | str | Callable,
                      delta_uts: float):
        
        self.ultimate_tensile_stress = self._parse_material_property_item(ultimate_tensile_stress,
                                                                          module = em_common_util)
        self.recrystallization_fraction = self._parse_material_property_item(recrystallization_fraction,
                                                                             module = rx_common_util)
        self.delta_uts    = delta_uts

    def __call__(self,time: np.ndarray | float,temperature: np.ndarray) -> np.ndarray:
        if isinstance(time, np.ndarray):
            assert time.shape == temperature.shape, f"Invalid shape: {time.shape} must be same as {temperature.shape}"
        
        return self.ultimate_tensile_stress(temperature) - \
                self.recrystallization_fraction(time,temperature + 273.15) * self.delta_uts

class RecrystallizedTotalElongation(RecrystallizedModel):
    r"""
    Class for recrystallized total elongation
    $$
    \varepsilon_{t,rx} = \varepsilon_{t}(T) - X(t,T) \Delta_{rx} \varepsilon_{t}
    $$

    where $\varepsilon_{t,rx}$ is the recrystallized total elongation, $\varepsilon_{t}(T)$ is the total elongation at temperature $T$,
    $X(t,T)$ is the recrystallization fraction at time $t$ and temperature $T$, and $\Delta_{rx} \varepsilon_{t}$ is the difference between the total elongation and the recrystallized total elongation.
    
    Parameters
    ----------
    total_elongation : LoadableMaterialProperty
        Total elongation
    recrystallization_fraction : LoadableMaterialProperty
        Recrystallization fraction
    delta_te : float
        Difference between the total elongation and the recrystallized total elongation
    """

    def __init__(self,total_elongation: LoadableMaterialProperty | str | Callable,
                      recrystallization_fraction: LoadableMaterialProperty | str | Callable,
                      delta_te: float):
        
        self.total_elongation = self._parse_material_property_item(total_elongation,
                                                                    module = em_common_util)
        self.recrystallization_fraction = self._parse_material_property_item(recrystallization_fraction,
                                                                             module = rx_common_util)    
        self.delta_te    = delta_te
    
    def __call__(self,time: np.ndarray | float,temperature: np.ndarray) -> np.ndarray:  
        if isinstance(time, np.ndarray):
            assert time.shape == temperature.shape, f"Invalid shape: {time.shape} must be same as {temperature.shape}"
        
        return self.total_elongation(temperature) - \
                self.recrystallization_fraction(time,temperature + 273.15) * self.delta_te
    
class RecrystallizedUniformElongation(RecrystallizedModel):

    r"""
    Class for recrystallized uniform elongation
    $$
    \begin{matrix*}[l]
    \varepsilon_{u,rx} = \varepsilon_{u}(T) - X(t,T) \Delta_{rx} \varepsilon_{u}(T) \\ 
    \Delta_{rx} \varepsilon_{u}(T) = frac{\varepsilon_{u,0}(T)}{\varepsilon_{tr,0}(T)} \Delta_{rx} \varepsilon_{tr}
    \end{matrix*}
    $$

    where $\varepsilon_{u,rx}$ is the recrystallized uniform elongation, $\varepsilon_{u}(T)$ is the uniform elongation at temperature $T$,
    $X(t,T)$ is the recrystallization fraction at time $t$ and temperature $T$, and $\Delta_{rx} \varepsilon_{u}$ is the difference between the uniform elongation and the recrystallized uniform elongation.
    
    Parameters
    ----------
    uniform_elongation : LoadableMaterialProperty
        Uniform elongation
    recrystallization_fraction : LoadableMaterialProperty
        Recrystallization fraction
    delta_ue : float
        Difference between the uniform elongation and the recrystallized uniform elongation
    """

    def __init__(self,uniform_elongation: LoadableMaterialProperty | str | Callable,
                      total_elongation: LoadableMaterialProperty | str | Callable,
                      recrystallization_fraction: LoadableMaterialProperty | str | Callable,
                      delta_te: float):
        
        self.uniform_elongation = self._parse_material_property_item(uniform_elongation,
                                                                    module = em_common_util)
        self.total_elongation = self._parse_material_property_item(total_elongation,
                                                                    module = em_common_util)
        self.recrystallization_fraction = self._parse_material_property_item(recrystallization_fraction,
                                                                             module = rx_common_util)
        self.delta_te    = delta_te
    
    def __call__(self,time: np.ndarray | float,temperature: np.ndarray) -> np.ndarray:
        if isinstance(time, np.ndarray):
            assert time.shape == temperature.shape, f"Invalid shape: {time.shape} must be same as {temperature.shape}"
        
        delta_ue = self.delta_te * self.uniform_elongation(temperature) / self.total_elongation(temperature)
        return self.uniform_elongation(temperature) - \
                self.recrystallization_fraction(time,temperature + 273.15) * delta_ue

@dataclass  
class ElasticMaterialModel:

    """
    Class for conviniently storing and loading material properties. There are a bunch of material properties
    required, that all tend to be temperature or time or both depednent. This can be be a real 
    headache to deal with, so this class bundles them all up and provides a convinient interface to access them.

    Parameters
    ----------
    ultimate_tensile_stress : str | Callable
        Ultimate tensile stress
    uniform_elongation : str | Callable
        Uniform elongation
    total_elongation : str | Callable
        Total elongation
    youngs_modulus : str | Callable
        Young's modulus
    conductivity : str | Callable
        Conductivity
    coefficient_of_thermal_expansion : str | Callable
        Coefficient of thermal expansion
    """

    ultimate_tensile_stress: str | Callable = None
    uniform_elongation: str | Callable = None
    total_elongation: str | Callable = None
    youngs_modulus: str | Callable = None   
    conductivity: str | Callable = None
    coefficient_of_thermal_expansion: str | Callable = None

    def __post_init__(self):
        items = ['ultimate_tensile_stress','uniform_elongation',
                 'total_elongation','youngs_modulus',
                 'conductivity','coefficient_of_thermal_expansion']
        for item in items:
            atrr_ = getattr(self,item)
            if isinstance(atrr_,LoadableMaterialProperty):
                pass
            elif isinstance(atrr_,str) or isinstance(atrr_,PurePath):
                setattr(self,item,LoadableMaterialProperty(file = atrr_))
            elif callable(atrr_):
                setattr(self,item,LoadableMaterialProperty(material_property = atrr_))
            elif atrr_ is not None:
                raise TypeError(f"Invalid type for {item}: must be str or Callable")

def get_nogami_material_property(material: str):

    """
    Function to get the material property from the Nogami database. For nogami's data specifically, multiple 
    material properties are provided for each material, so this is a convinience function fro retrieving the
    whole elastic material model. 

    Note that the elastic/young's modulus is consistent across alloys, and the coefficient 
    of thermal expansion is not provided in the database, so these must be set a posteriori.

    Parameters
    ----------
    material : str
        Material name. Check in the engineering_models/.model directory for the available materials 
    """

    _material = material
    emm = ElasticMaterialModel(ultimate_tensile_stress = _ENGINEERING_MODEL_PATH.joinpath(f'{_material}_uts.pkl'),
                               uniform_elongation = _ENGINEERING_MODEL_PATH.joinpath(f'{_material}_ue.pkl'),
                               total_elongation = _ENGINEERING_MODEL_PATH.joinpath(f'{_material}_te.pkl'),
                               youngs_modulus = _EXT_PARENT_DIR.joinpath(f'_external/.model/youngs_modulus'),
                               conductivity = _ENGINEERING_MODEL_PATH.joinpath(f'{_material}_k.pkl'))
    
    return emm

def get_nogami_material_recrystallization_property(material: str,
                                                   rx_material: str,
                                                   delta_uts: float,
                                                   delta_te: float,
                                                   rx_model =None):
    
    """ 
    Function to get the material property from the Nogami database with an accompying recrystallization
    model.

    Parameters
    ----------
    material : str
        Material name. Check in the engineering_models/.model directory for the available materials
    rx_material : str
        Material name for the recrystallization model. Check in the recrystallization/.inference directory for the available materials
    delta_uts : float
        Difference between the ultimate tensile stress and the recrystallized ultimate tensile stress
    delta_te : float
        Difference between the total elongation and the recrystallized total elongation
    """

    emm = get_nogami_material_property(material)
    try:
        rx_model = read_jmak_model_inference(_RX_INF_PATH.joinpath(f'JMAK_{rx_material}_trunc_normal_params.csv')) if rx_model is None else rx_model
    except FileNotFoundError:
        rx_model = read_jmak_model_inference(_RX_INF_PATH.joinpath(f'{rx_material}_tpa_map_trunc_normal.pkl')) if rx_model is None else rx_model
    
    uts = RecrystallizedUltimateTensileStress(emm.ultimate_tensile_stress,
                                              LoadableMaterialProperty(material_property= rx_model),
                                              delta_uts)
    te = RecrystallizedTotalElongation(emm.total_elongation,
                                        LoadableMaterialProperty(material_property= rx_model),
                                        delta_te)
    ue = RecrystallizedUniformElongation(emm.uniform_elongation,
                                        emm.total_elongation,
                                        LoadableMaterialProperty(material_property= rx_model),
                                        delta_te)
    return ElasticMaterialModel(ultimate_tensile_stress = uts,
                               uniform_elongation = ue,
                               total_elongation = te,
                               youngs_modulus = emm.youngs_modulus,
                               conductivity = emm.conductivity,
                               coefficient_of_thermal_expansion = emm.coefficient_of_thermal_expansion)

def main():

    #rx_model =  read_jmak_model_inference(_RX_INF_PATH.joinpath(f'JMAK_Lopez et al. (2015) - MR_trunc_normal_params.csv'))
    rx_model = read_jmak_model_inference(_RX_INF_PATH.joinpath('K-doped 3%Re W_tpa_map_trunc_normal.pkl'))
    T = 1200.0 + 273.15
    t = np.linspace(0,100*3600,4)
    rx = rx_model(t,T)
    
    T = lambda x: 1/3600*x + 1150 + 273.15
    t = np.linspace(0,100*3600,4)
    rx = rx_model(t,T)
    b = rx_model.effective_rate(t[-1],T)


if __name__ == "__main__":
    main()