import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple, Callable, Any    
from sklearn.preprocessing import MinMaxScaler  
from sklearn.linear_model import LassoLarsIC, LassoLarsCV, LinearRegression
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from functools import partial
from sklearn.pipeline import Pipeline
from numpy.polynomial import Polynomial

"""
Common utilities used among the jupyter notebooks in this folder
"""

class NogamiData:

    """
    convenience class to access Nogami data, and return it as numpy arrays
    """

    def __init__(self, key: str,
                       data_file: str):

        self.key = key
        self.data = pd.read_csv(data_file,index_col = 0)
    
    def __getitem__(self,key: str) -> Tuple[np.ndarray,np.ndarray]:

        data = self.data[[key]]
        data.dropna(inplace = True)
        return data.index.to_numpy()[:,np.newaxis],data[key].to_numpy()
    
    def keys(self):
        return [column for column in self.data.columns if self.key in column]
    
class NogamiUTSData(NogamiData):

    def __init__(self):
        super().__init__('UTS','structural_data/nogami_data.csv')

class NogamiUEData(NogamiData):

    def __init__(self):
        super().__init__('UE','structural_data/nogami_data.csv')

class NogamiConductivityData(NogamiData):

    def __init__(self):
        super().__init__('','conductivity_data/nogami_data.csv')


class TransformedFeature:
    """ 
    class for representing a feature transform, optionally with its derivative
    we could easily do this using some sort of autodiff, but this is relatively simply
    """

    def __init__(self,name: str, 
                      model: Callable,
                      derivative: Callable = None):

        self.name = name
        self.model = model
        self.derivative = derivative
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __call__(self,x: np.ndarray) -> np.ndarray:

        return self.model(x)
    
    def deriv(self,x: np.ndarray) -> np.ndarray:
        if self.derivative is not None:
            return self.derivative(x)
        else:
            raise AttributeError('No derivative defined for this feature')  

class SklearnTransform:

    def __init__(self,transforms: List[Callable],
                       scale = True):   
        
        self.transforms = transforms    
        self.scale = scale
        self.xscale = None

    def fit(self,x: np.ndarray, y = None):
        if self.scale:
            self.xscale = MinMaxScaler((1,2))
            self.xscale.fit(x)
    
    def transform(self,x: np.ndarray, y = None) -> np.ndarray:
        return self.feature_transform(self.xscale.transform(x)) if self.xscale else self.feature_transform(x)

    def feature_transform(self,x: np.ndarray, y = None):
        return np.concatenate([tform(x) for tform in self.transforms],axis = 1)
    
    def deriv(self,x: np.ndarray) -> np.ndarray:    
        try:
            return np.concatenate([tform.deriv(x) for tform in self.transforms],axis = 1)
        except AttributeError as ae:
            raise AttributeError(f'Must define derivaties for all features in a transform if derivative is called. No derivative defined for this feature: {str(ae)}')
    
    def fit_transform(self,x: np.ndarray,y = None) -> np.ndarray:
        self.fit(x)
        return self.transform(x)  
    
    def __len__(self):
        return len(self.transforms)
    
    
class OneDimensionalBasisExpansion:
    """ 
    expand the basis of the regressor variables to include
    the listed space of functions below
    """
    def __init__(self, features: List[TransformedFeature] = None, n: int = 7):

        self.features = features

    def __len__(self):
        return len(self.features)
    
    def transform(self,x: np.ndarray) -> np.ndarray:
        return np.concatenate([tform(x) for tform in self.features],axis = 1)
    
    def get_selected_features(self,selected: np.ndarray) -> np.ndarray:
        return [feature for i, feature in enumerate(self.features) if selected[i]]
    
    def __str__(self) -> str:
        return str([feature.name for feature in self.features])
    
    def make_sklearn_transform(self,scale = True) -> SklearnTransform:
        return SklearnTransform([feature for feature in self.features],scale = scale)

def feature_selection(x: np.ndarray,
                      y: np.ndarray,
                      features: List[TransformedFeature],
                      scale = True):
    """ 
    perform feature selection using Lasso/LARS regression
    and both AIC and cross validation critiera. 

    We'd like the most parsimonious model, so 
    use the minimum number of features selected by either

    notice that we specify the intercept by default. 
    """

    
    lasso = [LassoLarsIC('aic'), LassoLarsCV(cv = 5)]
    alpha,num_features = [], []

    
    features = features[:min(len(features),x.shape[0] - 2)]

    tform = OneDimensionalBasisExpansion(features)
    x_ = MinMaxScaler((1,2)).fit_transform(x) if scale else x.copy() # add 1 to avoid log(0)

    X = tform.transform(x_)
    for model_fs in lasso:
        alpha.append(model_fs.fit(X,y).alpha_)
        num_features.append(np.sum(model_fs.coef_ != 0))
    
    idx = 0 if num_features[0] < num_features[1] else 1
    if idx == 0:
        msg = 'selected features using BIC'
    else:
        msg = 'selected features using CV'  

    alpha,model  = alpha[idx],lasso[idx]
    selected = tform.get_selected_features(model.coef_ != 0)
    return selected,msg
        
def setup_axis_default(ax: plt.Axes):
    
    """
    convinience function to set up the axis
    """
    ax.tick_params('both',labelsize = 11,which = 'both',direction = 'in')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    return ax

def identity(x: np.ndarray) -> np.ndarray:
    return x

def inverted(x: np.ndarray) -> np.ndarray:
    return 1/x

def xlogx(x: np.ndarray) -> np.ndarray:
    return x*np.log(x)

def power(power: float,x: np.ndarray) -> np.ndarray:
    return np.power(x,power)

def function_product(func1: Callable,
                     func2: Callable,
                     x: np.ndarray) -> np.ndarray:
    return func1(x)*func2(x)

def constant(const: float,x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)*const

def power_derivative(p: float) -> np.ndarray:
    return partial(function_product,partial(power,p-1),partial(constant,p))

def get_k_most_commmon_feature_transform(data:List[Tuple[np.ndarray,np.ndarray]],
                                         k:int,
                                         input_features = None,
                                         scale = True) -> List[str]:

    """
    get the K most common features across data sets, using the selected features 
    from feature selection routine.
    """
    weights = dict()
    table = dict()


    input_features =  [TransformedFeature('x',identity),
                TransformedFeature('log x',np.log),
                TransformedFeature('x^0.5',np.sqrt),
                TransformedFeature('1/x',inverted),    
                TransformedFeature('x log x',xlogx),
                TransformedFeature('x^2',partial(power,2)),
                TransformedFeature('x^2 log x',partial(function_product,partial(power,2),np.log))] if input_features is None else input_features
    
    data_len = 0
    for (x,y) in data:
        features,msg = feature_selection(x,y, 
                                         features= [f for f in input_features],
                                         scale = scale)
        
        data_len += x.shape[0] 
        
        for feature in features:
            key = hash(feature)
            if key not in weights:
                table[key] = feature
                weights[key]  = x.shape[0]
            else:
                weights[key] += x.shape[0]

    for key in weights:
        weights[key]/=data_len

    weights = {key: weights[key] for key in sorted(weights,key = lambda x: weights[x],reverse = True)}
    features = []
    for i,key in enumerate(weights):
        if i == k:
            break
        features.append(table[key])
    
    transform = OneDimensionalBasisExpansion(
        [TransformedFeature('1',partial(constant,1.),derivative= partial(constant,0.))] + features)

    return transform

class SklearnLMDeriv:

    def __init__(self,pipe: Pipeline) -> None:
        self.pipe = pipe

    def __call__(self,Xnew: np.ndarray) -> np.ndarray:
        tform = self.pipe.named_steps['transform'].deriv(Xnew)
        return tform @ self.pipe.named_steps['model'].coef_


class StichPolynomialSplineModel:

    def __init__(self,pipe: Pipeline) -> None:
        
        self.model = pipe
        self.model_deriv = SklearnLMDeriv(pipe)
        self.poly = None
        self.break_points = None

    def fit_spline(self, break_points: np.ndarray):
        
        self.break_points = break_points    
        A= np.zeros((4,4))
        A[0:2,:] = np.concatenate([break_points**r for r in range(4)],axis = 1)
        A[2:,:] = np.concatenate([np.zeros_like(break_points)] + [r*break_points**(r-1) for r in range(1,4)],axis = 1)
        b = np.zeros(4)
        b[2] = self.model_deriv(break_points[0:1,:])
        b[3] = self.hl_model_deriv(break_points[1:,:])
        b[0] = self.model.predict(break_points[0:1,:])
        b[1] = self.hl_model(break_points[1:,:])
        self.poly = Polynomial(np.linalg.solve(A,b).squeeze())
    
    def fit(self, x: np.ndarray,y: np.ndarray, break_points: np.ndarray) -> None:
        self.model.fit(x,y)
        self.model_deriv = SklearnLMDeriv(self.model)
        self.fit_spline(break_points)

    def predict(self,x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        
        mask = x[:,0] <= self.break_points[0]
        if mask.any() and x.shape[0] > 0:
            y[mask,...] = self.model.predict(x[mask,...])[:,np.newaxis]
        
        mask = np.all([x > self.break_points[0],x < self.break_points[1]],axis = 0).squeeze()
        if mask.any() and x.shape[0] > 0:
            y[mask,...] = self.poly(x[mask,...])
        
        mask = np.squeeze(x >= self.break_points[1])
        if mask.any() and x.shape[0] > 0:
            y[mask,...] = self.hl_model(x[mask,...])
        
        return y

    @staticmethod
    def hl_model(x: np.ndarray) -> np.ndarray:
        poly_hl = Polynomial([137.976060119897,-0.03859083192697,1.19769362e-5,-1.484e-9])
        return poly_hl(x) + 3.866e6/(x + 273.15)**2

    @staticmethod
    def hl_model_deriv(x: np.ndarray) -> np.ndarray:
        poly_hl = Polynomial([137.976060119897,-0.03859083192697,1.19769362e-5,-1.484e-9])
        return poly_hl.deriv()(x) - 2*3.866e6/(x + 273.15)**3
    
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

def get_p_from_generic_model(model: LinearRegression | Pipeline | Callable) -> int:
    """
    try and get the number of parameters from a generic model
    """

    if isinstance(model,Pipeline):
        try:
            return model.named_steps['regression'].coef_.shape[0]
        except KeyError:
            try:
                return model.named_steps['model'].coef_.shape[0]
            except KeyError:
                return len(model.named_steps['transform'])
    elif isinstance(model,LinearRegression):
        return model.coef_.shape[0] 
    else:
        raise ValueError('Model type not recognized')


class OneDPiecewiseFunction:

    def __init__(self,models: List[Any],
                              intervals: List[Tuple[int]]) -> None:
        self.models = models
        self.intervals = intervals

    def predict(self,x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for i,model in enumerate(self.models):
            mask = np.all([x >= self.intervals[i][0],x <= self.intervals[i][1]],axis = 0)
            y[mask,...] = model.predict(x[mask,...])
        return y
    
    def __call__(self,x: np.ndarray) -> np.ndarray:
        return self.predict(x)
    