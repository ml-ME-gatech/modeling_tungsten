import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from typing import List, Tuple, Callable    
from sklearn.preprocessing import MinMaxScaler  
from sklearn.linear_model import LassoLarsIC, LassoLarsCV
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

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
    
class OneDimensionalBasisExpansion:
    """ 
    expand the basis of the regressor variables to include
    the listed space of functions below
    """
    def __init__(self, features: List[TransformedFeature] = None, n: int = 7):

        self.features = features

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
    ax.set_xticks(np.arange(0,1500,300))
    ax.set_yticks(np.arange(200,1900,300))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    return ax

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


    input_features =  [TransformedFeature('x',lambda x: x),
                TransformedFeature('log x',lambda x: np.log(x)),
                TransformedFeature('x^0.5',lambda x: x**0.5),
                TransformedFeature('1/x',lambda x: 1/x),
                TransformedFeature('x log x',lambda x: x*np.log(x)),
                TransformedFeature('x^2',lambda x: x**2),
                TransformedFeature('x^2 log x',lambda x: x**2*np.log(x))] if input_features is None else input_features
    
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
        [TransformedFeature('1',lambda x: np.ones_like(x),derivative= lambda x: np.zeros_like(x))] + features)

    return transform