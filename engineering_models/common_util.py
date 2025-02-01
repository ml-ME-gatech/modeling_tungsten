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
import copy
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from statsmodels.regression.linear_model import OLS,OLSResults
from dataclasses import dataclass
import sys

import sys
if 'win' in sys.platform.lower():
    from pathlib import WindowsPath as Path
else:
    from pathlib import PosixPath as Path

#globals
_PARENTDIR = Path(str(__file__)).parent.resolve()
_MAKE_PROJECT_PATH_DIR = False

"""
Common utilities used among the jupyter notebooks in this folder
"""

class mk_pdir:

    """
    context manager to create a directory if it doesn't
    exist, used in combination with ProjectPaths
    """
    
    def __init__(self):
        pass

    def __enter__(self,*args,**kwargs):
        global _MAKE_PROJECT_PATH_DIR
        _MAKE_PROJECT_PATH_DIR = True
    
    def __exit__(self,*args,**kwargs):
        global _MAKE_PROJECT_PATH_DIR
        _MAKE_PROJECT_PATH_DIR = False

class ProjectPaths:

    _defaults = {'MODEL': '.model',
                 'SCRATCH': '.scratch',
                'STRUCTURAL_DATA': 'data/structural_data',
                'CONDUCTIVITY_DATA': 'data/conductivity_data',
                'MODELING': 'modeling',
                'DATA_EXPLORATION': 'data_exploration',
                'GIT_IMAGES': '.git_images',
                'GIT_TABLES': '.git_tables',
                'IMAGES': 'images',
                 'parent': _PARENTDIR}
    
    def __init__(self,**kwargs):
        for dkey,dvalue in self._defaults.items():
            if dkey not in kwargs:
                kwargs[dkey] = dvalue
        
        self.paths = kwargs
        if not isinstance(self.paths['parent'], Path):
            raise TypeError(f'parent must be of type: {type(Path)} for expected behavior')
    
    def __getattr__(self, __name: str) -> Path:
        try:
            path =  self.paths[__name] if __name == 'parent' else self.paths['parent'].joinpath(self.paths[__name])
            if not path.exists() and _MAKE_PROJECT_PATH_DIR:
                path.mkdir(parents = True)
            return path
        
        except KeyError as ke:
            raise KeyError(f'path {__name} not in predifined paths\n' + str(ke))


def setup_plotting_format():
    
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Times New Roman'],'weight': 'bold'})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = "".join([r"\usepackage{newtxtext,newtxmath}",r"\boldmath"])


class NogamiData:

    """
    convenience class to access Nogami data, and return it as numpy arrays
    """

    def __init__(self, key: str,
                       data_file: str):

        self.key = key
        self.data = pd.read_csv(data_file,index_col = 0)
        self.cols = set(self.data.columns)

    
    def _fancy_key(self,key: str) -> str:
        if key in self.cols:
            return key
        else:
            for col in self.cols:
                if col.endswith(key) and col.startswith(self.key):
                    return col

    def get_df(self,key: str,keep_column: bool = True) -> pd.DataFrame:
        df = self.data[[self._fancy_key(key)]]
        if not keep_column:
            df.columns = [self.key]

        return df
    
    def __getitem__(self,key: str) -> Tuple[np.ndarray,np.ndarray]:
        
        _key = self._fancy_key(key)
        data = self.data[[_key]]    
        data.dropna(inplace = True)
        return data.index.to_numpy()[:,np.newaxis],data[_key].to_numpy()
    
    def keys(self):
        return [column for column in self.data.columns if self.key in column]
    
class NogamiUTSData(NogamiData):

    def __init__(self, project_paths: ProjectPaths = ProjectPaths()):
        
        if isinstance(project_paths,str):
            super().__init__('UTS',project_paths)
        else:
            super().__init__('UTS',project_paths.STRUCTURAL_DATA.joinpath('nogami_data.csv'))

class NogamiUEData(NogamiData):

    def __init__(self,project_paths: ProjectPaths = ProjectPaths()):
        if isinstance(project_paths,str):
            super().__init__('UE',project_paths)
        else:
            super().__init__('UE',project_paths.STRUCTURAL_DATA.joinpath('nogami_data.csv'))

class NogamiTEData(NogamiData):

    def __init__(self,project_paths: ProjectPaths = ProjectPaths()):
        if isinstance(project_paths,str):
            super().__init__('TE',project_paths)
        else:
            super().__init__('TE',project_paths.STRUCTURAL_DATA.joinpath('nogami_data.csv'))

class NogamiConductivityData(NogamiData):

    def __init__(self,project_paths: ProjectPaths = ProjectPaths()):
        if isinstance(project_paths,str):
            super().__init__('','conductivity_data/nogami_data.csv')
        else:
            super().__init__('',project_paths.CONDUCTIVITY_DATA.joinpath('nogami_data.csv'))

class NogamiDataCollection:
    material_to_cond = {'K-W Plate (H)': 'K-doped W (H) Plate',
                        'K-W3%Re Plate (H)': 'K-doped W-3%Re (H) Plate',
                        'K-W3%Re Plate (L)': 'K-doped W-3%Re (H) Plate',
                        'W3%Re Plate (H)': 'W-3%Re (H) Plate',
                        'W3%Re Plate (L)': 'W-3%Re (H) Plate',
                        'W Plate (H)': 'Pure W (H) Plate'}
    
    def __init__(self):
        self.uts = NogamiUTSData()
        self.ue = NogamiUEData()
        self.te = NogamiTEData()
        self.conductivity = NogamiConductivityData()

    def uts_key(self,key: str) -> str:
        return 'UTS [MPa] ' + key
    
    def ue_key(self,key: str) -> str:
        return 'UE [%] ' + key
    
    def conductivity_key(self,key: str) -> str:
        return self.material_to_cond[key]
    
    def te_key(self,key: str) -> str:
        return 'TE [%] ' + key
    
    def __getitem__(self,key: str) -> Tuple[Tuple[np.ndarray]]:
        return self.uts[self.uts_key(key)],self.ue[self.ue_key(key)],self.conductivity[self.conductivity_key(key)],self.te[self.te_key(key)]
    
    def keys(self):
        return [key[10:].strip() for key in self.uts.keys()]

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
                      scale = True,
                      cv = 5):
    """ 
    perform feature selection using Lasso/LARS regression
    and both AIC and cross validation critiera. 

    We'd like the most parsimonious model, so 
    use the minimum number of features selected by either

    notice that we specify the intercept by default. 
    """

    
    lasso = [LassoLarsIC('aic'), LassoLarsCV(cv = cv)]
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
        
def setup_axis_default(ax: plt.Axes,labelsize: float = 11.):
    
    """
    convinience function to set up the axis
    """
    ax.tick_params('both',labelsize = labelsize,which = 'both',direction = 'in')
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
                                         scale = True,
                                         cv = 5) -> List[str]:

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
                                         scale = scale,
                                         cv = cv)
        
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


def make_radar_plot(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes. Hacked from stackoverflow

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels,**kwargs)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

class ParamterizedLinearModel:

    """
    Need to pickle this for later, so just make it a class
    """
    def __init__(self,model: Pipeline,
                       model_spread: Pipeline = None) -> None:
        
        self.x,self.y = None,None
        self.model = copy.deepcopy(model)
        self.model_spread = copy.deepcopy(model if model_spread is None else model_spread)

    def fit_spread(self, x: np.ndarray,y: np.ndarray, 
                        *args,alpha = 0.95,**kwargs) -> None:
        
        ci = hdi(y,alpha = alpha).T
        d = 0.5*(ci[1] - ci[0])
        self.model_spread.fit(x,0.5*(ci[1] - ci[0]),*args,**kwargs)
        d_max = d.max()
        self.gamma_bracket = (-d_max*2,d_max*2)

    def fit_mean(self, x: np.ndarray,y: np.ndarray,*args,**kwargs) -> None:
        self.model.fit(x,y,*args,**kwargs)

    def __call__(self,xnew: np.ndarray,gamma: float) -> np.ndarray:
        return self.model.predict(xnew).squeeze() + gamma*self.model_spread.predict(xnew).squeeze()
    
    def find_gamma(self,material_model: Callable,xlim: Tuple[float,float]) -> float:
        return self._find_closest_model(material_model,self,xlim,self.gamma_bracket)
    
    def mean_value(self,xlim: Tuple[float],gamma: float) -> float:
        def integrate(x: float):
            _x = np.array([x])[:,np.newaxis]
            return self(_x,gamma)

        return 1/(xlim[1] - xlim[0])*quad(integrate,*xlim)[0]

    
    @staticmethod
    def _find_closest_model(material_model: Callable,
                            parametric_model: Callable,
                            xlim: Tuple[float,float],
                            gamma_lim: Tuple[float,float]):
        
        """
        Could have done using analytical integrals, but this is easier to implemnt. 
        """
        
        def model_diff(gamma: float):
            
            def wrapped(x: float):
                _x = np.array([x])[:,np.newaxis]
                y = material_model(_x).squeeze() - parametric_model(_x,gamma)
                return y
            
            return np.abs(quad(wrapped,*xlim)[0])

        res = minimize_scalar(model_diff,bounds = gamma_lim)
        if res.success:
            return res.x
        else:
            raise ValueError("Optimization failed")
        

class LarsonMiller:

    """ 
    Fit the model:
    S_t = sum([beta_i x^i for i in range(deg+1)])
    x = T*(C + log(t))
    """

    def __init__(self):
        self.C,self.deg = None,None

    def lmp(self,t: np.ndarray,T: np.ndarray,C: float) -> np.ndarray:
        return T*(C + np.log(t))

    def _feature_transform(self,t: np.ndarray,T: np.ndarray, C, deg: int)-> np.ndarray:
        x = self.lmp(t,T,C)
        return np.concatenate([x[:,np.newaxis]**i for i in range(deg+1)],axis = 1)


    def refit_model(self,t: np.ndarray,
                         T: np.ndarray,
                         stress: np.ndarray):
        
        self.ols_results= OLS(stress,self._feature_transform(t,T,self.C,self.deg)).fit()
        return self
    
    def coeff(self):
        return self.ols_results.params

    def make_improved_model(self,v: float):

        new_model = copy.deepcopy(self)
        new_model.C *= (1+ v)
        new_model.ols_results.params[0] *= (1+v) 
        return new_model
    
    def fit(self,t: np.ndarray,
                 T: np.ndarray,
                 stress: np.ndarray, 
                 deg = 1,
                 C_bounds = (1,1e3),
                 **opt_kwargs):

        
        self.deg = deg

        def _objective_function(C: np.ndarray):
            X = self._feature_transform(t,T,C,deg)
            result  = OLS(stress,X).fit()
            stress_hat = result.predict(X)
            return np.linalg.norm(stress - stress_hat)

        opt_result = minimize_scalar(_objective_function,bracket = C_bounds,
                                bounds = C_bounds,**opt_kwargs)

        if opt_result.success:
            self.C = opt_result.x
            X = self._feature_transform(t,T,self.C,deg)
            self.ols_results = OLS(stress,X).fit()
        else:
            raise RuntimeError('failed to fit creep model')

        return self

    def predict(self,t: np.ndarray,T: np.ndarray):      
        X = self._feature_transform(t,T,self.C,self.deg)
        return self.ols_results.predict(X)


def markdown_table_from_df(df: pd.DataFrame,
                            title: str,
                            caption: str,
                            replace_nan: str = 'N/A') -> str:
    
    title_caption = '**' + title + '**:' + caption + '\n'
    table_str = df.to_markdown()
    if replace_nan is not None:
        table_str = table_str.replace('nan',replace_nan)

    return title_caption + table_str + '\n'