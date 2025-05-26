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
from matplotlib.path import Path as MPath
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from statsmodels.regression.linear_model import OLS,OLSResults
from dataclasses import dataclass
import sys
from scipy.interpolate import interp1d
from scipy.optimize import bisect
import math
import sys
from statsmodels.tools.tools import add_constant
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
                'CREEP_DATA': 'data/creep_data',
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
            return MPath(self.transform(path.vertices), path.codes)

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
                              path=MPath.unit_regular_polygon(num_vars))
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
        



def markdown_table_from_df(df: pd.DataFrame,
                            title: str,
                            caption: str,
                            replace_nan: str = 'N/A') -> str:
    
    title_caption = '**' + title + '**:' + caption + '\n'
    table_str = df.to_markdown()
    if replace_nan is not None:
        table_str = table_str.replace('nan',replace_nan)

    return title_caption + table_str + '\n'

def bf_minimize_scalar(func: callable,bounds: Tuple[float,float],interval: int = 10,**kwargs) -> float:
    
    """
    Brute force minimize scalar function by dividing the interval into smaller intervals and using scipy's minimize_scalar
    """
    intervals = np.linspace(bounds[0],bounds[1],interval + 1)
    f = np.ones(interval)*np.inf
    x = np.zeros(interval)
    for i in range(interval):
        try:
            res = minimize_scalar(func,bounds = (intervals[i],intervals[i+1]),**kwargs)
            x[i]  = res.x
            f[i] = res.fun
        except ValueError:
            pass
    
    return x[np.argmin(f)]

def bf_bisect(func: callable,bounds: Tuple[float,float],interval: int = 50,**kwargs) -> float:
    """"
    Brute force bisection method by dividing the interval into smaller intervals and using scipy's bisect
    """
    intervals = np.linspace(bounds[0],bounds[1],interval + 1)
    f = np.ones(interval)*np.inf
    x = np.zeros(interval)
    for i in range(interval):
        try:
            xx = bisect(func,intervals[i],intervals[i+1],**kwargs)
            x[i]  = xx
            f[i] = func(xx)
        except ValueError:
            pass
    
    return x[np.argmin(f)]

def estimate_eu(df: pd.DataFrame,**interp_kwargs) -> Tuple[float,float]:
    """
    Estimate the uniform elongation and corresponding ultimate tensile stress of the material
    """
    func = interp1d(df.index,df.to_numpy().squeeze(),**interp_kwargs)
    def _opt_func(x):
        return -func(x)

    eps_u = bf_minimize_scalar(_opt_func,bounds = (df.index.min(),df.index.max()),method = 'bounded')
    Su = func(eps_u)
    return eps_u,Su

def estimate_tr(df: pd.DataFrame,dec_frac = 0.25,**interp_kwargs) -> Tuple[float,float]:
    """
    Estimate the total elongation and corresponding stress at failure of the material
    """
    func = interp1d(df.index,df.to_numpy().squeeze(),**interp_kwargs)
    def _opt_func(x):
        return -func(x)
    
    eu = bf_minimize_scalar(_opt_func,bounds = (df.index.min(),df.index.max()),method = 'bounded')
    Su = func(eu)
    
    if df.iloc[-1].squeeze() > Su*(1-dec_frac):
        eps_tr = df.index[-1]

    else:
        eps_tr = bf_bisect(lambda x: func(x) - Su*(1.-dec_frac),(eu,df.index[-1]))

    return eps_tr,func(eps_tr)

def moving_average(data: np.ndarray,window: int) -> np.ndarray:
    return np.convolve(data,np.ones(window)/window,mode = 'same')

def estimate_ym(data: pd.DataFrame,smooth: int = 6,E_frac = 0.5,
                    return_end_point: bool = False,
                         **interp_kwargs) -> Tuple[float,float]:

    """
    estimate the young's modulus of the material - this is not a very general function,
    would require adapation/modification for different data.
    """
    smooth = int(math.ceil(smooth/2)*2)
    signal = moving_average(data.to_numpy().squeeze(),smooth)[::smooth//2]
    xx = data.index.to_numpy()[1:][::smooth//2]
    signal = signal[1:] if signal.shape[0] > xx.shape[0] else signal
    deriv = np.gradient(signal,xx)
    
    func = interp1d(xx,deriv,**interp_kwargs)
    E = deriv[smooth:smooth*2].mean()

    def _opt_func(x):
        return func(x) - E_frac*E

    end_point = smooth*4
    while end_point < signal.shape[0] - 1 and deriv[end_point] > 0.0:
        end_point += 1
     
    
    end_point*= smooth//2
    yield_point = bf_bisect(_opt_func,bounds = (data.index[smooth*4],data.index[end_point]))
    
    _data = data.loc[data.index < yield_point*0.9]    
    _data = _data.iloc[smooth*2::]

    E = np.mean(_data.to_numpy()/_data.index.to_numpy())
    return E if not return_end_point else (E,end_point)

def estimate_yield_point(data: pd.DataFrame,smooth: int = 6,**interp_kwargs) -> Tuple[float,float]:
    """"
    Estimate the yield point of the material.
    """
    E,ei = estimate_ym(data,smooth = smooth,return_end_point = True,**interp_kwargs)
    sigma_02_offset = lambda eps: E*(eps -0.2*1e-2)
    signal = moving_average(data.to_numpy().squeeze(),smooth)[::smooth//2]
    xx = data.index.to_numpy()[1:][::smooth//2]
    signal = signal[1:] if signal.shape[0] > xx.shape[0] else signal
    func = interp1d(xx,signal,**interp_kwargs)

    def _intersection_func(eps: float ) -> float:
        return func(eps) - sigma_02_offset(eps)
    
    eps_yield = bf_bisect(_intersection_func,bounds = (0.0,data.index[ei]))

    return eps_yield,func(eps_yield)


def estimate_power_law(data_: pd.Series,
                       smooth: int = 6,
                       tol: float = 1e-3,
                       E_frac: float = 0.5,
                       **interp_kwargs) -> OLSResults:   
    
    eu,Su = estimate_eu(data_,**interp_kwargs)
    E  = estimate_ym(data_,smooth = smooth,E_frac= E_frac,**interp_kwargs)
    print(E)
    data = data_.copy()
    sigma_true = data.values * (1.0 + data.index.to_numpy())
    eps_true = np.log1p(data.index)

    eps_pl   = eps_true - sigma_true/E                             # ε_p = ε_true − σ_true/E
    print(eps_pl.max())
    mask     = (eps_pl > tol) & (data.index.to_numpy() < eu - tol)

    x = add_constant(np.log(eps_pl[mask]))   
    y = np.log(sigma_true[mask])
    ols_results = OLS(y,x).fit()
    return ols_results



    