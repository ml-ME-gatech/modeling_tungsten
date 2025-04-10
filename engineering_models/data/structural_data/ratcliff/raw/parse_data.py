import pandas as pd 
import numpy as np
from scipy.interpolate import interp1d
import warnings 
from scipy.signal import savgol_filter

FtoC = lambda x: (x - 32) * 5.0/9.0
KSI_TO_MPA = 6.89476

def homogenize_bounds(lb: pd.Series,
                      ub: pd.Series) -> pd.DataFrame:

    if len(lb) > len(ub):
        df = homogenize_bounds(ub,lb)
        df.columns = ['ub', 'lb']
        return df
    

    with warnings.catch_warnings(action = 'ignore'):
        lb.sort_index(inplace=True) 
        ub.sort_index(inplace=True)
    
    Tmin = min(lb.index[0], ub.index[0])
    Tmax = max(lb.index[-1], ub.index[-1])
    lb = lb.loc[Tmin:Tmax]
    ub = ub.loc[Tmin:Tmax]

    interp = interp1d(lb.index, lb.values, kind='linear', fill_value='extrapolate',bounds_error= False)
    lb = pd.Series(interp(ub.index), index=ub.index)
    df = pd.concat([lb.to_frame(), ub.to_frame()], axis=1)
    for col in df.columns:
        df.loc[:,col] = savgol_filter(df.loc[:,col],20,2)

    df.columns = ['lb', 'ub']
    df.index.name = 'Temperature (C)'
    return df



def main():
    # Load the data
    lb = pd.read_csv('uts_wrought_lb.csv', index_col=0, header=None).squeeze()
    ub = pd.read_csv('uts_wrought_ub.csv', index_col=0, header=None).squeeze()

    # Homogenize the bounds
    bounds = homogenize_bounds(lb, ub)
    bounds*= KSI_TO_MPA
    bounds.index = np.array([FtoC(float(i)) for i in bounds.index]) 

    # Save the result to a new CSV file
    bounds.to_csv('../uts_bounds.csv', index=True, header=True)

if __name__ == "__main__":
    main()



