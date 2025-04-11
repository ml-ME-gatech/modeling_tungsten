import numpy as np
import pandas as pd
from functools import wraps
from typing import Callable

def maybe_nan(func: Callable) -> Callable:

    @wraps(func)
    def wrapped(s: str):
        if s == '-':
            return np.nan
        return  func(s)
    
    return wrapped

def main():

    transforms = [lambda x: x,
                  lambda x: int(x.replace(',','')),
                  lambda x: int(x.replace(',','')),
                  lambda x: np.nan,
                  lambda x: np.nan, 
                  lambda x: float(x),
                  lambda x: int(x.replace(',',''))]
    
    transforms = [maybe_nan(t) for t in transforms] 

    with open('Re25W_raw.csv','r') as f:
        columns = f.readline().strip().split(',')
        data  = []
        for line in f.readlines():
            line = ''.join(line.strip().split(',')[:-6])
            if line[0] == '"':
                line = line[1:]
            if line[-1] == '"':
                line = line[:-1]
            
            dat = [t(s) for t,s in zip(transforms,line.split(" "))]
            data.append(dat)

    
    data = pd.DataFrame(data, columns=columns)
    data = data[['Material Source','Test Temperature (°C)', 'Stress (psi)',
                 'Time to Rupture (hr)',
                  'Total Elongation (%)']]
    data.dropna(inplace= True)
    data.to_csv('../Re25WSheet.csv')   
            


    
    # Read the data into a pandas DataFrame


if __name__ == "__main__":
    main()