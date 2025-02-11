import pandas as pd
import numpy as np
import math

def assign_bins(values: np.ndarray,
                bin_edges: np.ndarray,
                assume_sorted: bool = False) -> np.ndarray:
    
    if not assume_sorted:
        values = np.sort(values)
    
    k,j = 1,0
    new_values = np.zeros(values.shape[0])
    while k < bin_edges.shape[0] and j < values.shape[0]:
        while k < bin_edges.shape[0] and values[j] > bin_edges[k]:
            k+=1
        new_values[j] = (bin_edges[k] + bin_edges[k-1])/ 2
        j+=1
    
    return new_values

def bin_data(width: float,df: pd.DataFrame) -> pd.DataFrame:

    temperatures = df.index
    nbins = int(math.ceil((temperatures.max() - temperatures.min()) / width) )

    edges = np.histogram_bin_edges(temperatures,bins=nbins)
    temps = assign_bins(temperatures,edges)
    df.index = pd.Series(temps,name = 'temperature')
    return df

def main():
    df = pd.read_csv('shah_data/data.csv',index_col = 0,header = 0)
    df.set_index('temperature',inplace=True)
    df = bin_data(50,df)
    df.index = np.round(df.index)
    df['temperature'] = df.index
    df.index = range(df.shape[0])
    df.to_csv('shah_data/binned_temperature_data.csv')

if __name__ == '__main__':
    main()
