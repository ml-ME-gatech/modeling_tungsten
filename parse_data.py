import pandas as pd
from typing import List,Dict, Any
import numpy as np
from pathlib import WindowsPath
import warnings
from scipy.optimize import linear_sum_assignment

class DataReader:

    def __init__(self, file: WindowsPath):
        self.file = file

    @staticmethod
    def _read(file):
        df = pd.read_csv(file,index_col = 0,header = None,dtype = np.float64).squeeze()
        df.index.name = 'time'
        df.name ='temperature' 
        return df
    
    def read(self):
        return self._read(self.file)
    
    def read_uncertainty(self):
        file = self.file.parent.joinpath((self.file.stem + '_u' + self.file.suffix))
        df = self._read(self.file).to_frame()
        u_df =  self._read(file)
        if df.shape[0] != u_df.shape[0]:
            raise ValueError('Data and uncertainty files have different lengths')

        index = self.pair_uncertainty(df.index.to_numpy(),u_df.index.to_numpy())
        df['std'] = np.abs(df.to_numpy().squeeze() - u_df.iloc[index].to_numpy())
        return df

    def pair_uncertainty(self,df: np.ndarray,u_df: np.ndarray):
        _,row = linear_sum_assignment(
            np.abs(df[:,np.newaxis] - u_df[np.newaxis,:])
            )
        return row
    

class DataCombiner:

    def __init__(self, file_factors: Dict,
                        reader: Any = DataReader):       
        self.file_factors = file_factors
        self.reader = reader

    def combine(self,
                columns: List[str] = ['time','X','std','temperature']):
        

        X = []
        _u = True
        for factor in self.file_factors:
            reader = self.reader(self.file_factors[factor])
            try:
                df = reader.read_uncertainty()
                X.append(np.concatenate([df.index.to_numpy()[:,np.newaxis],
                                        df.to_numpy(),
                                        factor*np.ones([len(df),1])],axis = 1))
            except FileNotFoundError:
                _u = False
                warnings.warn('Uncertainty file not found, using data file')
                df = reader.read()
                X.append(np.concatenate([df.index.to_numpy()[:,np.newaxis],
                                        df.to_numpy()[:,np.newaxis],
                                        factor*np.ones([len(df),1])],axis = 1))

        if not _u and 'std' in columns:
            columns.remove('std')
        
        df = pd.DataFrame(np.concatenate(X,axis = 0),
                          columns = columns if _u else columns)
        return df
    
class DataParser:

    def __init__(self,folder: str):

        self.folder = WindowsPath(folder)
        csv_files  = [file for file in self.folder.iterdir() if file.suffix == '.csv' and 'hv' in file.stem]
        self.files = {int(file.name[3:7]):file for file in csv_files if 'u' not in file.stem}
        self.u_files = {int(file.stem[3:7]):file for file in csv_files if 'u' in file.stem}

    def parse(self):
        combiner = DataCombiner(self.files)
        df = combiner.combine()
        return df



def main():
    folder = 'rf_data/yu_data'
    parser = DataParser(folder)
    df = parser.parse()
    print(df.columns)
    df.to_csv('rf_data/yu_data/hardness_data.csv')


if __name__ == '__main__':
    main()
