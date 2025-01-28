import pandas as pd
import numpy as np
from pathlib import WindowsPath
from scipy.spatial.distance import cdist

ALLOY_MAP = {'k_doped_w_3percent_re_plate_h':r'K-W3%Re Plate (H)',
              'k_doped_w_3percent_re_plate_l':r'K-W3%Re Plate (L)',
              'k_doped_w_plate_h':r'K-W Plate (H)',
              'pure_w_plate_h':r'W Plate (H)',
              'w_3percent_re_1percent_la2o3_plate_l':r'W3%Re-1%La2O3 Plate (L)',
              'w_3percent_re_plate_h':r'W3%Re Plate (H)',
              'w_3percent_re_plate_l':r'W3%Re Plate (L)'}

TEMP_BINS = np.array([20,100,200,300,400,500,600,700,800,900,1000,1200,1300]).astype(float)
TEMP_THRESH = 40.0

def bin_temperatures(temps: np.ndarray) -> np.ndarray:

    dist = cdist(temps[:,None],TEMP_BINS[:,None])
    index = np.argmin(dist,axis = 1)
    return TEMP_BINS[index]

def fix_temperatures(temp: np.ndarray):

    if temp.max() > 1700:
        temp -= temp.min()
    
    return temp + 20

def get_column_name(label: str,
                    fname: str):

    try:
        return label + ' ' + ALLOY_MAP[fname.split('.')[0].lower()]
    except KeyError:
        raise KeyError('Cannot map file: {}'.format(fname))
    
def convert_folder_to_frame(folder: WindowsPath,
                            label: str) -> pd.DataFrame:

    folder = WindowsPath(folder)
    df_list = []
    for file in folder.iterdir():
        df = pd.read_csv(file,header = None,index_col = 0).squeeze()
        df.index = fix_temperatures(df.index.to_numpy())
        df.index = bin_temperatures(df.index.to_numpy())
        df.name = get_column_name(label ,file.name)
        df_list.append(df.copy())
    
    return pd.concat(df_list,axis = 1)

def main():

    df_list = []
    folders = {'uts':'UTS [MPa]',
               'uniform_elongation':'UE [%]',
               'total_elongation':'TE [%]'}
    
    for folder,label in folders.items():
        df = convert_folder_to_frame(folder,label)
        df_list.append(df.copy())
    
    df = pd.concat(df_list,axis = 1)
    df.index.name = 'T [C]'
    df.to_csv('nogami_data.csv')



if __name__ == '__main__':
    main()
