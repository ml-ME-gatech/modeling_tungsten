import pandas as pd
from pathlib import WindowsPath
from typing import List

def read_data(file: str)-> pd.DataFrame:
    df = pd.read_csv(file,header = None,index_col= None)
    mattype,temp = file.split('_')
    temp = float(temp[1:].split('.')[0][:-1])
    df.columns = ['time to rupture [h]','Creep Stress [MPa]']
    df['Temperature [C]'] = temp
    df['Material Type'] = mattype
    return df

def read_multiple(files: List[str]) -> pd.DataFrame:

    data = []
    for file in files:
        data.append(read_data(file))
    
    return pd.concat(data,ignore_index= True)

def main():

    files = [file.name for file in WindowsPath('').iterdir() if 'WL10' in file.name]
    print(files)
    df = read_multiple(files)
    df.to_csv('WL10.csv')

if __name__ == '__main__': 
    main()

