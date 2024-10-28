import pandas as pd
import numpy as np

def main():

    df = pd.read_csv('tschudia_data_raw.csv',header = 1)
    with open('tschudia_data_raw.csv', 'r') as f:
        headers = [s.strip() for s in f.readline().split(',') if s.strip() and 'error' not in s]

    for i in range(len(headers)):
        data = df.iloc[:,i*4:i*4+4]
        data.columns = ['temperature','hardness','none','hardness error']
        data = data[['temperature','hardness','hardness error']]
        data.loc[:,'hardness error'] = np.abs(data['hardness error'].to_numpy() - data['hardness'].to_numpy())
        data.dropna(inplace= True)
        data.to_csv('tschudia_data_{}.csv'.format(headers[i]),index=False)

if __name__ == '__main__':
    main()