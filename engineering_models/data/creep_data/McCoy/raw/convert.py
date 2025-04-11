import pandas as pd
import numpy as np

def main():

    df = pd.read_csv("PureWSheetCleaned.csv",header = 0)
    df.loc[:,'Stress (psi)'] = np.array([int(x.replace(',','')) for x in df['Stress (psi)']])
    df.loc[:,'Rupture Time (hr)'] = np.array([float(x.replace(',','.')) for x in df['Rupture Time (hr)']])
    df.to_csv('../PureWSheet.csv',index = False)

if __name__ == "__main__":
    main()
