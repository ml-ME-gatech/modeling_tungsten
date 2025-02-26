from numpy.polynomial import Polynomial
import pandas as pd
import pickle

def fit_external_ym():
    data = pd.read_csv('data/youngs_modulus',index_col = 0,header = 0)
    
    poly = Polynomial.fit(data.index,data['E [GPa]'],deg = 2)
    with open('.model/youngs_modulus','wb') as f:
        pickle.dump(poly,f)

def main():
    fit_external_ym()

if __name__ == '__main__':
    main()
