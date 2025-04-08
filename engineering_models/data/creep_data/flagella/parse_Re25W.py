import numpy as np
import pandas as pd

def main():

    with open('raw/Re25W_raw.csv','rb') as f:
        columns = f.readline().strip().split(',')
        data  = []
        for line in f.readlines():
            dat = line.strip().split("")[1]
            print(dat)


    
    # Read the data into a pandas DataFrame


if __name__ == "__main__":
    main()