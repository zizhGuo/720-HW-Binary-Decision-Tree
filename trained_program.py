import pandas as pd
import numpy as np
import sys

def main():
    test_data_path=sys.argv[1]
    test_data=pd.read_csv(test_data_path)
    category_pre=[]
    threshold=0

    for row in test_data:
        if(row['']> threshold):
            category_pre.append('Assam')
        elif(row['']> threshold):
            category_pre.append('Assam')
        elif(row['']> threshold):
            category_pre.append('Assam')
        elif(row['']> threshold):
            category_pre.append('Assam')
        elif(row['']> threshold):
            category_pre.append('Assam')
        elif(row['']> threshold):
            category_pre.append('Assam')
        elif(row['']> threshold):
            category_pre.append('Assam')
        elif(row['']> threshold):
            category_pre.append('Assam')
        else:
            category_pre.append('Bh')
    
    df=pd.DataFrame(category_pre)
    df.to_csv('output.csv')


if __name__ == "__main__":
    main()
        