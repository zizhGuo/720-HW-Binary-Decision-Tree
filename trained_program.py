import pandas as pd
import numpy as np
import sys

def main():
    #test_data_path=sys.argv[1]
    test_data_path=('D:/git/720-HW-Binary-Decision-Tree/Abominable_Data_HW05_v720.csv')
    test_data=pd.read_csv(test_data_path)
    category_pre=[]

    for row in test_data:
        if(row['BangLn']>= 6.0):
            category_pre.append('Assam')
        elif(row['Ht']<= 140):
            category_pre.append('Bhuttan')
        elif(row['EarLobes']<=0):
            category_pre.append('Bhuttan')
        elif(row['TailLn']<= 12.0):
            category_pre.append('Assam')
        elif(row['BangLn']>= 5.0):
            category_pre.append('Assam')
        else:
            category_pre.append('Bhuttan')
    
    df=pd.DataFrame(category_pre)
    df.to_csv('D:/git/output.csv')


if __name__ == "__main__":
    main()
        