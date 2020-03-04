# -----------------------------------------------------------
# CSCI-720-Big-Data-Analytics
# Topic: Decistion Tree
# Assignment: HW05
#
# Author: Zizhun Guo / maybe someone joins
# Email: zg2808@cs.rit.edu
# 
# RIT, Rochester, NY
# 
# Zizhun GUO @ All rights researved
# -----------------------------------------------------------

import numpy as np # fundamental scientific computing package
import pandas as pd # data analysis and manipulation package
import matplotlib.pyplot as plt # plotting package
from datetime import datetime # python std data package
import sys # for using command line argument


class node:
    def __init__(self, value = 0, left_node = None, right_node = None):
        self.value = value
        self.left_node = left_node 
        self.right_node = right_node


# def create_tree(val):
#     if val == 3:
#         return node(3)
#     current_node = node(val)
#     current_node.left_node = create_tree(val + 1)
#     current_node.right_node = create_tree(val + 1)
#     return current_node

# def traverse(node):
#     print(" "* value + str(node.value))
#     if node.left_node != None: traverse(node.left_node)
#     if node.right_node != None: traverse(node.right_node)


# def main():
    # root = create_tree(0)
    # traverse(root)

def data_preprocessing(dataframe_data):
    """
    Paras:
        @dataframe_data: a dataframe of data read from CSV file
    Return:
        dataframe_data: a dataframe of data
    """
    columns_size = np.size(dataframe_data.columns)
    columns = dataframe_data.columns[0 : columns_size - 1]
    # print(columns)
    for column in columns:
        # print(column)
        data = np.round(dataframe_data[column] / 1) * 1
        dataframe_data[column] = data
    
    data_ages= np.round(dataframe_data['Age'] / 2) * 2 # 
    dataframe_data['Age'] = data_ages
    data_height= np.round(dataframe_data['Ht'] / 5) * 5 # 
    dataframe_data['Ht'] = data_height
    return dataframe_data

def entropy_shannon(dataframe_data):
    

def main():
    df_snowfolks_data_raw = pd.read_csv('Abominable_Data_HW05_v720.csv')
    df_snowfolks_data = data_preprocessing(df_snowfolks_data_raw)


    # print(df_snowfolks_data_raw.columns)
    # print(df_snowfolks_data_raw.columns[1:3]) # select columns titles
    # print(df_snowfolks_data_raw['Reach']- 100) # split dataframe based on threshold
    
    # data_ages= np.round(df_snowfolks_data_raw['Age'] / 2) * 2 # 
    # df_snowfolks_data_raw['Age'] = data_ages

    # data_height= np.round(df_snowfolks_data_raw['Ht'] / 5) * 5 # 
    # df_snowfolks_data_raw['Ht'] = data_height



if __name__ == "__main__":
    main()