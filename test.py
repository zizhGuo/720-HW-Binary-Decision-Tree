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
import math


class dt_node:
    def __init__(self, attribute = None, threshold = None, ent_average = None, left_node = None, right_node = None, dataframe = None):
        self.attribute = attribute
        self.threshold = threshold
        self.ent_average = ent_average
        # self.split_rate = split_rate
        self.left_node = left_node 
        self.right_node = right_node
        self.dataframe = dataframe


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

def data_preprocessing(dataframe):
    """
    Paras:
        @dataframe: a dataframe of data read from CSV file
    Return:
        dataframe: a dataframe of data
    """
    columns_size = np.size(dataframe.columns)
    columns = dataframe.columns[0 : columns_size - 1]
    # print(columns)
    for column in columns:
        # print(column)
        data = np.round(dataframe[column] / 1) * 1
        dataframe[column] = data
    
    data_ages= np.round(dataframe['Age'] / 2) * 2 # 
    dataframe['Age'] = data_ages
    data_height= np.round(dataframe['Ht'] / 5) * 5 # 
    dataframe['Ht'] = data_height
    return dataframe

def entropy_average(dataframe, attribute, threshold):
    """
    Paras:
        @dataframe: a dataframe
        @attribute: a string of attribute name
        @threshold: a float threshold
    Return:
        dataframe: a dataframe of data
    """
    df_left = dataframe[dataframe[attribute] <= threshold]
    df_right = dataframe[dataframe[attribute] > threshold]
    split_rate = np.abs((np.size(df_left[attribute]) \
                        - np.size(df_right[attribute]))) \
                        / np.size(dataframe[attribute])
    df_left_Assam = df_left[df_left['Class'] == 'Assam']
    df_right_Assam = df_right[df_right['Class'] == 'Assam']

    p_left = np.size(df_left_Assam[attribute]) / np.size(df_left[attribute])
    p_right = np.size(df_right_Assam[attribute]) / np.size(df_right[attribute])
    
    # print('  threshold = ' + str(threshold))
    # print('  p_left_assum = ' + str(p_left) )
    # print('  p_right_assum = ' + str(p_right) )
    # print('  ' + str(split_rate))
    # print("")
    # print("p_left = " + str(p_left))
    # print("p_right = " + str(p_right))
  
    ent_left =  0 if p_left == 0 or p_left == 1 \
                else - p_left * math.log(p_left, 2) \
                    - (1 - p_left) * math.log ((1 - p_left), 2)
    ent_right = 0 if p_right == 0 or p_right == 1 \
                else - p_right * math.log(p_right, 2) \
                    - (1 - p_right) * math.log ((1 - p_right), 2)
  
    # print("ent_left = " + str(ent_left))
    # print("ent_right = " + str(ent_right))

    ent_average = (ent_left + ent_right) / 2
    # print("ent_average = " + str(ent_average) )
    # print("")
    return ent_average, split_rate

def threshold_selection_in_attribute(dataframe, attribute, step):
    min_threshold = np.min(dataframe[attribute]) # starting threshold 
    max_threshold = np.max(dataframe[attribute]) # ending threshold

    best_threshold = min_threshold # best threshold that provides minimized entropy
    best_ent_average = np.inf # the minimized average entropy
    threshold = min_threshold # threshold for traversing
    best_split_rate = np.inf

    while threshold != max_threshold:
        
        ent_average, split_rate = entropy_average(dataframe, attribute, threshold)
        
        if ent_average < best_ent_average:
            best_ent_average = ent_average
            best_threshold = threshold
            best_split_rate = split_rate
        if ent_average == best_ent_average and split_rate < best_split_rate:
            best_ent_average = ent_average
            best_threshold = threshold
            best_split_rate = split_rate
        threshold += step
    # print("attribute = " + attribute)
    # print("best ent_average = " + str(best_ent_average))
    # print("best threshold = " + str(best_threshold))
    # print("best spli rate = " + str(best_split_rate))
    # print("")
    return attribute, best_threshold, best_ent_average, best_split_rate

def class_assign(dataframe):
    dataframe_assam = dataframe[dataframe['Class'] == 'Assam']
    dataframe_bhuttan = dataframe[dataframe['Class'] == 'Buhttan']
    assam_count = np.size(dataframe_assam['Age'])
    bhuttan_count = np.size(dataframe_bhuttan['Age'])

    if assam_count < bhuttan_count:
        print("Class: Bhuttan")
    else:
        print("Class: Assam")

def decision_tree(dataframe, depth):
    # ---------------- Stop Criteria--------------------------------
    if np.size(dataframe['Age']) < 5:
        class_assign(dataframe)
        return None

    df_Assam = dataframe[dataframe['Class'] == 'Assam']
    class_rate = np.abs(np.size(df_Assam['Age']) / np.size(dataframe['Age']))
    if  class_rate > 0.95 or class_rate < 0.05:
        class_assign(dataframe)
        return None
    
    if depth > 3:
        class_assign(dataframe)
        return None
 
    # ---------------- Attribute Selection-------------------------------- 
    attributes = dataframe.columns
    best_attribute = ""
    best_threshold = np.inf # best threshold that provides minimized entropy
    best_ent_average = np.inf # the minimized average entropy
    best_split_rate = np.inf

    for attribute in dataframe.columns[0 : np.size(attributes) - 1]:
        # print(attribute)
        step = 1.0
        if attribute == 'Age': step = 2.0
        if attribute == 'Ht': step = 5.0
        attribute, threshold, ent_average, split_rate \
            = threshold_selection_in_attribute(dataframe, attribute, step)
        if ent_average < best_ent_average:
            best_attribute = attribute
            best_threshold = threshold
            best_ent_average = ent_average
            best_split_rate = split_rate

    # print("--------------------------------------------")
    # print("best attribute = " + best_attribute)
    # print("best threshold = " + str(best_threshold))
    # print("best ent_average = " + str(best_ent_average))
    # print("best spli rate =  " + str(best_split_rate))

    # ----------------Create current Node and recursively create child nodes--------------------------------

    left_node = decision_tree(dataframe[dataframe[best_attribute] <= best_threshold], depth + 1)
    right_node = decision_tree(dataframe[dataframe[best_attribute] > best_threshold], depth + 1)

    current_node = dt_node(best_attribute, best_threshold, best_ent_average, left_node, right_node, dataframe)
    print(depth)
    return current_node

def main():
    df_snowfolks_data_raw = pd.read_csv('Abominable_Data_HW05_v720.csv')
    df_snowfolks_data_quantized = data_preprocessing(df_snowfolks_data_raw)

    root = decision_tree(df_snowfolks_data_quantized, 0)


    # print(df_snowfolks_data[df_snowfolks_data['Class'] == 'Assam'])
    # print(df_snowfolks_data_raw.columns)
    # print(df_snowfolks_data_raw.columns[1:3]) # select columns titles
    # print(df_snowfolks_data_raw['Reach']- 100) # split dataframe based on threshold
    
    # data_ages= np.round(df_snowfolks_data_raw['Age'] / 2) * 2 # 
    # df_snowfolks_data_raw['Age'] = data_ages

    # data_height= np.round(df_snowfolks_data_raw['Ht'] / 5) * 5 # 
    # df_snowfolks_data_raw['Ht'] = data_height



if __name__ == "__main__":
    main()