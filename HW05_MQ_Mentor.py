# -----------------------------------------------------------
# CSCI-720-Big-Data-Analytics
# Topic: Decistion Tree
# Assignment: HW05
#
# Author: Zizhun Guo
# Email: zg2808@cs.rit.edu
# Author: Martin Qian
# Email: jq3513@rit.edu
# 
# RIT, Rochester, NY
#
# 
# 
# 
#
#
# Zizhun GUO @ All rights researved
# -----------------------------------------------------------

import numpy as np # fundamental scientific computing package
import pandas as pd # data analysis and manipulation package
import matplotlib.pyplot as plt # plotting package
from datetime import datetime # python std data package
import sys # for using command line argument
import math # for entropy calculation


class dt_node:
    """ Decision Tree Node
    Memember:
        @attribute: a string that stores the current condition attribute
        @threshold: a float value that stores the current condition threshold
        @ent_average: a float value that represents the current responding entropy
        @left_node: a dt_node that represents the left child decision node
        @right_node: a dt_node that represents the right child decision node
        @dataframe: a Pandas dataframe that stores the copy of the current sliced dataframe
        @leaf: a str that represents which side is the leaf node
        @category: a boolean that indicates the category of the leaf node
    """
    def __init__(self, attribute = None, threshold = None, loss = None,\
         left_node = None, right_node = None, dataframe = None):
        self.attribute = attribute
        self.threshold = threshold
        self.loss = loss
        self.left_node = left_node 
        self.right_node = right_node
        self.dataframe = dataframe
        
        if(left_node == None and right_node == None):
            data_leaf=dataframe[dataframe[attribute] <= threshold]
            dataframe_assam = data_leaf[data_leaf['Class'] == 'Assam']
            dataframe_bhuttan = data_leaf[data_leaf['Class'] == 'Bhuttan']
            assam_count = np.size(dataframe_assam['Age'])
            bhuttan_count = np.size(dataframe_bhuttan['Age'])
            self.leaf = 'last_node'
            self.category = assam_count>bhuttan_count
        elif(left_node == None):
            data_leaf=dataframe[dataframe[attribute] <= threshold]
            dataframe_assam = data_leaf[data_leaf['Class'] == 'Assam']
            dataframe_bhuttan = data_leaf[data_leaf['Class'] == 'Bhuttan']
            assam_count = np.size(dataframe_assam['Age'])
            bhuttan_count = np.size(dataframe_bhuttan['Age'])
            self.leaf = "left"
            self.category = assam_count>bhuttan_count
        elif(right_node == None):
            data_leaf=dataframe[dataframe[attribute] >= threshold]
            dataframe_assam = data_leaf[data_leaf['Class'] == 'Assam']
            dataframe_bhuttan = data_leaf[data_leaf['Class'] == 'Bhuttan']
            assam_count = np.size(dataframe_assam['Age'])
            bhuttan_count = np.size(dataframe_bhuttan['Age'])
            self.leaf = "right"
            self.category = assam_count>bhuttan_count


def pre_traverse_(node):
    ''' pre traverse the tree, from the root

    '''
    print('attribute:'+str(node.attribute))
    print('threshold:'+ str(node.threshold))
    print('loss:'+str(node.loss))
    print('size:'+str(np.size(node.dataframe['Age'])))
    print('leaf node:'+str(node.leaf))
    print('category==Assam:'+str(node.category)+'\n')

    if node.left_node != None:
        pre_traverse_(node.left_node)
    
    if node.right_node != None: 
        pre_traverse_(node.right_node)

def pre_traverse(node,tree_body,if_flag):
    ''' pre traverse the tree
        used for generating trained program
    '''

    ''' doing all decisions of parameters

    '''
    if(node.leaf=='right'):
        symbol='>'
    else: symbol='<='
    if(node.category):
        category='Assam'
    else: category='Bhuttan'
    if(if_flag):
        if_str='if'
    else: if_str='elif'

    tree_body+='\t\t'+if_str+"(row['"+str(node.attribute)+"'])"+\
        symbol+str(node.threshold)+':\n'
    tree_body+="\t\t\tcategory_pre.append('"+category+"')\n"

    if node.left_node != None:
        tree_body=pre_traverse(node.left_node, tree_body, False)
    
    if node.right_node != None: 
        tree_body=pre_traverse(node.right_node, tree_body, False)

    ''' special case for the last node

    '''
    if(node.leaf=='last_node'):
        if(node.category):
            category='Bhuttan'
        else: category='Assam'
        tree_body+='\t\telse: \n'
        tree_body+="\t\t\tcategory_pre.append('"+category+"')\n"
    
    return tree_body

def data_preprocessing(dataframe):
    """ Data preprocessing doing quantization
    Paras:
        @dataframe: a dataframe of data read from CSV file
    Return:
        dataframe: the modified dataframe after quantization
    """
    columns_size = np.size(dataframe.columns)
    columns = dataframe.columns[0 : columns_size - 1]

    for column in columns:
        data = np.round(dataframe[column])
        dataframe[column] = data
    
    data_ages= np.round(dataframe['Age'] / 2) * 2 # 
    dataframe['Age'] = data_ages
    data_height= np.round(dataframe['Ht'] / 5) * 5 # 
    dataframe['Ht'] = data_height
    return dataframe

def lost_function(dataframe, attribute, threshold):
    """ LostFunction = ObjectiveFunction + Regularization
        ObjectiveFunction: misclassification rate
        Regularization: how unbalanced split rate is
    Paras:
        @dataframe: a dataframe of quantized data
        @attribute: a string of the chosen attribute name
        @threshold: a float type of chosen threshold
    Returns:
        @Lost Function: cost function we get
        @split rate: evalating how unbalanced the split is

    """

    # left sliced dataframe (<= threshold) 
    df_left = dataframe[dataframe[attribute] <= threshold]
    # right sliced dataframe (> threshold)      
    df_right = dataframe[dataframe[attribute] > threshold]
     
    split_rate = np.abs((np.size(df_left[attribute]) \
                        - np.size(df_right[attribute]))) \
                        / np.size(dataframe[attribute])
    df_left_Assam = df_left[df_left['Class'] == 'Assam']
    df_right_Assam = df_right[df_right['Class'] == 'Assam']

    pa_left = np.size(df_left_Assam[attribute]) / \
                    np.size(df_left[attribute])
    pa_right = np.size(df_right_Assam[attribute]) / \
                    np.size(df_right[attribute])
    
    # meeting the requirements for accuracy
    if((pa_left < 0.8 and pa_right < 0.8)and(pa_left > 0.2 and pa_right > 0.2)):
        return 10,split_rate
    
    # calculate cost function
    Regularization = split_rate
    ObjectiveFunction=min(min(1- pa_left, 1- pa_right),min(pa_right, pa_left))
    return (ObjectiveFunction + 2*Regularization)/3, split_rate

def entropy_average(dataframe, attribute, threshold):
    """ Average weighted Purity calculation for one chosen attribute
        and threshold, this function is not used due to the change of 
        cost function  
    Paras:
        @dataframe: a dataframe of quantized data
        @attribute: a string of the chosen attribute name
        @threshold: a float type of chosen threshold
    Return:
        dataframe: a dataframe of data
    """
    # left sliced dataframe (<= threshold) 
    df_left = dataframe[dataframe[attribute] <= threshold] 
    # right sliced dataframe (> threshold)    
    df_right = dataframe[dataframe[attribute] > threshold]      
    split_rate = np.abs((np.size(df_left[attribute]) \
                        - np.size(df_right[attribute]))) \
                        / np.size(dataframe[attribute])
    df_left_Assam = df_left[df_left['Class'] == 'Assam']
    df_right_Assam = df_right[df_right['Class'] == 'Assam']

    p_left = np.size(df_left_Assam[attribute]) / np.size(df_left[attribute])
    p_right = np.size(df_right_Assam[attribute]) / np.size(df_right[attribute])
    
    ent_left =  0 if p_left == 0 or p_left == 1 \
                else - p_left * math.log(p_left, 2) \
                    - (1 - p_left) * math.log ((1 - p_left), 2)
    ent_right = 0 if p_right == 0 or p_right == 1 \
                else - p_right * math.log(p_right, 2) \
                    - (1 - p_right) * math.log ((1 - p_right), 2)

    weight_left = np.size(df_left['Age']) / np.size(dataframe['Age'])
    weight_right =  1 - weight_left

    # getting entropy of the split
    ent_average = ent_left * weight_left + ent_right * weight_right

    return ent_average, split_rate

def threshold_selection_in_attribute(dataframe, attribute, step):
    min_threshold = np.min(dataframe[attribute]) # starting threshold 
    max_threshold = np.max(dataframe[attribute]) # ending threshold

    # best threshold that provides minimized entropy
    best_threshold = min_threshold 
    best_loss = np.inf # the minimized average entropy
    threshold = min_threshold # threshold for traversing
    best_split_rate = np.inf # best split rate

    while threshold < max_threshold:
        
        loss, split_rate = lost_function(dataframe, attribute, threshold)
        
        if loss < best_loss:
            best_loss = loss
            best_threshold = threshold
            best_split_rate = split_rate
        if loss == best_loss and split_rate < best_split_rate:
            best_loss = loss
            best_threshold = threshold
            best_split_rate = split_rate
        
        threshold += step
    
    return attribute, best_threshold, best_loss, best_split_rate

def class_assign(dataframe):
    # a printing function for all leaf node we get
    dataframe_assam = dataframe[dataframe['Class'] == 'Assam']
    dataframe_bhuttan = dataframe[dataframe['Class'] == 'Bhuttan']
    assam_count = np.size(dataframe_assam['Age'])
    bhuttan_count = np.size(dataframe_bhuttan['Age'])
    
    print("Class: Assam: " + str(assam_count))
    print("Class: Bhuttan : "+ str(bhuttan_count))
    

def decision_tree(dataframe, depth):
    # ---------------- Stop Criteria--------------------------------
    df_Assam = dataframe[dataframe['Class'] == 'Assam']
    class_rate = np.abs(np.size(df_Assam['Age']) / np.size(dataframe['Age']))
    if  class_rate > 0.8 or class_rate < 0.2:
        print('leaf node')
        class_assign(dataframe)
        return None

    if np.size(dataframe['Age']) < 15:
        print('leaf node')
        class_assign(dataframe)
        return None
    
    if depth >= 8:
        print('leaf node generated:')
        class_assign(dataframe)
        return None
 
    # ---------------- Attribute Selection-------------------------------- 
    attributes = dataframe.columns
    best_attribute = ""
    best_threshold = np.inf # best threshold that provides minimized entropy
    best_loss = np.inf # the minimized average entropy
    best_split_rate = np.inf

    for attribute in dataframe.columns[0 : np.size(attributes) - 1]:
        # attributes selection and threshold testing
        step = 1.0
        if attribute == 'Age': step = 2.0
        if attribute == 'Ht': step = 5.0
        attribute, threshold, loss, split_rate \
            = threshold_selection_in_attribute(dataframe, attribute, step)
        if loss < best_loss:
            best_attribute = attribute
            best_threshold = threshold
            best_loss = loss
            best_split_rate = split_rate

    # ------Create current Node and recursively create child nodes-------

    left_node = decision_tree(dataframe[dataframe[best_attribute] \
                                <= best_threshold], depth + 1)
    right_node = decision_tree(dataframe[dataframe[best_attribute] \
                                > best_threshold], depth + 1)

    current_node = dt_node(best_attribute, best_threshold, best_loss, \
                                left_node, right_node, dataframe)

    return current_node

def trained_program_gen(root):
    headers=\
'import pandas as pd\nimport numpy as np\nimport sys\n\
def main():\n\t\
test_data_path=sys.argv[1]\n\t\
test_data=pd.read_csv(test_data_path)\n\t\
category_pre=[]\n'

    tails=\
'\tdf=pd.DataFrame(category_pre)\n\t\
df.to_csv("./HW05_MQ_MyClassifications.csv")\n\
if __name__ == "__main__":\n\t\
main()'
    
    tree_body='\tfor index,row in test_data.iterrows():\n'
    tree_body=pre_traverse(root,tree_body,if_flag=True)
    # generating codes
    return(headers+tree_body+tails)

def main():
    # read_csv
    df_snowfolks_data_raw = pd.read_csv(sys.argv[1])
    # quantized data
    df_snowfolks_data_quantized = data_preprocessing(df_snowfolks_data_raw)
    # tree generating
    root = decision_tree(df_snowfolks_data_quantized, 0)
    # show results
    print('tree structure')
    pre_traverse_(root)
    # trained_program_gen
    trained_program=open('./HW05_MQ_Trained_Classifier.py','w')
    trained_program.write(trained_program_gen(root))

if __name__ == "__main__":
    main()