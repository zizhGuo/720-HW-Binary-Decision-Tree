---
puppeteer:
  landscape: false
  format: "A4"
  timeout: 3000 # <= Special config, which means waitFor 3000 ms
  printBackground: true
---

 #  <center> CSCI 720 Big Data Analytics HW03 Results </center> 
---
Student: Guo, Zizhun
Student: Qian, Martin
Submission: Mar/8/2020
Due Date: Mar/8/2020 11:59 PM 

---


### a) What were your names?
Martin Qian and Zizhun Guo

### b) Who did what roles during the assignment?
**Zizhun Guo:**
1 - decision node data structure
2 - entropy calculation function
3 - cost function for using average entropy
3 - split rate function
4 - decision tree building function (recursion)
5 - report 

**Martin Qian:**
1 - Cost function modification for new requirment (misclassification + spit rate)
2 - Decision tree parameter testing
3 - trained file writting function
4 - csv file writting function
5 - report 

### c) What cost function did you design for your splitting criterion?

How did you design it?

What was your objective function?

What part was your regularization?

$Cost = (misclassficationRate + 2 * splitRate)/3$

```python
    # calculate cost function
    Regularization = split_rate
    ObjectiveFunction=min(min(1- pa_left, 1- pa_right),min(pa_right, pa_left))
    return (ObjectiveFunction + 2*Regularization)/3, split_rate
```
See conclusion part to see why we choose this.

## d) Describe the structure of your final trained classifier program (the resulting classifier).
What does it tell you about the relative importance of the attributes?

```python 
    for row in test_data:
        if(row['BangLn']> 6.0):
            category_pre.append('Assam')
        elif(row['Ht']<= 140):
            category_pre.append('Bhuttan')
        elif(row['EarLobes']<=0):
            category_pre.append('Bhuttan')
        elif(row['TailLn']<= 12.0):
            category_pre.append('Assam')
        elif(row['BangLn']<= 5.0):
            category_pre.append('Bhuttan')
        else:
            category_pre.append('Assam')

```


The closer to the root decision node is, the heavier the importance is. Since the root decision node has the minimal cost than other nodes, so does the second minimal, the third monimal and so forth.

### c) Generate a confusion matrix for the given training data:
||True|False|
|---|---|---|
|Assam|347|53|
|Bhutan|351|48|


How many Assam were classified as Assam?
347
How many were classified as Bhutan?
53
How many Bhutan were classified as Assam?
48
How many Bhutan were classified as Bhutan?
351

### d) What was the accuracy of your resulting classifier, on the training data?
$(799 - 101) / 799 = 0.8736 $

### e) What was the hardest part of getting all this working?

The most difficult part is to balance the "**pure**" favors and "**big**" favors for the cost function. It is hard to define which alpha value is the one that can produce the decision tree with the highest accuracy. 


### f) Did anything go wrong?
This answers links to the previous question. When using 0.95 as the accuracy for one of the stop criteria, the last tree node would still have about 600 data records that have cannot be classified properly, which leads to really bad results.

### g) Discussion:
What would happen if you changed the number 15 to 45?

This value stands for the minimum number of records inside one node. It is used to determine when the recursion of decision tree would stop(It has the similar function of limit for depth. However, depth tells us what the highest depth is, this number tells us what when you can stop in advance if not reaching the highest depth). In my case, the second last node contains 56 element so chaning from 15 to 45 will make no difference. 

What if it was 5?

Changing from 15 to 5 will make the decision tree a little bigger, which means it has more depth.

What does this value control?

This value helps determine when to stop, so it controls the depth of the tree.

### h) Conclusions

#### Why use misclassification?
We have used two impurity measures as the object function: weighted entropy and misclassification.
To build a fast-decision cascade, the idea itself is based on correctness so it is more intuitive by using misclassification than entropy. Also, entropy is more complex and has little more impact than misclassification rate for this question.

#### **Decision Tree design decisions**
In order to find out the decision node for each level of the decision tree, we assign a COST function (Cost = object function + regularization) to determines the attribute and its corresponding threshold. Therefore, the cost function is expressed in the following:

$Cost = misclassficationRate + 2 * splitRate$

**Specifically, the object(misclassification rate) refers to "pure" favors, while regularization (split rate) is assigned for "big" favors.**

```python
    if  class_rate > 0.8 or class_rate < 0.2:
        print('leaf node')
        class_assign(dataframe)
        return None
```

#### Design Decision 1: Purily test on "pure":
Set **stop criteria** (accuracy) to 0.95 for misclassification to split leaf node. All size of node sliced dataframe is small except for the last node. (There are around 600 records left). The reason is the tree depth is limited to 8. For the last node, impurity is very high, which if it is used for determining the class for the final round, has small accuracy and it is the majority of the records in the dataset.
```Python
    if  class_rate > 0.95 or class_rate < 0.05:
        print('leaf node')
        class_assign(dataframe)
        return None
```

#### Design decision 2: Mix "pure" and "big"
In order to solve this problem, it decides to adjust the proportion by favoring the "big" number of records. The approach is to increase the **weight of the regularization** portion within the cost function by assigning a weight in value of 2. The results improve a bit. However, it still cannot satisfy the expectation to have records distribute evenly along with the tree nodes. Even after multiple tries by increasing the portion of regularization, the model still has the same problem.
```Python
(ObjectiveFunction + 2*Regularization)
```
#### Design decision 3: Adjust weight of "pure"
So in order to make the improvement, it is necessary to modify the stop criteria. Based on the experiments by having the value of stop criteria tested in a value range from 0.7 to 0.9 (step equals to 0.05), it was found that 0.8 is the best stop criteria, which gives us the best overall misclassification rate.   
**From**
```Python
    if  class_rate > 0.70 or class_rate < 0.30:
        print('leaf node')
        class_assign(dataframe)
        return None
```
**to**
```Python
    if  class_rate > 0.90 or class_rate < 0.10:
        print('leaf node')
        class_assign(dataframe)
        return None
```

#### **Underfitting vs Overfitting**

**Case 1: Decision tree becomes OneR model**
In the step of experimenting adjusting the stop criteria value, we found when the value equals to 0.7, the decision tree becomes in OneR model. This implies the tree is underfitting comparing to what was created when the value equals to 0.8.

```python 
    for row in test_data:
        if(row['BangLn']> 6.0):
            category_pre.append('Assam')
        else:
            category_pre.append('Bhuttan')

```

**Case 2: One more level of depth whereas accuracy unchanged**
However, when the stop criteria switched to 0.85, the model will do one more decision than that of 0.8. However, by comparing them we found the first 5 decisions made by them are all the same and the model formed by 0.85 is simply doing another decision inside the last node of the model of 0.8 and the total misclassification rate remained the same. So this suggests overfitting.
```python 
    for row in test_data:
        if(row['BangLn']> 6.0):
            category_pre.append('Assam')
        ...
        ...
        ...
        ...
        
        else:
            category_pre.append('Assam') # +1 level which is too overfitting

```


