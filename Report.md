a) Describe the structure of your final trained classifier program (the resulting classifier).

```python 
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

```


What does it tell you about the relative importance of the attributes?

The closer to the root decision node is, the heavier the importance is. Since the root decision node has the minimal cost than other nodes, so does the second minimal, the third monimal and so forth.


b) Generate a confusion matrix for the given training data:
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

c) What was the accuracy of your resulting classifier, on the training data?
$(799 - 101) / 799 = 0.8736 $

d) What was the hardest part of getting all this working?

The most difficult part is to balance the "pure" favors and "big" favors for cost function. It is hard to define which alpha value is the one that can produce the decision tree with highest accuracy. 


e) Did anything go wrong?
This answers links to the previous question. When using 0.95 as the accuracy for one of the stop criterias, the last tree would still have about 600 data records that has not yet been classified. The solution for this issue is written in conclusion part.

f) Discussion:
What does the number 23 have to do with anything?
What does this have to do with math, statistics, or decision making?
Or, did the professor pull this number out of a hat?



g) Conclusions

#### Contribution:
**Zizhun Guo:**
1 - decision node data structure
2 - entropy calculation function
3 - cost function for using average entropy
3 - split rate function
4 - decision make function (recursion)
5 - report writting
6 - code comments

**Martin Qian:**
1- Cost function modification for new requirment (misclassification + spit rate)
2 - Decision tree making testing
3 - trained file writting function
4 - csv file writting function


### Why use misclassification?
We have used two impurity measures as the object function: weighted entropy and misclassification.
It is more intuitive by using misclassification.

### **Decision Tree design decisions**
In order to find out the decision node for each level of decision tree, we assign a COST function (Cost = object function + regularization) to determines the attribute and its corresponding threshold. Therefore, the cost function is expressed in following:

$Cost = misclassficationRate + 2 * splitRate$

**Specifically, the object(misclassification rate) refers to "pure" favors, while regularization (split rate) is assigned for "big" favors.**

#### Design Decision 1: Purily test on "pure":
stop criteria =  accuracy 0.95 for misclassification for splitting leaf node.
> all size of node sliced dataframe is small except for the last node. (around 600 records left)
> The reason is the tree depth is limitted to 8.
> last node, impurity is big, which if it is used for determine the class for the final round, it has small accuray for the majority of the records in dataset.

#### Design decision 2: Mix "pure" and "big"
> In order to solve this problem, decides to adjust the porportion by favoring "big" number of records. The approach is to increase the weight of regualrtion portion within the cost function by assign a weight in value of 2. 
> The results improves a bit. However, it still cannot satisify the exptation to have records distribute evenly along with the tree nodes. 
> Even after with multiple tries by increasing the portaion of rgulartion, it still hurts the model. 
> This proves that the "pure" parameters has issue, so that all tested adjusted "big" parameters contributes less if only considering ajust split rates. *?*

#### Design decision 3: Mix still but adjust "pure" weight
> So in order to make improvement, it is necessary to modify the stop criteria.
> Based on the experiments by having the value of stop criteria tested in value range from 0.5 to 0.9 (step equals to 0.5), it was found that 0.8 is the best stop criteria.   


###**Underfitting vs Overfitting**

**Case 1: Decision tree becomes OneR model**
In the step of experiementing adjusting the stop criteria value, we found when the value equals to 0.7, the decision tree becomes in OneR model. This implys the tree is underfitting comparing to what was created when the value euqals to 0.8.

**Case 2: One more level of depth whereas accuracy unchanged**
However, when the stop criteria switched to 0.85, by having only one more decision node different with the tree created in 0.8, it became overfitting. 


