#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd


# In[80]:


from sklearn import tree
df = pd.read_csv("C:/Users/gengl/Desktop/Artificial Intel and Mach Learn/project2/drug200.csv", delimiter =",")
df.head()


# In[95]:


"""
Using my_data as the Drug.csv data read by pandas, declare the following variables:

X as the Feature Matrix (data of my_data)
y as the response vector (target)
"""
#Remove the column containing the target name since it doesn't contain numeric values.
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# In[96]:


"""
As you may figure out, some features in this dataset are categorical such as Sex or BP. 
Unfortunately, Sklearn Decision Trees do not handle categorical variables. 
But still we can convert these features to numerical values. pandas.get_dummies() 
Convert categorical variable into dummy/indicator variables.
"""
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# In[97]:


#Now we can fill the target variable.
y = df["Drug"]
y[0:5]


# In[98]:


"""

Setting up the Decision Tree
We will be using train/test split on our decision tree. 
Let's import train_test_split from sklearn.cross_validation.
"""
from sklearn.model_selection import train_test_split


# In[100]:


"""
Now train_test_split will return 4 different parameters. We will name them:
X_trainset, X_testset, y_trainset, y_testset

The train_test_split will need the parameters:
X, y, test_size=0.3, and random_state=3.

The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, 
and the random_state ensures that we obtain the same splits.
"""

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# In[101]:


"""
Practice
Print the shape of X_trainset and y_trainset. 
Ensure that the dimensions match
"""
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))


# In[102]:


print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))


# In[104]:


"""
Modeling
We will first create an instance of the DecisionTreeClassifier called drugTree.
Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
"""
from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# In[105]:


#Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)


# In[106]:


"""Prediction"""
predTree = drugTree.predict(X_testset)


# In[107]:


"""
You can print out predTree and y_testset if you want to visually compare the prediction to the actual values.
"""
print (predTree [0:5])
print (y_testset [0:5])


# In[108]:


#Evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[109]:


"""
Accuracy classification score computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
In multilabel classification, the function returns the subset accuracy. 
If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
"""


# In[ ]:




