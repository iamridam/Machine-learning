#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[97]:


import numpy as np
import pandas as pd
import sys

# In[98]:


# For enabling the GPU support 
# %env CUDA_DEVICE_ORDER = PCI_BUS_ID
# %env CUDA_VISIBLE_DEVICES = 0,1


# In[99]:


alphabet = list("ACDEFGHIKLMNPQRSTVWXY")


# In[100]:


len(alphabet)


# # Training dataset

# In[101]:


trainfile=sys.argv[1]
testfile=sys.argv[2]
outputfile=sys.argv[3]
#train_data = pd.read_csv("Training_dataset.csv")
train_data = pd.read_csv(trainfile)



# In[102]:


train_data.info()


# In[103]:


train_data.head()


# In[104]:


train_data['Label'].value_counts()


# # Creating Simple Amino Acid Composition

# In[105]:


X_train = []

for label, sequence in train_data.values:
    frequency = {}
    for letter in alphabet:
        frequency[letter] = 0
    for letter in sequence:
        frequency[letter] = (frequency[letter] * len(sequence) + 1) / len(sequence)
    X_train.append(list(frequency.values()))


# In[106]:


X_train_df_1 = pd.DataFrame(X_train, columns=alphabet)


# In[107]:


X_train_df_1.head()


# In[108]:


X_train_df = pd.concat([X_train_df_1, train_data[['Label']]], axis=1)


# # Loading Bipeptide Amino Acid Composition

# In[109]:


bipeptide = pd.read_csv('bipeptide_train.csv')


# In[110]:


bipeptide.head()


# In[111]:


bipeptide.shape


# # Loading Tripeptide Amino Acid Composition

# In[112]:


tripeptide = pd.read_csv('tripeptide_train.csv')


# In[113]:


tripeptide.shape


# In[114]:


X_train_df.shape


# In[115]:


X_train_df = pd.concat([tripeptide, bipeptide, X_train_df], axis=1)


# In[116]:


X_train_df.head()


# # Creating Training and Testing Data

# In[117]:


X_train = X_train_df.iloc[:, :-1].values
y_train = X_train_df.iloc[:, -1].values


# In[118]:


for index, label in enumerate(y_train):
    if label == -1:
        y_train[index] = 0


# In[119]:


y_train


# In[120]:


print(X_train.shape)
print(y_train.shape)


# # Feature Selection using ExtraTreeClassifier

# In[121]:


# Selecting top features using ExtraTreesClassifier.

from sklearn.ensemble import ExtraTreesClassifier

TOP_FEATURES = 2900

forest = ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in forest.estimators_],
    axis=0
)
indices = np.argsort(importances)[::-1]
indices = indices[:TOP_FEATURES]

print('Top features:')
for f in range(TOP_FEATURES):
    print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))




# Selecting top features from all the features.
X1 = X_train_df.columns.drop(['Label'])
X1=X_train_df.columns.tolist()
X1 = [X1[i] for i in indices]


# In[124]:


X_train = X_train_df[X1]


# # Running the model

# In[125]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)




# # Cross Validation on Train Data

# In[126]:


from sklearn.model_selection import cross_val_score
cross_val_rfc = cross_val_score(estimator=RandomForestClassifier(random_state=0), X=X_train, y=y_train, cv=10, n_jobs=-1)
print("Cross Validation Accuracy : ",round(cross_val_rfc.mean() * 100 , 2),"%")


# # Loading the Test Dataset

# In[127]:


#test_data = pd.read_csv("validation_dataset.csv")
test_data = pd.read_csv(testfile)

# In[128]:


test_data.info()


# In[129]:


test_data.head()


# In[130]:


X_test = []

for _id, sequence in test_data.values:
    frequency = {}
    for letter in alphabet:
        frequency[letter] = 0
    for letter in sequence:
        frequency[letter] = (frequency[letter] * len(sequence) + 1) / len(sequence)
    X_test.append(list(frequency.values()))


# In[131]:


X_test_df_1 = pd.DataFrame(X_test, columns=alphabet)


# In[132]:


X_test_df_1.head()


# In[133]:


bipeptide_test = pd.read_csv('bipeptide_test.csv')


# In[134]:


bipeptide_test.head()


# In[135]:


tripeptide_test = pd.read_csv('tripeptide_test.csv')


# In[136]:


X_test_df = pd.concat([tripeptide_test, bipeptide_test, X_test_df_1], axis=1)


# In[137]:


X_test_df = X_test_df[X1]


# In[138]:


X_test_df.shape


# # Predicting the output

# In[139]:


predict = clf.predict(X_test_df)
predict.shape




# In[140]:


valid = pd.read_csv('validation_dataset.csv')
lab = valid.ID


# In[141]:


predict = pd.DataFrame(predict)


# In[142]:


#Creating the output dataframe
output = pd.concat([lab,predict],axis=1)
output.head()


# In[143]:


# creating the output csv file
output.to_csv(outputfile, header=['ID','Label'], index=False)




