#!/usr/bin/env python
# coding: utf-8

# # Dataset Information
# 
# The data set contains The dataset has 7 columns all categorized by their region.
# Try dataset to salute the world's various drug enforcement departments.
# https://www.kaggle.com/problemsolverdraly/drug-seizures/data
# 

# # Import modules

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import  matplotlib.pyplot as plt


# In[ ]:





# # Loading the dataset

# In[3]:


drug = pd.read_csv(r"C:\Users\king\Desktop\New folder (25)\Book1.csv")
drug.head(11)


# In[6]:


# delete a column
drug = drug.drop(columns = ['ISO Code'])
drug.head()


# In[7]:


# to display stats about data
drug.describe()


# In[8]:


# to basic info about datatype
drug.info()


# In[10]:


# to display no. of samples on each class
drug['Drug Group'].value_counts()


# In[11]:


drug['Drug'].value_counts()


# # Preprocessing the dataset

# In[12]:


# check for null values
drug.isnull().sum()


# In[ ]:





# # Exploratory Data Analysis

# In[20]:


# histograms
drug['Region'].hist()
plt.title('Region',fontsize=15)


# In[35]:


sns.set_style('whitegrid')
plt.figure(figsize=(35,15))
sns.countplot(x="Drug", data=drug, palette='cubehelix');
plt.xticks(rotation=45)


# In[37]:


drug['Drug Group'].hist()
plt.xticks(rotation=100)


# In[44]:


plt.figure(figsize=(30,15))
sns.countplot(x='Country',data=drug,palette='hls',order=drug['Country'].value_counts().head(10).index)
plt.xticks(rotation =45,fontweight='bold',fontsize=15)
plt.title('Top countries had seizures',fontweight='bold',fontsize=45)


# In[45]:


drug['Drug Group'].value_counts()


# In[46]:


drug['Drug'].value_counts()


# In[47]:


drug['Country'].value_counts()


# In[ ]:





# In[48]:


drug['Region'].value_counts()


# In[78]:


plt.figure(figsize=(15,5))
sns.countplot(x='Drug',data=drug,palette='cubehelix',order=drug['Drug'].value_counts().head(10).index)
plt.xticks(rotation=70,fontsize=15)
plt.title('Top drug groups causing seizures',fontweight='bold',fontsize=30)


# In[90]:


colors = ['red', 'orange']
species = ['Asia','Africa']


# In[97]:


for i in range(2):
    x = drug[drug['Region'] == species[i]]
    plt.scatter(x['Drug Group'], x['Drug'], c = colors[i], label=species[i])
    plt.xlabel("Drug Group")
plt.ylabel("Drug")
plt.legend()


# # Coorelation Matrix
# 
# A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The value is in the range of -1 to 1. If two varibles have high correlation, we can neglect one variable from those two.

# In[82]:


drug.corr()


# In[83]:


corr = drug.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# # Label Encoder
# 
# In machine learning, we usually deal with datasets which contains multiple labels in one or more than one columns. These labels can be in the form of words or numbers. Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form

# In[84]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[103]:


drug['Drug'] = le.fit_transform(drug['Drug'])
drug.head(25)


# In[104]:


drug['Drug Group'] = le.fit_transform(drug['Drug Group'])
drug.head(25)


# # Model Training

# In[ ]:




