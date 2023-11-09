#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


columns=['UserId','ProductId','Ratings','timestamp']
df=pd.read_csv("C:\\Users\\Honey\\Documents\\datasets\\projectsAI\\ratings_Electronics.csv",names=columns)


# In[3]:


df.head()


# In[4]:


df.drop('timestamp',axis=1,inplace=True)


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


df1=df.iloc[:50000,0:]


# In[10]:


df1['Ratings'].describe().transpose()


# In[11]:


df1.Ratings.max()


# In[12]:


df1.Ratings.min()


# In[13]:


df1['Ratings'].value_counts()


# In[14]:


df1.isnull().sum()


# In[15]:


sns.countplot(x='Ratings',data=df)


# In[16]:


print('Number of unique users in Raw data = ',  df1['UserId'].nunique())
# Number of unique product id  in the data
print('Number of unique product in Raw data = ', df1['ProductId'].nunique())


# In[17]:


most_rated=df1.groupby('UserId').size().sort_values(ascending=False)[:10]
print(most_rated)


# In[18]:


counts=df1.UserId.value_counts()


# In[62]:


print(counts)


# In[19]:


df1_final=df1[df1.UserId.isin(counts[counts>=15].index)]


# In[63]:


print(df1_final)


# In[20]:


print(len(df1_final))


# In[21]:


print(df1_final['UserId'].nunique())


# In[22]:


print(df1_final['ProductId'].nunique())


# In[23]:


final_ratings_matrix=df1_final.pivot(index='UserId',columns ='ProductId', values = 'Ratings').fillna(0)
print(final_ratings_matrix)


# In[24]:


final_ratings_matrix.shape


# In[25]:


#rating density,rating frequency,rating distribution for reccomendation system.
#calculating the density matrix
given_num_of_ratings=np.count_nonzero(final_ratings_matrix)
print('given nmber of ratings', given_num_of_ratings)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)
density = (given_num_of_ratings/possible_num_of_ratings)
density *= 100
print ('density: {:4.2f}%'.format(density))


# In[26]:


train_data,test_data=train_test_split(df1_final,test_size=0.3,random_state=0)


# In[27]:


train_data.head()


# In[28]:


train_data.shape


# In[29]:


test_data.shape


# In[30]:


#Count of user_id for each unique product as recommendation score 
train_data_grouped = train_data.groupby('ProductId').agg({'UserId': 'count'}).reset_index()
train_data_grouped.rename(columns = {'UserId': 'score'},inplace=True)
train_data_grouped.head(40)


# In[31]:


#Sort the products on recommendation score 
train_data_sort = train_data_grouped.sort_values(['score', 'ProductId'], ascending = [0,1]) 
      
#Generate a recommendation rank based upon score 
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') 
          
#Get the top 5 recommendations 
popularity_recommendations = train_data_sort.head(5) 
popularity_recommendations 


# In[32]:


def recommend(user_id):     
    user_recommendations = popularity_recommendations 
          
    #Add user_id column for which the recommendations are being generated 
    user_recommendations['userId'] = user_id 
      
    #Bring user_id column to the front 
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendations = user_recommendations[cols] 
          
    return user_recommendations 


# In[33]:


find_recom = [10,100,150]   # This list is user choice.
for i in find_recom:
    print("The list of recommendations for the userId: %d\n" %(i))
    print(recommend(i))    
    print("\n") 


# In[34]:


electronics_df_CF = pd.concat([train_data, test_data]).reset_index()
electronics_df_CF.head()


# In[35]:


pivot_df = electronics_df_CF.pivot(index = 'UserId', columns ='ProductId', values = 'Ratings').fillna(0)
pivot_df.head()


# In[36]:


pivot_df.shape


# In[37]:


#define user index from 0 to 10
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)
pivot_df.head()


# In[38]:


#define user index from 0 to 10
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)
pivot_df.head()


# In[39]:


pivot_df.set_index(['user_index'], inplace=True)
# Actual ratings given by users
pivot_df.head()


# In[44]:


type(pivot_df)


# In[45]:


# Singular Value Decomposition
U, sigma, Vt = svds(pivot_df.to_numpy(), k = 10)


# In[46]:


print("Left Singular",U)


# In[47]:


sigma =np.diag(sigma)
print(sigma)


# In[48]:


print("Right Singular",Vt)


# In[50]:


all_user_predicted_items=np.dot(np.dot(U,sigma),Vt)


# In[52]:


preds_df = pd.DataFrame(all_user_predicted_items, columns = pivot_df.columns)
preds_df.head()


# In[53]:


def recommend_items(userID, pivot_df, preds_df, num_recommendations):
    # index starts at 0  
    user_idx = userID-1 
    # Get and sort the user's ratings
    sorted_user_ratings = pivot_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_ratings
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_predictions
    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    temp.index.name = 'Recommended Items'
    temp.columns = ['user_ratings', 'user_predictions']
    temp = temp.loc[temp.user_ratings == 0]   
    temp = temp.sort_values('user_predictions', ascending=False)
    print('\nBelow are the recommended items for user(user_id = {}):\n'.format(userID))
    print(temp.head(num_recommendations))


# In[54]:


userID = 4
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations) 


# In[55]:


userID = 7
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations) 


# In[57]:


final_ratings_matrix.head()


# In[58]:


# Average ACTUAL rating for each item
final_ratings_matrix.mean().head()


# In[59]:


rmse_df = pd.concat([final_ratings_matrix.mean(), preds_df.mean()], axis=1)
rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
print(rmse_df.shape)
rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)
rmse_df.head()


# In[60]:


RMSE = round((((rmse_df.Avg_actual_ratings - rmse_df.Avg_predicted_ratings) ** 2).mean() ** 0.5), 5)
print('\nRMSE SVD Model = {} \n'.format(RMSE))


# In[61]:


# Enter 'userID' and 'num_recommendations' for the user #
userID = 9
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)


# In[ ]:




