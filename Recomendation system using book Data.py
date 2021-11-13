#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[8]:


#import Dataset 
book = pd.read_csv("C:\\Users\\Admin\\Downloads\\book (3).csv")
book.head


# In[10]:


book.shape #shape
book.columns


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus


# In[14]:


# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words="english")    #taking stop words from tfid vectorizer 


# In[25]:


# replacing the NaN values in overview column with
# empty string
book["Book.Title"].isnull().sum() 
book["Book.Title"] = book["Book.Title"].fillna(" ")


# In[26]:


# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix = tfidf.fit_transform(book.columns)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294,46
# with the above matrix we need to find the 
# similarity score
# There are several metrics for this
# such as the euclidean, the Pearson and 
# the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity 
# between 2 movies 
# Cosine similarity - metric is independent of 
# magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)


# In[28]:


from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)


# In[35]:


# creating a mapping of anime name to index number 
book_index = pd.Series(book.index,index=book['Book.Title']).drop_duplicates()


# In[52]:


book_index["The Kitchen God's Wife"]

def get_book_recommendations(Name,topN):

    #topN = 10
    # Getting the movie index using its title 
    User.ID = book_index[int]


# In[54]:


#topN = 10
   # Getting the movie index using its title 
   User.ID = book_index[Name]
# Getting the pair wise similarity score for all the anime's with that 
   # anime
   cosine_scores = list(enumerate(cosine_sim_matrix[User.ID]))


# In[56]:


# Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    


# In[58]:


# Get the scores of top 10 most similar anime's 
   cosine_scores_10 = cosine_scores[0:topN+1]


# In[59]:


# Getting the anime index 
   book_idx  =  [i[0] for i in cosine_scores_10]
   anime_scores =  [i[1] for i in cosine_scores_10]
   
   # Similar movies and scores
   book_similar_show = pd.DataFrame(columns=["name","Score"])
   book_similar_show["name"] = anime.loc[anime_idx,"name"]
   book_similar_show["Score"] = anime_scores
   book_similar_show.reset_index(inplace=True)  
   book_similar_show.drop(["index"],axis=1,inplace=True)
   print (anime_similar_show)
   #return (anime_similar_show)

   
# Enter your anime and number of anime's to be recommended 
get_book_recommendations("PLEADING GUILTY",topN=15)

