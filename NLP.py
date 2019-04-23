
# coding: utf-8

# In[104]:


#Import Libraries
import numpy as np
from sklearn.utils import shuffle
import re
import nltk
nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
corpus = []


# In[105]:


#Import Data
data = pd.read_csv(r"C:/users/asus/desktop/all3.csv", encoding='latin-1')


# In[106]:


data.info()


# In[107]:


sns.countplot("type",data=data)
#Plot reveals that the dataset is highly unbalanced, Therefore,
#it is better to take equal number of samples for each class to avoid
# one class dominate the predictions


# In[108]:


#Balancing Dataset
data1 = data[data["type"]=="Love"]
data1 =data1.sample(n=100)
data2 = data[data["type"]=="Nature"]
data2 =data2.sample(n=100)
data3 = data[data["type"]=="Divine"]
data3 =data3.sample(n=100)
data4 = data[data["type"]=="WAR"]
data4 =data4.sample(n=100)
frames = [data1,data2,data3,data4]
dataf=pd.concat(frames)
dataf=shuffle(dataf)
dataf


# In[109]:


#Encoding Categorical Variables
y = dataf.iloc[:,-1].values
poem = dataf.iloc[:,1:2].values
#y = data["type"]
#poem = data["content"]
from sklearn.preprocessing import LabelEncoder
y1 = LabelEncoder().fit_transform(y)
y1 = np.reshape(y1,(400,1))


# In[110]:


#Cleaning datasets
corpus = []
for i in range(0,400):
    poem1 = re.sub(r'\s+',' ',poem[i,0])
    poem2 = re.sub('[^a-zA-Z]',' ',poem1)
    poem2.lower()
    poem2=poem2.split()
    #poem2.split()
    ps = PorterStemmer()
    data3 = [ps.stem(word) for word in poem2 if not word in set(stopwords.words('english'))]
    data4 = ' '.join(data3)
    corpus.append(data4)


# In[111]:


#Bag of works
cv = CountVectorizer(max_features = 3500)
X = cv.fit_transform(corpus).toarray()


# In[112]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = 0.20, random_state = 0)


# In[137]:


#Using Random forest Classifier
classifier = RandomForestClassifier(n_estimators=1000,random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
ac_RF = accuracy_score(y_test,y_pred)
cm_RF = confusion_matrix(y_test,y_pred)
print(ac_RF*100) 
print('--------------------')
print(cm_RF)


# In[138]:


#Using Naive Bayes Classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train,y_train)
predictions_naive = Naive.predict(X_test)
ac_naive= accuracy_score(y_test,predictions_naive)
cm_naive = confusion_matrix(y_test,predictions_naive)
print(ac_naive*100) 
print('--------------------')
print(cm_naive)

