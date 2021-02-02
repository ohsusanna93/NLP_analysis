
# Introduction

We are building a model that can predict the sentiment of a tweet based on its content. Giving insight to the manufactures of Google and Apple hardware. The tweet reviews are on the following products and services.


- IPhone
- Ipad
- Apple Apps (Iphone/Ipad)
- Other Apple product or service 


- Google
- Android
- Android Apps
- Other Google product or service 


Using a Natural Language Processing model allows us to analyze text data, which makes analyzing the score of the 9,093 product tweets possible. Finding the correlation and important features of the positive and negative feedback helps provide insight to the what makes up a positive review and negative review.


The data set we used has three columns which includes the tweet (the review), which product the review is referring to, and whether or not the review was positive, negative, or neutral. 


Below are all libraries and programs used in building our models:


```python
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import regexp_tokenize

from wordcloud import WordCloud
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from IPython.display import Image  

import warnings
warnings.filterwarnings('ignore')
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/susannahan/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


Functions:


```python
def evaluate_model(y_test,y_pred,X_test,clf=None,
                  scoring=metrics.recall_score,verbose=False,
                   figsize = (10,4),
                   display_labels=('Negative','Neutral','Positive')):
    """
    Arguments needed to evaluate the model are y_test, y_pred, x_test, 
    the model, and display labels.
    
    Evaluate_model prints out the precision, recall, and f1-score score. As well as accuracy, 
    macro average, and weighted average.
    
    Below, a Confusion Matrix visual shows the percentage of how accurate the model fit both predicted and actual
    values. 
    
    """
    ## Classification Report / Scores 
    print(metrics.classification_report(y_test,y_pred))
    # plots Confusion Matrix
    metrics.plot_confusion_matrix(clf,X_test,y_test,cmap="Blues",
                                  normalize='true', 
                                  display_labels = display_labels)
    #plt.title('Confusion Matrix')
    plt.show()

    try: 
        df_important = plot_importance(clf)
    except:
        df_important = None
        
```


```python
def plot_importance(tree_clf, top_n=20,figsize=(10,8)):
    """ Arguments needed to plot an importance bar graph is the model, number of features to display, and 
    desired figsize for the graph. 
    
    This function displays a bar graph of top 10 important features from most to least important."""
    
    #calculates which feature was used the most in the model.
    df_importance = pd.Series(tree_clf.feature_importances_,vectorizer.get_feature_names())
    
    #sorts 20 important features data in ascending order
    df_importance.sort_values().tail(10).plot(
        kind='barh', figsize=figsize)

    #graph labels
    
    #plt.title('Top Important Features')
    plt.xlabel('Features Importance')
    plt.ylabel('Features')


    plt.show() 

    return df_importance
```

# Observations

First, the data is imported and explored looking at the different columns and rows. What type of data is present and how to manipulate the data for some insight.


One of the first observations we see is that the column names are very long which makes it more difficult to recall and work with. Therefore, they were modified and renamed to tweets, product, and emotion as shown below.

We have decided to keep all missing values in the product column to add more data to the correlation between tweet and emotion. The one missing row from the tweets column is dropped because it adds no value to the dataset.

Now that we have taken care of the missing values we look into all the categories in each columns. In the "emotion" column we see that there are 4 categories and the " I can't tell " category only having 1% of the data. Therefore, dropping that category as it does not provide much information needed and is only a small portion of the dataset. 


We are able to see the imbalance in our dataset as there are many more neutral tweets than there are positive and negative. Therefor we are going to separate the data into it's individual dataset to extract information from what is given.

# Processing Data

After cleaning out the dataset we started to clean the text in the tweets column to properly train the model to process the given text. We use word_tokenize to separate each word and punctuation to more accurately get rid of empty spaces/words when using stopwords. 

Stopwords is a list of common words that do not add meaning to a sentence. A 'more_punc' list was created to add to the stopwords list that were common in the data texts that didn't add any value.

We used regular expression to find all urls, hashtags, retweets, and mention patterns to be replaced with '', which are all common tweet texts. 

Then we figured out the frequency distribution for all the stopped_tokens as well as the bigrams to see the top most common words used in the text.

A target column was created that indicated the tweet expressed a Positive emotion. Another column was created for tweets that had no emotion toward brand or product. Lastly, the negative column was created to count for the negative responses in the text.


Creating separate columns helps vectorize the text when pulling the text through the model created. 

# Model 1
The base model uses Random Forest Classifier. The overall base model has an accuracy of 70.4%.

# Model 2

The second model also uses Random Forest Classifier but the parameters are tested using Randomized Search CV. The model has an overall accuracy of 69.8%

# Results and Conclusion

Even though Model 2 had a one percent increase in the overall accuracy. The negative responses accuracy decreased in the second model. Which means Model 1 is the better model for predicting positive and negative responses in a given text.

Model 1 performance:
    - Positive 49% accuracy 
    - Negative 22% accuracy
    - Neutral 87% accuracy
    
    
    
 Model 2 performance:
     - Positive 50% accuracy
     - Negative 11% accuracy
     - Neutral 87% accuracy
     
     

Both models are overfit as there is not enough text data for positive and negative responses to train the model. If the end result was to analyze data that have neutral responses. Model 2 would be the best model to use as it is not as overfit as model 1.

# Insight and Recommendations

Due to the time frame at which the text data was extracted. It is recommended to keep in mind the context of the results as there are not many negative responses as positive, and positive as neutral. Also, positive responses uses a bigger variety of words where as negative responses use the same repeated words. 

The top words used in negative responses include:
 - ipad2, 61
 - store, 44
 - new, 43
 - like, 39
 - design, 28
 - social, 28
 - people, 27
 - circles, 26
 - apps, 26
 - need, 25

The top words used in positive responses include:
 - know, 1
 - awesome, 1
 - ipad/iphone, 1
 - likely, 1
 - appreciate, 1
 - design, 1
 - giving, 1
 - free, 1
 - ts, 1
 - wait, 1

### Things to look into:
1. There are a lot of negative responses correlated with the ipad2.


2. In both the negative and positive responses people mention the design. Knowing that a lot of people tweeting about apple and google products are interested in the design. 


3. There are positive responses towards the ipad/iphone.

