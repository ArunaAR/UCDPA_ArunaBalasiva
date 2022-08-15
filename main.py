import pandas as pd
import numpy as np
# import re

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 14100)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns
import collections
import string
import nltk

from nltk.corpus import stopwords
nltk.download('stopwords')

import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

from collections import Counter
from wordcloud import WordCloud
# from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn import naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

#Data Import, Preprocessing
spamham_data = pd.read_csv(r"C:\Users\aruna\OneDrive\Desktop\UCDPA_Assignment\spam.csv", encoding="ISO-8859-1")
print(spamham_data.head(5))

# number of rows and columns in this dataset
print(spamham_data.shape)

# print columns names and dataType.There's no null value in this dataset
print(spamham_data.info())

# removed Column that start with U
spamham_data = spamham_data.loc[:, ~spamham_data.columns.str.contains('^U')]
print(spamham_data.head(5))

#Rename Column V1 and V2
spamham_data = spamham_data.rename(
    columns={spamham_data.columns[0]: 'catergories', spamham_data.columns[1]: 'messages'})
print(spamham_data.head(5))

#Checking if Null values presents in the table
print(spamham_data.isnull().sum())

#Total SPAM and HAM
print(spamham_data['catergories'].value_counts())

#Adding new column Length to the table
spamham_data['Length'] = 0
for x in np.arange(0, len(spamham_data.messages)):
    spamham_data.loc[x, 'Length'] = len(spamham_data.loc[x, 'messages'])

print(spamham_data.head(5))

# Checking Ratio of catergories HAM and SAPM
print("Not SPAM email Ratio catergory-HAM :",
      round(len(spamham_data[spamham_data['catergories'] == 'ham']) / len(spamham_data['catergories']), 2) * 100, "%")
print("Spam Email Ratio catergory-SPAM:",
      round(len(spamham_data[spamham_data['catergories'] == 'spam']) / len(spamham_data['catergories']), 2) * 100, "%")

#Bar based on Catergories
plt.figure(figsize=(10,10))
sns.barplot(x=spamham_data['catergories'].value_counts().index,y=spamham_data['catergories'].value_counts(),data=spamham_data)
plt.title("Number of SMS based on Catergories")
plt.show()

#Histogram
spamham_data.hist(column='Length',by='catergories',bins=70,figsize=(15,5));
plt.xlim(-40,950);
plt.show()

#Adding new column no_of_words to the table and converting all the text to lowercase
spamham_data['no_of_words'] = spamham_data['messages'].apply(lambda x: len(nltk.word_tokenize(x)))
spamham_data['messages'] = spamham_data['messages'].apply(lambda x: x.lower())
print(spamham_data.head(5))

#Scatter Graph based on Length and No of Words
fig = px.scatter_matrix(spamham_data, dimensions=["Length",'no_of_words'],
                        color = "catergories",template='gridon',
                        color_discrete_map = {'ham': 'forestgreen', 'spam': 'red'},
                        title = "SPAM and HAM ")
#fig.update_traces(diagonal_visible=False)
fig.show()

#Word counts before removing stopwords
def word_count_plot(spamham, title):
    # finding words along with count
    word_counter = collections.Counter([word for sentence in spamham for word in sentence.split()])
    most_count = word_counter.most_common(30)  # 30 most common words
    # sorted data frame
    most_count = pd.DataFrame(most_count, columns=["Words", "Occurance"]).sort_values(by="Occurance", ascending=False)
    fig = px.bar(most_count, x="Words", y="Occurance", color="Occurance", template='ggplot2', title=title)
    fig.show()

word_count_plot(spamham_data["messages"], "Word Count Before Removing StopWords")

#removing stopwords
def msg_process(msg):
    msg = msg.translate(str.maketrans('', '', string.punctuation))
    msg = [word for word in msg.split() if word.lower() not in stopwords.words('english')]
    return msg

spamham_data['removed_stopwords'] = spamham_data['messages'].apply(lambda row: msg_process(row))
print(spamham_data.head(5))

#Removing extra StopWords
remove_extra_stopwords = ['u', 'im', '2', 'ur', 'ill', '4', 'lor', 'r', 'n', 'da', 'oh', 'dun','lar','den','hor','nah']

spamham_data['removing_extra_stopwords'] = spamham_data['removed_stopwords'].apply(lambda msg: [word for word in msg if word not in remove_extra_stopwords])
print(spamham_data.head(5))

# Joining Cleaning text and adding new Length to the table
def get_final_text(msg):
    final_text=" ".join([word for word in msg])
    return final_text
spamham_data['final_text']=spamham_data['removing_extra_stopwords'].apply(lambda row : get_final_text(row))
spamham_data['clean_length'] =spamham_data.final_text.str.len()
print(spamham_data.head(5))

#Word Count After removing StopWords and Extra StopWords
word_count_plot(spamham_data["final_text"],"Word Count After Removing StopWords")

#Total Words remove after removing StopWords and Extra StopWords
print("Original Length:",spamham_data.Length.sum())
print("Cleaned Length:",spamham_data.clean_length.sum())
print("Total Words Removed:",(spamham_data.Length.sum()) - (spamham_data.clean_length.sum()))

ham_words = list(spamham_data.loc[spamham_data.catergories == 'ham', 'removing_extra_stopwords'])
ham_words = list(np.concatenate(ham_words).flat)
ham_words = Counter(ham_words)
ham_words = pd.DataFrame(ham_words.most_common(30), columns = ['word', 'frequency'])

#WordCloud used in HAM messages
ham_cloud = list(spamham_data.loc[spamham_data.catergories == 'ham', 'final_text'])

wordcloud_ham = WordCloud(width = 500,
                     height = 500,
                     background_color ='white', min_font_size = 9).generate(' '.join(ham_cloud))
plt.figure(figsize = (10, 9),dpi=50, facecolor = None)
plt.imshow(wordcloud_ham)
plt.axis("off")
plt.title('WordCloud for Ham message')
plt.tight_layout(pad = 0)
plt.show()

#Top 25 words in HAM messages
ham_data_top25 = ham_words.sort_values(by='frequency', ascending=False)
ham_data_top25 = ham_data_top25.head(25)

plt.figure(figsize=(15,10))
sns.barplot(x=ham_data_top25['word'], y=ham_data_top25['frequency'])
plt.title("Words in HAM messages")
plt.xticks(rotation=30)
plt.show()


spam_words = list(spamham_data.loc[spamham_data.catergories == 'spam', 'removing_extra_stopwords'])
spam_words = list(np.concatenate(spam_words).flat)
spam_words = Counter(spam_words)
spam_words = pd.DataFrame(spam_words.most_common(30), columns = ['word', 'frequency'])

# Word Cloud in SPAM messages
spam_cloud = list(spamham_data.loc[spamham_data.catergories == 'spam', 'final_text'])

wordcloud_spam = WordCloud(width = 800, height = 800,
                     background_color ='white', min_font_size = 9).generate(' '.join(spam_cloud))
plt.figure(figsize = (10, 9),dpi=55, facecolor = None)
plt.imshow(wordcloud_spam)
plt.axis("off")
plt.title('WordCloud for Spam message')
plt.tight_layout(pad = 0)
plt.show()

#Top25 words used in SPAM messages
spam_data_top25 = spam_words.sort_values(by='frequency', ascending=False)
spam_data_top25 = spam_data_top25.head(25)

plt.figure(figsize=(15,10))
sns.barplot(x=spam_data_top25['word'], y=spam_data_top25['frequency'])
plt.title("Words in SPAM messages")
plt.xticks(rotation=30)
plt.show()

#Replace "ham" to 0 and SPAM to 1
spamham_data = spamham_data.replace(['ham','spam'],[0, 1])
print(spamham_data.head(5))

#ML
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(spamham_data['final_text'])
y=spamham_data['catergories'].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred=mnb.predict(X_test)
print('Accuracy score of Multinomial NB is: ',accuracy_score(y_test,y_pred))
print('Confusion Matrix of Multinomial NB is: ',confusion_matrix(y_test,y_pred))
print('Precision score of the Multinomial NB is',precision_score(y_test,y_pred))

matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, annot = True, cmap='Blues', fmt = 'd')
plt.show()

print("Accuracy of the model : {0:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))

print(classification_report(y_test, y_pred))


def test_classifier(sms):
    transformed = vectorizer.transform([sms])
    prediction = mnb.predict(transformed)

    if prediction == 0:
        return "This message is NOT spam!"
    else:
        return "This message is spam!"


print(test_classifier("mobile 11 months entitled update latest colour..."))
print(test_classifier("How are You?"))
print(test_classifier("free entry wkly comp win fa cup final tkts 21s..."))
print(test_classifier("Good morning Vincent"))
print(test_classifier("Free Entry"))
print(test_classifier("Urgent call this number now!"))