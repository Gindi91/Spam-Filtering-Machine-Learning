import pandas
from sklearn.cross_validation import train_test_split
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def process_dataset(filename):
    dataset=pandas.read_csv(filename, delimiter=",")
    return dataset

def get_matrix(dataset):
    return dataset.values

def get_emails(dataset):
    return dataset['text'].values

def get_labels(dataset):
    return dataset['spam'].values

def preprocess(dataset):
    cv=CountVectorizer()
    vectorized=(cv.fit_transform(dataset))#.toarray()
    tf_transformer=TfidfTransformer(use_idf=False).fit(vectorized)
    tf=(tf_transformer.transform(vectorized)).toarray()
    return tf
   

filename=#path_to_emails.csv
dataset=process_dataset(filename)
preprocessed=preprocess(get_emails(dataset))

X=preprocessed
y=get_labels(dataset)

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.8, random_state=17)
y_expect=y_test
#=======================================================================#
#Naive Bayes Classifier

#Bernoulli
bnb=BernoulliNB(binarize=True)
bnb.fit(X_train,y_train)

print bnb

y_pred=bnb.predict(X_test)
print accuracy_score(y_expect,y_pred)

average_precision = average_precision_score(y_expect,y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_expect,y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Bernoulli Bayes Classifier Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

#Gaussian
gnb=GaussianNB()
gnb.fit(X_train,y_train)

print gnb
y_pred=gnb.predict(X_test)
print accuracy_score(y_expect, y_pred)

average_precision = average_precision_score(y_expect, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_expect, y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Gaussian Bayes Classifier Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

#Multinomial
mnb=MultinomialNB()
mnb.fit(X_train,y_train)

print mnb
y_pred=mnb.predict(X_test)
print accuracy_score(y_expect, y_pred)

average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Multinomial Bayes Classifier Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
#=======================================================================#
#Support vector machine

#SVC
svc=svm.SVC()
svc.fit(X_train,y_train)

print svc
y_pred=svc.predict(X_test)
print accuracy_score(y_expect, y_pred)

y_score=svc.decision_function(X_test)

average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('SVC Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

#LinearSVC
lsvc=svm.LinearSVC()
lsvc.fit(X_train,y_train)

print lsvc
y_pred=lsvc.predict(X_test)
print accuracy_score(y_expect, y_pred)

y_score=lsvc.decision_function(X_test)

average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Linear SVC Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
#=======================================================================#
#Percettrone multistrato (rete neurale)

mplc=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mplc.fit(X_train,y_train)

print mplc
y_pred=mplc.predict(X_test)
print accuracy_score(y_expect, y_pred)

average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Neural Network Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))