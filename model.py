import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from nltk import classify
from nltk import NaiveBayesClassifier
import pickle
url = r"C:\Users\MSI\Desktop\Mr BenTaleb Ahmed\\train.csv" 
names = ['polarity','Title','description','avis']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,0:8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set

classifier = NaiveBayesClassifier.train()
classifier.classify()
model = NaiveBayesClassifier ()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)