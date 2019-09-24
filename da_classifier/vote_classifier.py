import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
import feature_extractor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
ext = feature_extractor.feature_extractor()
training_set = ext.create_featuresets()

test_set = training_set[200:400]
print(test_set)
training_set=training_set[0:200]
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, test_set))*100)
print(MNB_classifier.classify())
#
#source:
#https://pythonprogramming.net/combine-classifier-algorithms-nltk-tutorial/
#
#

class VoteClassifier(ClassifierI):

    def __init__(self,*classifiers):
        self._classifiers = classifiers
        
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
        
        
        