import string
import numpy as np
from nltk.corpus import stopwords
import nltk.corpus as corpus
import itertools as IT
from sematch.semantic.similarity import WordNetSimilarity
import sys
wordnet = corpus.wordnet

class WordNetSimilarity(object):
	"""docstring for WordNetSimilarity"""
	def wnSim(self, list1, list2):

		a = np.array(list1)
		l1 = a.tolist()
		b = np.array(list2)
		l2 = b.tolist()
		ss = []
		for word1, word2 in IT.product(l1, l2):
			wordFromList1 = wordnet.synsets(word1)[0]
			wordFromList2 = wordnet.synsets(word2)[0]
			s = wordFromList1.wup_similarity(wordFromList2)
			if s == None:
				s1 = 0.0
			else:
				s1 = s
			ss.append(s1)
			
		if ss == 0.0:
			print('no matches')
		else:
			best = max(ss)
			print(best)
		return best

if __name__ == '__main__':

	topic = []
	with open ("topic.txt", "r") as tpfile:
		for line in tpfile:
			term = line
			t = term.replace('\n', '')
			topic.append(t)

	ldtopic = []
	with open("lda_topic.txt", "r") as ldatpfile:
		for line in ldatpfile:
			lda_term = line
			lterm = lda_term.replace('\n', '')
			ldtopic.append(lterm)
	sim = WordNetSimilarity()
	sim.wnSim(topic, ldtopic)