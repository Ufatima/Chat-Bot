import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import sys
import numpy as np
import re
import ast
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
# from sematch.semantic.similarity import WordNetSimilarity
import sys
import itertools
from itertools import product
from nltk.tokenize import word_tokenize

class lda_modeling(object):
	
	def __init__(self):
		#self.docs = docs
		pass

	def preprocess_sentences(self, doc_complete):
		stop = set(stopwords.words('english'))
		exclude = set(string.punctuation)
		lemma = WordNetLemmatizer()
		
		norm = []
		for doc in doc_complete:
			stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
			punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
			normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
			norm.append(normalized.split())
		return norm

	def generate_topic(self):

		doc_complete = []
		with open ("input.txt", "r") as tpfile:
			for line in tpfile:
				term = line
				doc_complete.append(term)

		texts = self.preprocess_sentences(doc_complete)
		dictionary = corpora.Dictionary(texts)
		doc_term_matrix = [dictionary.doc2bow(text) for text in texts]

		Lda = gensim.models.ldamodel.LdaModel
		ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=300)
		str = ldamodel.print_topics(num_topics=3, num_words=3)
		#print(str)
		#list to tuple
		s = tuple(str)
		t = "\n".join(item[1] for item in s)
		result = re.findall('[a-zA-Z]+',t)
		print('topics from user input', result)

		file = open("lda_topict.txt", "w")
		for i in range(0, len(result)):
			file.write(result[i]+"\n") 
		file.close()
		return result

# if __name__ == '__main__':
# 	doc_complete = []
# 	with open ("..\\user_input.txt", "r") as tpfile:
# 		for line in tpfile:
# 			term = line
# 			doc_complete.append(term)
# 	lda = lda_modeling(*doc_complete)
# 	lda.generate_topic()