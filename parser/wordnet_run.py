from nltk.corpus import wordnet as wn 
from nltk.corpus import wordnet_ic
from textparse import sentence_tokenize
from textparse import parse_statement
from textparse import check_question
import sys

print(parse_statement(sys.argv[1]))
print(check_question(sys.argv[1]))

input=sentence_tokenize(sys.argv[1])
mylist=input[0].split(" ")

print(mylist)
for element in mylist:
 element = wn.synset(element+'.n.01')
 print (element.definition())
 print (element.lemmas())
 print (element.lemma_names('ita'))
