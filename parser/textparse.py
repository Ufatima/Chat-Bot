import spacy, nltk
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
nlp = spacy.load('en_core_web_sm')
lemmatizer = spacy.lemmatizer.Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

def sentence_tokenize(raw_text):
    doc = nlp(raw_text)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences

def check_question(sentence):
    if '?' in nltk.word_tokenize(sentence):
        return True
    else:
        return False
    
def parse_statement(raw_text, lemmatize = False):
    doc = nlp(raw_text)
    phrase = []
    subtree_queue = []
    root = [token for token in doc if token.head == token][0]
    subtree_queue.append(root)
    
    while len(subtree_queue) > 0:
        dependencies = {}
        root = subtree_queue.pop(0)
        conjv = None
        
        for descendant in root.subtree: #check for conjunctive phrase
            if ((descendant.dep_ == 'conj') and (descendant.pos_ == 'VERB')) and (descendant != root):
                subtree_queue.append(descendant)
                conjv = descendant
                
        for descendant in root.subtree: #initial assignment
            if (conjv is None) or ((not conjv.is_ancestor(descendant)) and (descendant != conjv)):
                dependencies.setdefault(descendant.dep_, []).append(descendant)
                
        for dep, tokens in dependencies.items(): #extend conjuncts
            for token in tokens:
                if dep != 'ROOT':
                    dependencies[dep].extend(token.conjuncts)
            
           
        if 'conj' in dependencies:
            for token in dependencies['conj']: #re-assign conjunctive verb
                if token.pos_ == 'VERB':
                    conjv = token
                    dependencies['conj'].remove(token)
                    dependencies.setdefault('conjv', []).append(conjv)
                
        for dep, tokens in dependencies.items(): #change tokens to strings
            for index, token in enumerate(tokens):
                if lemmatize == True:
                    text = lemmatizer(token.text, token.pos_)[0]
                else:
                    text = token.text
                dependencies[dep][index] = text
        
        phrase.append(dependencies)
    return phrase