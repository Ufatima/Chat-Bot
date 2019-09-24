import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import random
import swda
import string

class feature_extractor(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable = ['ner', 'textcat'])
        self.lemmatizer = spacy.lemmatizer.Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
        
    def find_features(self, utterance, n = 9, is_previous_speaker = None, previous_da = None):
        features = {}
        doc = self.nlp(utterance)
        lemmatized = [self.lemmatizer(token.text, token.pos_)[0] for token in doc]
        
        features['is_question'] = ('?' in [token.text for token in doc]) #question mark
        
        for i in range(n): #first n tokens
            try:
                features['word'+str(i+1)] = lemmatized[i]
            except IndexError:
                features['word'+str(i+1)] = ''
                
        for i in range(n): #first n pos-tags
            try:
                features['word'+str(i+1)+'_pos_tag'] = doc[i].pos_
            except IndexError:
                features['word'+str(i+1)+'_pos_tag'] = ''
                
        if is_previous_speaker != None: #previous speaker
            features['is_previous_speaker'] = is_previous_speaker
        else:
            features['is_previous_speaker'] = ''
            
        if previous_da: #previous dialogue act
            features['previous_da'] = previous_da
        else:
            features['previous_da'] = ''
            
        try: #predicate verb
            predicate, predicate_pos = [(token.text, token.pos_) for token in doc if token.dep_ == 'ROOT'][0]
            features['predicate'] = self.lemmatizer(predicate, predicate_pos)[0]
        except IndexError:
            features['predicate'] = ''
            
        try: #subject
            subject, subject_pos = [(token.text, token.pos_) for token in doc if (token.dep_ == 'nsubj') or (token.dep_ == 'nsubjpass') or (token.dep_ == 'csubj')][0]
            features['subject'] = self.lemmatizer(subject, subject_pos)[0]
        except IndexError:
            features['subject'] = ''
            
        try: #object
            object, object_pos = [(token.text, token.pos_) for token in doc if (token.dep_ == 'iobj') or (token.dep_ == 'obj') or (token.dep_ == 'dobj') or (token.dep_ == 'pobj')][0]
            features['object'] = self.lemmatizer(object, object_pos)[0]
        except IndexError:
            features['object'] = ''
            
        return features
        
    def create_featuresets(self, max_transcripts = 20, n = 9):
        corpus = swda.CorpusReader('swda')
        utterances = []
        
        i = 1
        
        for trans in corpus.iter_transcripts(display_progress = True):
            if i > max_transcripts:
                break
            previous_tag = None
            previous_caller = None
            for utt in trans.utterances:
                if utt.act_tag not in ('x', 't3', '%', '+'): #discard non-verbal, uninterpretable and third-party talk da:s as well as continued sentences
                    try:
                        previous_tag = utterances[-1][1]
                        previous_caller = utterances[-1][2]
                    except IndexError:
                        pass
                    utterances.append((self.clean_utterance(utt.text), utt.act_tag, utt.caller, previous_tag, previous_caller))
            i += 1
                
        print('\nProcessing {} utterances... this will take some time.'.format(str(len(utterances))))
        random.shuffle(utterances)
        
        featuresets = [(self.find_features(text, n, is_previous_speaker = (caller == previous_caller), previous_da = previous_tag), tag) for (text, tag, caller, previous_tag, previous_caller) in utterances]
        
        return featuresets
        
    def clean_utterance(self, utterance):
        ttable = dict((ord(char), None) for char in string.punctuation)
        for key in '?!.,':
            if ord(key) in ttable: del ttable[ord(key)]
        for key in 'CDEFG':
            ttable[ord(key)] = None
        utterance = (utterance.translate(ttable)).replace('  ', ' ')
        if utterance[0] == ' ':
            utterance = utterance[1:]
        return utterance
