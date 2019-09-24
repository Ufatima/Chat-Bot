from sys import path
path.extend(['da_classifier','WNAffect','Personality_recognizer','Natural_language_generation','parser'])
import pickle,os,time,datetime
from operator import itemgetter
import subprocess
import nltk
import os
import re


from WNAffect.wnaffect import WNAffect
from WNAffect.emotion import Emotion  # if needed
from WNAffect.runner import Emotion_detection
wna = WNAffect('WNAffect/wordnet-1.5/', 'WNAffect/wn-domains-3.2/')

from feature_extractor import feature_extractor
from sklearn.linear_model import LogisticRegression
from textparse import parse_statement

from Lda.lda_topic import lda_modeling
# from Lda.wn_similarity import WordNetSimilarity

from NN_instance import NN_instance



class ChatInstance(object):
    def __init__(self,nn_model_dir):
        self.counter = 0
    
        #dialog act classifier
        with open('da_classifier\\logisticregression.pickle', 'rb') as f:
            self.da_classifier = pickle.load(f)
        self.fex = feature_extractor()
        #Dialog act matching, for ranking NN outputs. Dict key matches input DA, value is a tuple of DA's that make sense as output.
        #Check https://web.stanford.edu/~jurafsky/ws97/manual.august1.html for explanations of tags
        with open('da_matches.pickle','rb') as f:
            self.da_matches = pickle.load(f)
        
        
        #emotion detection
        self.emotion = Emotion_detection(wna)
        
        #topic detection/LDA
        self.lda = lda_modeling()
        #FIX THIS

        
        #NN-model
        self.NN = NN_instance(nn_model_dir)

        #logging and 20 slot queue of previous conversation (list of dictionaries {'actor', 'utterance', 'emotions', 'personality', 'da', 'topic'})
        #do different emotions and personality need to be marked separately?
        self.prev_conversation = [{'actor': 'user', 'utterance': None,
                                    'emotions': None, 'personality': [0,0,0,0,0],
                                    'da': None, 'topic': None}]
        self.log_out = 'LOG_' + time.strftime('%H_%M_%S') +'.txt'
        with open(self.log_out,'w') as f:
            f.write('time,actor,utterance,da,emotion,personality,topic\n')

        
    def get_emotion(self,utterance):
        emotion_result=self.emotion.get_emotion(utterance)
        print(emotion_result)
        negative = emotion_result.count("negative-emotion")
        positive = emotion_result.count('positive-emotion')
        emotions = "not defined"
        print('positive', positive)
        print('negative', negative)
        if(positive > negative):
            emotions = "positive emotion"
           
        if (negative > positive):
            emotions = "negative emotion"
        return emotions
        return None
        
    def get_personality(self, utterance):
        """Get personality traits"""
        outputfile = open('Personality_recognizer/PersonalityRecognizer/input.txt', 'w')
        outputfile.write(utterance)
        outputfile.close()

        wd=os.getcwd()
        os.chdir("Personality_recognizer/PersonalityRecognizer")
        with open('input.txt','w') as f:
            f.write(utterance)
        subprocess.call('PersonalityRecognizer -i input.txt -t 2 -m 4 > output.txt', shell=True)
        with open('output.txt','r') as f:
            personality = f.readline()
        os.chdir(wd)

        #Read output from output file
        personality = []
        inputfile = 'Personality_recognizer/PersonalityRecognizer/output.txt'
        with open(inputfile, 'r') as f:
            temp = f.readlines()[7:]
        for lines in temp:
            numbers_float = re.findall('\d+\.\d+', lines)
            if(numbers_float != []):
               personality.append(numbers_float)
        while len(personality) < 5:
            personality.append(None)
        return personality
        
    def get_topic(self, input_str):
        with open("input.txt", "a") as file:
            file.write(input_str+"\n")
        topic = self.lda.generate_topic()
        return topic
        
    def get_NN_output(self, utterance):
        """Get outputs from the NN """
        output = self.NN.get_output(utterance)
        return output
        
    def select_best(self, outputs):
        """Rank possible outputs and return the one with highest score"""
        scored_outputs = []
        
        #Get input sentence
        input = self.prev_conversation[-1]
        input_utt = input['utterance']
        parsed_input_utt = parse_statement(input_utt, lemmatize = True)
        objs_input, subjs_input, verbs_input = self.select_best_helper(parsed_input_utt)
        prev_da = input['da']
        
        for utt in outputs:
            score = 0
            
            #check matching da's
            da = self.get_da(utt, 'bot')
            if da in self.da_matches[prev_da]:
                score += 10
            else:
                score -= 10
                
            #check if predicate/object/subject match in input and output
            parsed_output = parse_statement(utt, lemmatize = True)
            objs_output, subjs_output, verbs_output = self.select_best_helper(parsed_output)
            for obj_output in objs_output:
                for obj_input in objs_input:
                    if obj_output == obj_input: score += 1
            for subj_output in subjs_output:
                for subj_input in subjs_input:
                    if subj_output == subj_input: score += 1
            for verb_output in verbs_output:
                for verb_input in verbs_input:
                    if verb_output == verb_input: score += 1

            scored_outputs.append((utt, score))
            
        #sort by score
        scored_outputs.sort(key = itemgetter(1), reverse = True)
        return scored_outputs[0][0]
        
    def select_best_helper(self, parsed):
        objs, subjs, verbs = [], [], []
        for subsentence in parsed:
            for objtype in ['iobj', 'dobj', 'obj', 'pobj']:
                try:
                    objs.extend(subsentence[objtype])
                except KeyError:
                    pass
            for subjtype in ['nsubj', 'nsubjpass', 'csubj']:
                try:
                    subjs.extend(subsentence[subjtype])
                except KeyError:
                    pass
            for verbtype in ['ROOT', 'conjv']:
                try:
                    verbs.extend(subsentence[verbtype])
                except KeyError:
                    pass
        return objs, subjs, verbs
        
        
    def get_da(self, utterance, actor):
        """Get the dialogue act from an utterance"""
        try:
            prev_da = self.prev_conversation[-1]['da']
            prev_actor = self.prev_conversation[-1]['actor']
        except IndexError:
            prev_da = None
            prev_actor = None
            
        if (prev_da and prev_actor):
            if prev_actor == actor:
                da_features = self.fex.find_features(utterance, n = 10, is_previous_speaker = True, previous_da = prev_da)
            else:
                da_features = self.fex.find_features(utterance, n = 10, is_previous_speaker = False, previous_da = prev_da)
        else:
            da_features = self.fex.find_features(utterance, n = 10)
            
        da = self.da_classifier.classify(da_features)
        return da
        
        
    def get_features(self,input_str):
        """CALL EACH FEATURE METHOD HERE AND ASSIGN TO SELF OR SOMEWHERE"""
        da = self.get_da(input_str, 'user')
        emotions = self.get_emotion(input_str)
        topic = self.get_topic(input_str)
        new_personality = self.get_personality(input_str)
        try:
            personality = self.prev_conversation[-2]['personality']
        except IndexError:
            personality = [0,0,0,0,0]
        for i, value in enumerate(personality):
            personality[i] = (value*self.counter + float(new_personality[i][0])) / (self.counter + 1)
                
        
        #Add to queue, 'actor' = 'user'
        if len(self.prev_conversation) == 20: self.prev_conversation.pop(0)
        self.prev_conversation.append({'actor': 'user', 'utterance': input_str, 'emotions': emotions, 'personality': personality, 'da': da, 'topic': topic})
        
        
    def converse(self,input_str):
        """ Logs the conversation and responds to the user """
        """ ADD FEATURES TO HERE AS WELL!!!"""
        
        self.get_features(input_str)
        self.counter += 1
        with open(self.log_out,'a') as f:
            #Log input
            entry = '{},{},{},{},{},{},{}\n'.format(
                time.strftime('%H_%M_%S'),
                self.prev_conversation[-1]['actor'],
                self.prev_conversation[-1]['utterance'],
                self.prev_conversation[-1]['da'],
                self.prev_conversation[-1]['emotions'],
                '/'.join('{}'.format(e) for e in self.prev_conversation[-1]['personality']), #format instead of str() to handle Nonetype
                self.prev_conversation[-1]['topic'])
                
            f.write(entry)
            
            #Retrieve user input features from prev_conversation[-1] to pass to interface
            da = self.prev_conversation[-1]['da']
            emotions = self.prev_conversation[-1]['emotions']
            personality = self.prev_conversation[-1]['personality']
            topic = self.prev_conversation[-1]['topic']
            
            #Get NN outputs
            nn_outputs = self.get_NN_output(input_str)
            
            #Select best ones
            output_str = self.select_best(nn_outputs)
            
            #Add to queue, 'actor' = 'bot'. If the emotions, personality or topic from the bot are useful to someone can add them here.
            bot_da = self.get_da(output_str, 'bot')
            if len(self.prev_conversation) == 20: self.prev_conversation.pop(0)
            self.prev_conversation.append({'actor': 'bot', 'utterance': output_str, 'emotions': None, 'personality': [None,None,None,None,None], 'da': bot_da, 'topic': None})
            
            #Log output
            entry = '{},{},{},{},{},{},{}\n'.format(
                time.strftime('%H_%M_%S'),
                self.prev_conversation[-1]['actor'],
                self.prev_conversation[-1]['utterance'],
                self.prev_conversation[-1]['da'],
                self.prev_conversation[-1]['emotions'],
                '/'.join('{}'.format(e) for e in self.prev_conversation[-1]['personality']), #format instead of str() to handle Nonetype
                self.prev_conversation[-1]['topic'])
                
            f.write(entry)
        
        return output_str

