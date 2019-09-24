import nltk
from wnaffect import WNAffect
from emotion import Emotion
# wna = WNAffect('wordnet-1.6/', 'wn-domains-3.2/')

class Emotion_detection(object):
    def __init__(self,wna):
        self.queue = []
        self.wna = wna

    def get_emotion(self, string):
        string = string.strip(',')
        string = string.strip('.')
        string = string.strip(';')
        string = string.strip('?')
        string = string.strip('!')
        self.queue.append(string)
        if len(self.queue) > 5:
            self.queue.pop(0)
        emotions = []
        for sentence in self.queue:
            sentence = sentence.split()
            for word in sentence:
                print(word)
                emo = self.wna.get_emotion(word, 'JJ')
                if emo != None:
                    root = emo.get_level(4)
                    emotions.append(root.__str__())
        return emotions

# e = Emotion_detection()
# print(e.get_emotion('A hungry man is an angry one. You put the fun in together, The sad in apart, The hope in tomorrow, The joy in my heart. Acting is a total physical, emotional sensation.'))