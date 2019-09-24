from nmt_lib import inference
import json
import tensorflow as tf
from nmt_lib.core.tokenizer import apply_bpe,tokenize
class NN_instance(object):
    def __init__(self,model_dir):
        #model dir needs to have nn_files,hparams,vocab
        
        self.hparams = self.load_hparams(model_dir)
        self.model,self.flags,self.hparams = inference.do_start_inference(model_dir,self.hparams)
        
        
        
    def load_hparams(self,model_dir):
        # this is beatifully hard coded function yay
        with open(model_dir+'\\hparams','r') as f:
            hparams = json.load(f)
        hparams['src_vocab_file'] = model_dir +'\\vocab.bpe.from'
        hparams['tgt_vocab_file'] = model_dir +'\\vocab.bpe.from'
        hparams['best_bleu_dir'] = model_dir
        hparams['ckpt'] = model_dir + '\\translate.ckpt'
        popped = []
        for key,value in hparams.items():
            if value == None:
                popped.append(key)
        for key in popped:
            hparams.pop(key)
        with open(model_dir+'\\hparams','w') as f:
            f.write(json.dumps(hparams))
        with open(model_dir+'\\best_bleu\\hparams','w') as f:
            f.write(json.dumps(hparams))
        return hparams

    
    def get_output(self,input_str):
        input = []
        input_str = input_str.strip()
        input.append(apply_bpe(tokenize(input_str)))
        output = inference.do_inference(input,self.model,self.flags,self.hparams)
        return output[0]
        
        