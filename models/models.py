# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

class LMMBaseModel(object):
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', 'google/flan-t5-large')
        self.gen_len = kwargs.get('gen_len', 5)

    def predict(self, **kwargs):
        raise NotImplementedError("The model is not implemented!")

    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)

    

class LMMModel(object):

    def __init__(self, **kwargs):
        self.model = kwargs.get('model', None)
        self.infer_model = self.create_model(**kwargs)
        
    def create_model(self, **kwargs):
        if self.model == 'google/flan-t5-large':
            return T5Model(**kwargs)
        elif self.model == 'llama2-7b':
            return LlamaModel(**kwargs)
        return None

    def __call__(self, **kwargs):
        return self.infer_model


class T5Model(LMMBaseModel):

    def __init__(self, **kwargs):
        super(T5Model, self).__init__(**kwargs)
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model, device_map="auto")
        self.pipe = T5ForConditionalGeneration.from_pretrained(
            self.model, device_map="auto")
        
    def predict(self, input_text):
        input_ids = self.tokenizer(
            input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = self.pipe.generate(
                input_ids, max_length=self.gen_len)
        out = self.tokenizer.decode(outputs[0])
        return out


class LlamaModel(LMMBaseModel):
    
    def __init__(self, **kwargs):
        super(LlamaModel, self).__init__(**kwargs)
        model_dir = kwargs.get('model_dir', None)
        if not model_dir:
            raise ValueError("model_dir is required for llama model!")
        
        from transformers import LlamaForCausalLM, LlamaTokenizer

        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_dir, device_map="auto")
        self.pipe = LlamaForCausalLM.from_pretrained(
            model_dir, device_map="auto")
        
    def predict(self, input_text):
        input_ids = self.tokenizer(
            input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = self.pipe.generate(
                input_ids, max_length=self.gen_len)
        out = self.tokenizer.decode(outputs[0])
        return out


if __name__ == '__main__':
    model = LMMModel(model='llama2-7b', gen_len=5, model_dir='home/jindwang/mine/llama2-70b')()
    print(type(model))
    print()
    print(model.predict('The quick brown fox jumps over the lazy dog'))