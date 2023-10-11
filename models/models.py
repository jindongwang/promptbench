# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


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
        
    def create_model(self, **kwargs):
        if self.model == 'google/flan-t5-large':
            return T5Model(**kwargs)
        return None

    def __call__(self, **kwargs):
        return self.create_model(**kwargs)


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



if __name__ == '__main__':
    model = LMMModel(model='google/flan-t5-large', gen_len=5)()
    print(type(model))
    print(model.predict('The quick brown fox jumps over the lazy dog'))