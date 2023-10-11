from base_model import LLMBaseModel

class T5Model(LLMBaseModel):

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
                input_ids, max_length=self.gen_len, early_stopping=True)
        out = self.tokenizer.decode(outputs[0])
        return out
