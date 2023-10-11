class LMMBaseModel(object):
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', None)
        self.gen_len = kwargs.get('gen_len', None)

    def predict(self, **kwargs):
        return NotImplementedError("The model is not implemented!")

    def __call__(self, **kwargs):
        return self.predict(**kwargs)