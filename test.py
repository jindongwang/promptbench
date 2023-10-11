import sys
sys.path.append('/home/jindwang/mine/promptbench')  # Replace with the actual path

from models import LLMModel

if __name__ == "__main__":

    print(LLMModel.model_list())
    if __name__ == '__main__':
        model = LLMModel(model='vicuna-7b',
                        model_dir='/home/jindwang/mine/vicuna-7b')()
        print(model('The quick brown fox jumps over the lazy dog'))