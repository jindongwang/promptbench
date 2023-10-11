import sys
sys.path.append('/home/jindwang/mine/promptbench/models')  # Replace with the actual path

from promptbench.models import LLMModel

if __name__ == "__main__":
    x = LLMModel(model_name="gpt2")
    print(x.model_list())