# import sys
# sys.path.append('/home/jindwang/mine/promptbench')  # Replace with the actual path

from models import LLMModel

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vicuna-7b')
    parser.add_argument('--model_dir', type=str,
                        default='/home/jindwang/mine/vicuna-7b')
    args = parser.parse_args()
    model = LLMModel(model=args.model,
                     model_dir=args.model_dir)()
    print(model('The quick brown fox jumps over the lazy dog'))