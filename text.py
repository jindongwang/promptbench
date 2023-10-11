from models import InferenceModel

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/flan-t5-large")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--generate_len", type=int, default=20)
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()
    inference = InferenceModel(args=args).infer_model
    print(type(inference))
    prompt = 'translate English to German: '
    raw_data = 'I am a student.'
    print(inference.predict(prompt + raw_data))