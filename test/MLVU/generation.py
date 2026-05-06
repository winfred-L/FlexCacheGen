import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm

from flexcachegen.engine import VLMEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["8_sub_scene", "9_summary"], required=True)
    parser.add_argument("--pred-dir", required=True)

    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--model-type", default="qwen3vl-8b")
    parser.add_argument("--dataset-dir", default="/data1/lyc/datasets/MLVU/MLVU")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # initialize VLM engine
    vlm = VLMEngine(model_type=args.model_type)

    # load dataset json
    json_file_path = os.path.join(args.dataset_dir, f'json/{args.task}.json')
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if args.limit is not None and args.limit > 0:
        dataset = dataset[:args.limit]

    # generation
    results = []
    for sample in tqdm(dataset):
        video_path = os.path.join(args.dataset_dir, f"video/{args.task}/{sample['video']}")
        question = sample['question']
        predict_answer = vlm.generate_single(video_path, question)

        if args.task == "8_sub_scene":
            result = {
                'video_name': sample['video'],
                'duration': sample['duration'],
                'Q': sample['question'],
                'A': sample['answer'],
                'scoring_points': sample['scoring_points'],
                'pred': predict_answer,
            }
        elif args.task == "9_summary":
            result = {
                'video_name': sample['video'],
                'duration': sample['duration'],
                'Q': sample['question'],
                'A': sample['answer'],
                'pred': predict_answer,
            }
        else:
            raise ValueError(f"Unsupported task: {args.task}")
        results.append(result)
    
    # save result to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_type = args.model_type
    dataset_type = f'MLVU_{args.task}'
    os.makedirs(args.pred_dir, exist_ok=True)
    result_file_path = os.path.join(args.pred_dir, f'{model_type}--{dataset_type}--{timestamp}.json')
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print('All generation done!')



if __name__ == "__main__":
    main()