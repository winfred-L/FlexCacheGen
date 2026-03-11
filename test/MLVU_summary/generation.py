import json
import os
from datetime import datetime
from tqdm import tqdm

from flexcachegen.engine import VLMEngine


def main():
    model_path = '/data/lyc/models/Qwen3-VL-8B-Instruct'
    vlm = VLMEngine(model_path)

    # load dataset json
    dataset_path = '/data1/lyc/datasets/MLVU/MLVU'
    json_file_path = os.path.join(dataset_path, 'json/9_summary.json')
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    dataset = dataset[:3]

    # generation
    results = []
    for sample in tqdm(dataset):
        video_path = os.path.join(dataset_path, f"video/9_summary/{sample['video']}")
        question = sample['question']
        predict_answer = vlm.generate_single(video_path, question)

        result = {
            'video_name': sample['video'],
            'duration': sample['duration'],
            'Q': sample['question'],
            'A': sample['answer'],
            'pred': predict_answer,
        }
        results.append(result)
    
    # save result to file
    output_path = '/data1/lyc/flexcachegen_outputs'
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # model_type = 'qwen3vl-8b'
    # dataset_type = 'MLVU_summary'
    # result_file_path = os.path.join(output_path, f'{model_type}-{dataset_type}-{timestamp}.json')
    result_file_path = os.path.join(output_path, f'pred_summary_all.json')
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print('All generation done!')


if __name__ == "__main__":
    main()


'''
dataset json:

{'video': '217.mp4', 'duration': 480.0, 'question': 'Please summarize this video, including its main content.', 'answer': 'The video starts with waves lapping against the rocks, creating a spray. Then, a boat appears with two men on board, one with a hat and the other without. The man without a hat holds a camera, seemingly focusing on two whales. The hatless man changes into a diving suit and dives underwater for a closer shot of the whales. The video then uses animation techniques to help us understand more about the whales. The video switches between the characters and the whales, but primarily describes the human activity of filming the whales.', 'question_type': 'summary'}
'''