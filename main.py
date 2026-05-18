from flexcachegen.engine import VLMEngine
import os

DATASET_ROOT = os.environ.get('DATASET_ROOT', '/data/lyc/datasets')

def main():
    vlm = VLMEngine(model_type='qwen3vl-8b')
    
    samples = [
        ("./test/video/28s.mp4", 'Please describe this video in detail.'),
        # (f"{DATASET_ROOT}/MLVU/MLVU/video/9_summary/217.mp4", 'Please describe this video in detail.'),
    ]
    outputs = []


    # info
    for video_path, question in samples:
        output = vlm.generate_single_info(video_path, question)
        outputs.append(output)

    
    # # no info
    # for video_path, question in samples:
    #     output = vlm.generate_single(video_path, question)
    #     outputs.append(output)

    # for (video_path, question), output in zip(samples, outputs):
    #     print(f"{' '+'Input' + ' ':=^50}")
    #     print(f"Video: {video_path}")
    #     print(f"Question: {question}")
    #     print(f"{' '+'Output' + ' ':=^50}")
    #     print(output)


if __name__ == "__main__":
    main()