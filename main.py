from flexcachegen.engine import VLMEngine

def main():
    model_path = '/data/lyc/models/Qwen2.5-VL-7B-Instruct'
    # model_path = '/data/lyc/models/Qwen3-VL-8B-Instruct'
    vlm = VLMEngine(model_path)
    
    samples = [
        ("./test/video/28s.mp4", 'Please describe this video in detail.'),
        # ("./test/video/1_first10s.mp4", 'Please describe this video in detail.'),
        # ("/data1/lyc/datasets/MLVU/MLVU/video/9_summary/217.mp4", 'Please describe this video in detail.'),
    ]
    outputs = []
    
    for video_path, question in samples:
        output = vlm.generate_single(video_path, question)
        outputs.append(output)

    for (video_path, question), output in zip(samples, outputs):
        print("="*20 + "input" + "="*20)
        print(f"Video: {video_path}")
        print(f"Question: {question}")
        print("="*20 + "output" + "="*20)
        print(output)

if __name__ == "__main__":
    main()