from flexcachegen.engine import VLMEngine

def main():
    model_path = '/data/lyc/models/Qwen3-VL-8B-Instruct'
    vlm = VLMEngine(model_path)
    
    samples = [
        ("/data1/lyc/datasets/MLVU/MLVU/video/9_summary/217.mp4", 'Please describe this video in detail.'),
    ]
    outputs = []
    
    for video_path, question in samples:
        output = vlm.generate_single(video_path, question)
        outputs.append(output)

    for (video_path, question), output in zip(samples, outputs):
        print("\n")
        print(f"Video: {video_path}")
        print(f"Question: {question!r}")
        print(f"Output: {output!r}") # use !r to show quotes

if __name__ == "__main__":
    main()