"""
Profiling entry point for nsys decode pipeline analysis.

Usage:
  # Pipeline mode
  nsys profile -t cuda,nvtx --force-overwrite -o profile_pipeline \
      python scripts/profile_decode.py --pipeline --model-path <path> --video-path <path>

  # Non-pipeline mode
  nsys profile -t cuda,nvtx --force-overwrite -o profile_no_pipeline \
      python scripts/profile_decode.py --no-pipeline --model-path <path> --video-path <path>

  # Compare
  nsys stats profile_pipeline.nsys-rep
  nsys stats profile_no_pipeline.nsys-rep

  # Or open .nsys-rep files in Nsight Systems GUI to view timeline
"""
import argparse
from flexcachegen.engine import VLMEngine
from flexcachegen.config import Config


def main():
    parser = argparse.ArgumentParser(description="Profile decode with/without pipeline overlap")
    parser.add_argument("--pipeline", action="store_true", default=False,
                        help="Enable pipeline decode (DMA/compute overlap)")
    parser.add_argument("--no-pipeline", dest="pipeline", action="store_false")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--question", default="Please describe this video in detail.")
    args = parser.parse_args()

    Config.pipeline_decode = args.pipeline
    Config.nsys_nvtx = True

    vlm = VLMEngine(args.model_path)
    output = vlm.generate_single(args.video_path, args.question)
    print(output)


if __name__ == "__main__":
    main()
