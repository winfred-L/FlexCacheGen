'''
Unified MLVU generation evaluation script.

Features:
- Select task: 8_sub_scene / 9_summary
- Select metrics: gpt / bert / rouge / all
- GPT scoring with OpenAI-compatible API
- BERTScore with local roberta-large model
  - If /data1/lyc/models/roberta-large is missing or incomplete, download it from ModelScope automatically
- ROUGE scoring
'''

import argparse
import ast
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm


# GPT client is initialized lazily only when GPT metric is requested.
client = None
eval_model_type = None


DEFAULT_BERT_MODEL_ID = "AI-ModelScope/roberta-large"
DEFAULT_BERT_LOCAL_MODEL_PATH = "/data1/lyc/models/roberta-large"


# -----------------------------
# Common helpers
# -----------------------------

def normalize_metrics(metrics: Sequence[str]) -> List[str]:
    """Expand `all` and remove duplicates while preserving order."""
    if "all" in metrics:
        return ["gpt", "bert", "rouge"]

    normalized = []
    for metric in metrics:
        if metric not in normalized:
            normalized.append(metric)
    return normalized


def safe_average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def parse_score_json(output: str) -> Dict[str, Any]:
    """Parse model output as JSON first, then as Python literal as fallback."""
    try:
        return json.loads(output)
    except Exception:
        return ast.literal_eval(output)


def assert_pred_samples(pred_result: Any) -> None:
    if not isinstance(pred_result, list):
        raise ValueError("Prediction JSON must be a list of samples.")
    if len(pred_result) == 0:
        raise ValueError("Prediction JSON is empty.")


# -----------------------------
# GPT evaluation
# -----------------------------

def build_client(args: argparse.Namespace) -> None:
    """Initialize global OpenAI-compatible client for GPT scoring."""
    global client, eval_model_type

    from openai import OpenAI

    eval_model_type = args.eval_model_type
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"{args.api_key_env} not set in environment")

    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
    )


def annotate_gpt_sub_scene(
    question: str,
    correct_answer: str,
    scoring_points: Any,
    pred: str,
    max_retries: int = 1,
) -> Dict[str, Any]:
    if client is None or eval_model_type is None:
        raise RuntimeError("GPT client is not initialized. Call build_client(args) first.")

    messages = [
        {
            "role": "system",
            "content":
            """
                ##TASK DESCRIPTION:
                You are required to evaluate a respondent's answer based on a provided question, some scoring points, and the respondent's answer. You should provide two scores. The first is the accuracy score, which should range from 1 to 5. The second is the relevance score, which should also range from 1 to 5. Below are the criteria for each scoring category.
                ##ACCURACY Scoring Criteria:
                Evaluate the respondent's answer against specific scoring points as follows:
                Score 1: The response completely misses the scoring point.
                Score 3: The response mentions content related to the scoring point but is not entirely correct.
                Score 5: The response accurately addresses the scoring point.
                Calculate the average score across all scoring points to determine the final accuracy score.
                ##RELEVANCE Scoring Criteria:
                Assess how the respondent's answer relates to the original question:
                Score 1: The response is completely off-topic from the question.
                Score 2: The response is partially related to the question but contains a significant amount of irrelevant content.
                Score 3: The response primarily addresses the question, but the respondent seems uncertain about their own answer.
                Score 4: The response mostly addresses the question and the respondent appears confident in their answer.
                Score 5: The response is fully focused on addressing the question with no irrelevant content and demonstrates complete certainty.
                ----
                ##INSTRUCTION:
                1. Evaluate Accuracy: First, assess and score each scoring point based on the respondent's answer. Calculate the average of these scores to establish the final accuracy score. Provide a detailed rationale before assigning your score.
                2. Evaluate RELEVANCE: Assess the relevance of the respondent’s answer to the question. Note that when evaluating relevance, the correctness of the answer is not considered; focus solely on how relevant the answer is to the question. Provide a comprehensive rationale before assigning your score.
                3. Output Scores in JSON Format: Present the scores in JSON format as follows:
                {'score_accuracy': score_acc, 'score_relevance': score_rele, 'total_score': score_acc + score_rele}
            """,
        },
        {
            "role": "user",
            "content": f"""
                Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
                Question: {question}
                Standard Answer: {correct_answer}
                Scoring Points: {scoring_points}
                Respondent's Answer: {pred}
            """,
        },
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=eval_model_type,
                messages=messages,
                timeout=100,
                temperature=0,
                response_format={"type": "json_object"},
            )
            output = response.choices[0].message.content
            return parse_score_json(output)
        except Exception as e:
            print(f"Failed: {question}")
            print(f"Error: {e}")
            if attempt == max_retries - 1:
                return {"score_accuracy": 0, "score_relevance": 0, "total_score": 0}

    return {"score_accuracy": 0, "score_relevance": 0, "total_score": 0}


def annotate_gpt_summary(correct_answer: str, predict_answer: str, max_retries: int = 1) -> Dict[str, Any]:
    if client is None or eval_model_type is None:
        raise RuntimeError("GPT client is not initialized. Call build_client(args) first.")

    messages = [
        {
            "role": "system",
            "content":
            """
                ##TASK DESCRIPTION:
                You are required to evaluate the performance of the respondent in the video summarization task based on the standard answer and the respondent's answer. You should provide two scores. The first is the COMPLETENESS score, which should range from 1 to 5. The second is the RELIABILITY score, which should also range from 1 to 5. Below are the criteria for each scoring category:
                ##COMPLETENESS Scoring Criteria:
                The completeness score focuses on whether the summary covers all key points and main information from the video.
                Score 1: The summary hardly covers any of the main content or key points of the video.
                Score 2: The summary covers some of the main content and key points but misses many.
                Score 3: The summary covers most of the main content and key points.
                Score 4: The summary is very comprehensive, covering most to nearly all of the main content and key points.
                Score 5: The summary completely covers all the main content and key points of the video.
                ##RELIABILITY Scoring Criteria:
                The reliability score evaluates the correctness and clarity of the video summary. It checks for factual errors, misleading statements, and contradictions with the video content. If the respondent's answer includes details that are not present in the standard answer, as long as these details do not conflict with the correct answer and are reasonable, points should not be deducted.
                Score 1: Contains multiple factual errors and contradictions; presentation is confusing.
                Score 2: Includes several errors and some contradictions; needs clearer presentation.
                Score 3: Generally accurate with minor errors; minimal contradictions; reasonably clear presentation.
                Score 4: Very accurate with negligible inaccuracies; no contradictions; clear and fluent presentation.
                Score 5: Completely accurate with no errors or contradictions; presentation is clear and easy to understand.
                ----
                ##INSTRUCTION:
                1. Evaluate COMPLETENESS: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence.
                2. Evaluate RELIABILITY: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence.
                3. Output Scores in JSON Format: Present the scores in JSON format as follows:
                {'score_completeness': score_comp, 'score_reliability': score_reli, 'total_score': score_comp + score_reli}
            """,
        },
        {
            "role": "user",
            "content": f"""
                Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
                Standard Answer: {correct_answer}
                Respondent's Answer: {predict_answer}
            """,
        },
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=eval_model_type,
                messages=messages,
                timeout=100,
                temperature=0,
                response_format={"type": "json_object"},
            )
            output = response.choices[0].message.content
            return parse_score_json(output)
        except Exception as e:
            print(f"Failed: {correct_answer}")
            print(f"Error: {e}")
            if attempt == max_retries - 1:
                return {"score_completeness": 0, "score_reliability": 0, "total_score": 0}

    return {"score_completeness": 0, "score_reliability": 0, "total_score": 0}



def evaluate_gpt(pred_result: List[Dict[str, Any]], task: str, max_retries: int) -> List[Dict[str, Any]]:
    score_list = []
    score_1 = []
    score_2 = []
    total_score = []

    for sample in tqdm(pred_result, desc=f"gpt-{task}"):
        if task == "8_sub_scene":
            score = annotate_gpt_sub_scene(
                sample["Q"],
                sample["A"],
                sample["scoring_points"],
                sample["pred"],
                max_retries=max_retries,
            )
            score_1.append(float(score.get("score_accuracy", 0)))
            score_2.append(float(score.get("score_relevance", 0)))
        elif task == "9_summary":
            score = annotate_gpt_summary(
                sample["A"],
                sample["pred"],
                max_retries=max_retries
            )
            score_1.append(float(score.get("score_completeness", 0)))
            score_2.append(float(score.get("score_reliability", 0)))
        else:
            raise ValueError(f"Unsupported task: {task}")

        total_score.append(float(score.get("total_score", 0)))
        score_list.append(score)

    if task == "8_sub_scene":
        print(
            f"accuracy: {safe_average(score_1):.2f}, "
            f"relevance: {safe_average(score_2):.2f}, "
            f"total: {safe_average(total_score):.2f}"
        )
    elif task == "9_summary":
        print(
            f"completeness: {safe_average(score_1):.2f}, "
            f"reliability: {safe_average(score_2):.2f}, "
            f"total: {safe_average(total_score):.2f}"
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    return score_list


# -----------------------------
# BERTScore evaluation
# -----------------------------

def is_transformers_model_dir(model_dir: str) -> bool:
    """
    Check whether a local Transformers model directory is actually loadable.

    This is more reliable than checking a fixed list of filenames because
    different mirrors may provide different tokenizer config files.
    """
    if not model_dir:
        return False

    model_path = Path(os.path.abspath(os.path.expanduser(model_dir)))

    if not model_path.exists() or not model_path.is_dir():
        return False

    try:
        from transformers import AutoTokenizer, AutoModel

        AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=True,
        )

        AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        return True

    except Exception as e:
        print(f"Local Transformers model validation failed for {model_path}: {e}")
        return False


def ensure_bert_model_from_modelscope(
    local_model_path: str = DEFAULT_BERT_LOCAL_MODEL_PATH,
    model_id: str = DEFAULT_BERT_MODEL_ID,
) -> str:
    """Ensure roberta-large exists locally. Download from ModelScope if missing/incomplete."""
    local_model_path = os.path.abspath(os.path.expanduser(local_model_path))
    target_dir = Path(local_model_path)

    # 1. Prefer existing local target path.
    if is_transformers_model_dir(str(target_dir)):
        print(f"Using local BERTScore model: {target_dir}")
        return str(target_dir)

    print(f"Local BERTScore model is missing or incomplete: {target_dir}")
    print(f"Downloading from ModelScope: {model_id}")

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as e:
        raise ImportError(
            "ModelScope is required for automatic model download. Install it with:\n"
            "  pip install modelscope\n"
            f"Original import error: {e}"
        ) from e

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # 2. Download to ModelScope cache.
    downloaded_dir = Path(snapshot_download(model_id)).resolve()
    print(f"ModelScope downloaded directory: {downloaded_dir}")

    # 3. If ModelScope returned a directly loadable directory, copy it to target.
    if is_transformers_model_dir(str(downloaded_dir)):
        if downloaded_dir != target_dir.resolve():
            if target_dir.exists() and not is_transformers_model_dir(str(target_dir)):
                shutil.rmtree(target_dir)

            shutil.copytree(
                downloaded_dir,
                target_dir,
                dirs_exist_ok=True,
            )

        if is_transformers_model_dir(str(target_dir)):
            print(f"BERTScore model is ready: {target_dir}")
            return str(target_dir)

    # 4. Some ModelScope versions may return a cache root, not the real model root.
    #    Search likely subdirectories and use the first loadable one.
    candidate_dirs = [p for p in downloaded_dir.rglob("*") if p.is_dir()]
    for candidate_dir in candidate_dirs:
        if is_transformers_model_dir(str(candidate_dir)):
            print(f"Found valid Transformers model directory: {candidate_dir}")

            if target_dir.exists() and not is_transformers_model_dir(str(target_dir)):
                shutil.rmtree(target_dir)

            if candidate_dir.resolve() != target_dir.resolve():
                shutil.copytree(
                    candidate_dir,
                    target_dir,
                    dirs_exist_ok=True,
                )

            if is_transformers_model_dir(str(target_dir)):
                print(f"BERTScore model is ready: {target_dir}")
                return str(target_dir)

    # 5. Final diagnostics.
    target_files = []
    downloaded_files = []

    if target_dir.exists():
        target_files = sorted([p.name for p in target_dir.iterdir()])

    if downloaded_dir.exists():
        downloaded_files = sorted([p.name for p in downloaded_dir.iterdir()])

    raise RuntimeError(
        "Failed to prepare local BERTScore model.\n"
        f"Target directory: {target_dir}\n"
        f"Target files: {target_files}\n"
        f"ModelScope directory: {downloaded_dir}\n"
        f"ModelScope files: {downloaded_files}\n"
        "The directory must be loadable by:\n"
        "  AutoTokenizer.from_pretrained(path, local_files_only=True)\n"
        "  AutoModel.from_pretrained(path, local_files_only=True)\n"
    )


def annotate_bert(
    refs,
    preds,
    local_model_path: str = DEFAULT_BERT_LOCAL_MODEL_PATH,
    model_id: str = DEFAULT_BERT_MODEL_ID,
    num_layers: int = 17,
):
    from bert_score import score

    model_path = ensure_bert_model_from_modelscope(
        local_model_path=local_model_path,
        model_id=model_id,
    )

    _, _, F1 = score(
        preds,
        refs,
        model_type=model_path,
        num_layers=num_layers,
        verbose=False,
        rescale_with_baseline=False,
    )

    return F1.tolist(), F1.mean().item()


def evaluate_bert(
    pred_result,
    local_model_path: str = DEFAULT_BERT_LOCAL_MODEL_PATH,
    model_id: str = DEFAULT_BERT_MODEL_ID,
    num_layers: int = 17,
):
    refs_list = [sample['A'] for sample in pred_result]
    preds_list = [sample['pred'] for sample in pred_result]

    f1_list, f1_mean = annotate_bert(
        refs_list,
        preds_list,
        local_model_path=local_model_path,
        model_id=model_id,
        num_layers=num_layers,
    )

    print(f"bert_score : {f1_mean:.3f}")

    return {
        "bertscore_f1_list": f1_list,
        "bertscore_f1_mean": f1_mean,
        "bertscore_model_path": local_model_path,
        "bertscore_num_layers": num_layers,
    }


# -----------------------------
# ROUGE evaluation
# -----------------------------

def annotate_rouge(reference: str, prediction: str) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f1": float(scores["rouge1"].fmeasure),
        "rouge2_f1": float(scores["rouge2"].fmeasure),
        "rougeL_f1": float(scores["rougeL"].fmeasure),
    }


def evaluate_rouge(pred_result: List[Dict[str, Any]]) -> Dict[str, Any]:
    rouge1_f1_list = []
    rouge2_f1_list = []
    rougeL_f1_list = []
    score_list = []

    for sample in tqdm(pred_result, desc="rouge"):
        score = annotate_rouge(sample["A"], sample["pred"])
        rouge1_f1_list.append(score["rouge1_f1"])
        rouge2_f1_list.append(score["rouge2_f1"])
        rougeL_f1_list.append(score["rougeL_f1"])
        score_list.append(score)

    rouge_mean = {
        "rouge1_f1_mean": safe_average(rouge1_f1_list),
        "rouge2_f1_mean": safe_average(rouge2_f1_list),
        "rougeL_f1_mean": safe_average(rougeL_f1_list),
    }

    print(
        f"rouge1 : {rouge_mean['rouge1_f1_mean']:.2f}, "
        f"rouge2 : {rouge_mean['rouge2_f1_mean']:.2f}, "
        f"rougeL: {rouge_mean['rougeL_f1_mean']:.2f}"
    )

    return {
        "scores": score_list,
        "mean": rouge_mean,
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["8_sub_scene", "9_summary"], required=True)
    parser.add_argument("--metrics", nargs="+", choices=["gpt", "bert", "rouge", "all"], default=["all"])
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--pred-file-names", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)

    # GPT options
    parser.add_argument("--base-url", default="https://api3.wlai.vip/v1") #https://yunwu.ai/v1
    parser.add_argument("--eval-model-type", default="gpt-5-mini")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--max-retries", type=int, default=1)

    # BERTScore / ModelScope options
    parser.add_argument("--bert-local-model-path", default=DEFAULT_BERT_LOCAL_MODEL_PATH)
    parser.add_argument("--bert-model-id", default=DEFAULT_BERT_MODEL_ID)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = normalize_metrics(args.metrics)
    os.makedirs(args.output_dir, exist_ok=True)

    if "gpt" in metrics:
        build_client(args)

    for pred_file_name in args.pred_file_names:
        pred_file_path = os.path.join(args.pred_dir, f"{pred_file_name}.json")
        print(f"\nEvaluating prediction file: {pred_file_path}")

        with open(pred_file_path, "r", encoding="utf-8") as f:
            pred_result = json.load(f)
        assert_pred_samples(pred_result)

        pred_basename = os.path.basename(pred_file_path)

        if "gpt" in metrics:
            score_list = evaluate_gpt(pred_result, args.task, args.max_retries)
            out_path = os.path.join(args.output_dir, f"score_gpt_{pred_basename}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(score_list, f, indent=4, ensure_ascii=False)
            print(f"Saved GPT score file: {out_path}")

        if "bert" in metrics:
            bert_result = evaluate_bert(
                pred_result,
                local_model_path=args.bert_local_model_path,
                model_id=args.bert_model_id,
            )
            out_path = os.path.join(args.output_dir, f"score_bert_{pred_basename}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(bert_result, f, indent=4, ensure_ascii=False)
            print(f"Saved BERTScore file: {out_path}")

        if "rouge" in metrics:
            rouge_result = evaluate_rouge(pred_result)
            out_path = os.path.join(args.output_dir, f"score_rouge_{pred_basename}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rouge_result, f, indent=4, ensure_ascii=False)
            print(f"Saved ROUGE score file: {out_path}")


if __name__ == "__main__":
    main()
