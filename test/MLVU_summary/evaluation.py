'''
ref: https://github.com/JUNJIE99/MLVU/blob/main/evaluation/generation_evaluation/evaluate_summary.py
doc: https://yunwu.apifox.cn/api-232421917  https://yunwu.apifox.cn/doc-5459019
'''

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import os
import ast
from tqdm import tqdm
from openai import OpenAI

eval_model_type = 'gpt-5-mini-2025-08-07'
api_key = os.environ.get('API_KEY')
client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key=api_key,
)


def annotate(correct_answer, predict_answer):
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
            """
        },
        {
            "role": "user",
            "content": f"""
                Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
                Standard Answer: {correct_answer}
                Respondent's Answer: {predict_answer}
            """
        }
    ]
    response = client.chat.completions.create(
        model=eval_model_type,
        messages=messages,
        timeout=100,
        temperature=0,
    )
    output = response.choices[0].message.content
    score = ast.literal_eval(output)
    # print(score) # {'score_completeness': 3, 'score_reliability': 4, 'total_score': 7}
    return score




def main():
    pred_result_file_path = '/data1/lyc/flexcachegen_outputs/pred_summary_all.json'
    with open(pred_result_file_path, 'r', encoding='utf-8') as f:
        pred_result = json.load(f)

    score_completeness = []
    score_reliability = []
    total_score = []
    for sample in tqdm(pred_result):
        score = annotate(
            sample['A'],
            sample['pred'],
        )
        score_completeness.append(score['score_completeness'])
        score_reliability.append(score['score_reliability'])
        total_score.append(score['total_score'])

    print('='*20 + ' score ' + '='*20)
    print(f'completeness: {sum(score_completeness)/len(score_completeness):.2f}')
    print(f'reliability : {sum(score_reliability)/len(score_reliability):.2f}')
    print(f'total       : {sum(total_score)/len(total_score):.2f}')



if __name__ == "__main__":
    main()