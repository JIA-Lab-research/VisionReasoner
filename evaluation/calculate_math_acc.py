import os
import json
import glob
import numpy as np
from argparse import ArgumentParser
from math_verify import parse, verify
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    return parser.parse_args()

# def extract_choices(text):
#     """
#     从文本中提取choices并转换为list
#     格式: Choices:\n(A) 135°\n(B) 140°\n(C) 145°\n(D) 150°
#     """
#     if "Choices:\n" not in text:
#         return []
    
#     # 找到Choices:后面的内容
#     start_idx = text.find("Choices:\n") + len("Choices:\n")
#     choices_text = text[start_idx:]
    
#     # 按行分割并过滤空行
#     choices_lines = [line.strip() for line in choices_text.split('\n') if line.strip()]
    
#     # 提取选项内容（去掉选项标识符如(A), (B)等）
#     choices = []
#     for line in choices_lines:
#         # 匹配格式如 "(A) 135°" 或 "A) 135°"
#         if line.startswith('(') and ')' in line:
#             # 找到第一个)的位置
#             end_bracket = line.find(')')
#             if end_bracket != -1:
#                 choice_content = line[end_bracket + 1:].strip()
#                 choices.append(choice_content)
#         elif line[0].isalpha() and line[1] == ')':
#             # 匹配格式如 "A) 135°"
#             choice_content = line[2:].strip()
#             choices.append(choice_content)
    
#     return choices

def calculate_metrics(output_dir):
    # get all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "output_*.json")))
    
    if not output_files:
        print(f"cannot find output files in {output_dir}")
        return
    
    accurate_cnt = 0
    total_cnt = 0
    # for calculating think text length
    think_text_lengths = []

    # read and process all files
    for file_path in output_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # process all items in each file
        for item in results:
            # Calculate think text length if available
            if 'think' in item and item['think']:
                think_text_lengths.append(len(item['think'].split()))
            
            gt_parsed = parse(f"${item['ground_truth']}$")
            
            # if 'choices' in item and item['choices'] is not None:
            #     try: 
            #         item['prediction'] = item['choices'][ord(item['prediction'].lower()) - ord('a')]
            #         # print(f"Extracted choices: {choices}", f"prediction: {item['prediction']}",f"ground_truth: {item['ground_truth']}")
            #     except:
            #         pass
            
            answer_parsed = parse(
                f"${item['prediction']}$",
                extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
            )
            accurate = 1.0 if verify(answer_parsed, gt_parsed) else 0.0
            accurate_cnt += accurate
            
            total_cnt += 1
    
    # Calculate think text metrics
    if think_text_lengths:
        avg_think_length = sum(think_text_lengths) / len(think_text_lengths)
        min_think_length = min(think_text_lengths)
        max_think_length = max(think_text_lengths)
        print(f"\n-----------------Think Text Statistics----------------------------------")
        print(f"Number of think texts: {len(think_text_lengths)}")
        print(f"Average think text length: {avg_think_length:.2f} words")
        print(f"Minimum think text length: {min_think_length} words")
        print(f"Maximum think text length: {max_think_length} words")
        print(f"------------------------------------------------------------------\n")
    
    # print the results
    print(f"Accuracy (correct_pred / total_cnt): {accurate_cnt / total_cnt:.4f}")
    print(f"----------------------------------")

if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.output_dir)
