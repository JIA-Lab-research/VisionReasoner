import os
import json
import glob
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    return parser.parse_args()

def calculate_metrics(output_dir):
    # get all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "output_*.json")))
    
    if not output_files:
        print(f"cannot find output files in {output_dir}")
        return
    
    correct_pred = 0
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
            
            if item['accurate'] >= 1:
                correct_pred += 1
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
    # print(f"All one accuracy: {(1-np.array(all_anomaly_labels)).sum() / total_cnt:.4f}")
    print(f"Accuracy (correct_pred / total_cnt): {correct_pred / total_cnt:.4f}")
    print(f"----------------------------------")

if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.output_dir)
