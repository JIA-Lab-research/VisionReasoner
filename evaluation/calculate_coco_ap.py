import os
import json
import glob
import numpy as np
from argparse import ArgumentParser
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    parser.add_argument("--gt_json_path", type=str, required=True, help="path to COCO ground truth json file")
    return parser.parse_args()

def calculate_metrics(output_dir, gt_json_path):
    # get all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "output_*.json")))
    
    if not output_files:
        print(f"cannot find output files in {output_dir}")
        return
    
    # for accumulating all data
    pred_results = []
    
    pred_results_constant_score = []
    
    pred_results_constant_exist_score = []
    
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
            
            # Original bbox processing code
            bbox = item['bbox']
            bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
               
            pred_results.append({
                "image_id": item['image_id'], 
                "category_id": item['category_id'], 
                "bbox": bbox, 
                "score": item['score']
            })
            
            pred_results_constant_score.append({
                "image_id": item['image_id'], 
                "category_id": item['category_id'], 
                "bbox": bbox, 
                "score": 1.0
            })
            
            pred_results_constant_exist_score.append({
                "image_id": item['image_id'], 
                "category_id": item['category_id'], 
                "bbox": bbox, 
                "score": 1.0 if item['score'] <= 0.2 else 0.0
            })
    
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
    
    coco_gt = COCO(gt_json_path)  # load ground truth
    coco_dt = coco_gt.loadRes(pred_results)  # load prediction results

    # initialize evaluation object (task type: bbox/keypoints/segmentation)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')  # select task type

    # run evaluation
    coco_eval.evaluate()  # calculate matches
    coco_eval.accumulate()  # accumulate metrics
    coco_eval.summarize()  # output results
    

if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.output_dir, args.gt_json_path)
