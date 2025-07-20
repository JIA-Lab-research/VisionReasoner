import argparse
import torch
import json
import numpy as np
import os
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import sys
from scipy.optimize import linear_sum_assignment

# Add the parent directory to the Python path to import model module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_reasoner.models.vision_reasoner_model import VisionReasonerModel
from vision_reasoner.models.qwen_vl import QwenVLModel
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vision_reasoner")
    parser.add_argument("--model_path", type=str, default="pretrained_models/VisionReasoner-7B", choices=["Ricky06662/VisionReasoner-7B", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"])
    parser.add_argument("--task_router_model_path", type=str, default="pretrained_models/TaskRouter-1.5B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # for parallel evaluation
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    return parser.parse_args()

def compute_bbox_iou(bboxes1, bboxes2):
    """
    Calculate IOU matrix between two sets of bounding boxes
    bboxes1: shape (N, 4) prediction boxes
    bboxes2: shape (M, 4) ground truth boxes
    Returns: shape (N, M) IOU matrix
    """
    # Expand dimensions to support broadcasting
    bboxes1 = np.array(bboxes1)[:, None, :]  # (N, 1, 4)
    bboxes2 = np.array(bboxes2)[None, :, :]  # (1, M, 4)
    
    # Calculate intersection area
    x1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    x2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    
    # Calculate intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate the areas of the two sets of bboxes
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    iou = np.where(union > 0, intersection / union, 0)
    
    return iou

def main():
    args = parse_args()
    
    # Initialize model
    if args.model == "qwen2vl":
        model = QwenVLModel(model_path=args.model_path)
    elif args.model == "qwen25vl":
        model = QwenVLModel(model_path=args.model_path)
    elif args.model == "vision_reasoner":
        model = VisionReasonerModel(reasoning_model_path=args.model_path, 
                                    task_router_model_path=args.task_router_model_path, 
                                    segmentation_model_path=args.segmentation_model_path)
    
    # Load dataset
    dataset = load_dataset(args.test_data_path, split="test")
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = min(start_idx + part_size if args.idx < args.num_parts - 1 else total_len, total_len)
    range_list = range(start_idx, end_idx)
    
    dataset = dataset.select(range_list)
    
    all_outputs = []
    
    # Prepare batches
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
        batch_data = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
        
        batch_images = [item["image"].convert("RGB") for item in batch_data]
        batch_questions = [item["text"].lower().strip(".\"?!") for item in batch_data]
        id_list = [{
            "image_id": item["image_id"],
            "ann_id": item["ann_id"],
            "img_height": item["img_height"],
            "img_width": item["img_width"],
            "bbox": item["bbox"],
            "cat_id": item["cat_id"]
        } for item in batch_data]
        
        process_batch(model, batch_images, batch_questions, id_list, all_outputs)
    
    # Save results
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

def process_batch(model, batch_images, batch_questions, id_list, all_outputs):
    """Process a batch of images and questions"""
    batch_results = model.detect_objects_batch(batch_images, batch_questions)
    for i, result in enumerate(batch_results):
        try:
            thinking = result["thinking"]
            bboxes = result["bboxes"]
            
            gt_bboxes = id_list[i]["bbox"]

            if gt_bboxes and len(bboxes) > 0:
                # # Use vectorized calculation of IOU matrix
                # cost_matrix = -compute_bbox_iou(bboxes, gt_bboxes)  # Use negative IOU as cost
                
                # # Use Hungarian algorithm for matching
                # pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
                
                # # Assign scores to each predicted box
                # scores = np.zeros(len(bboxes))
                # for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                #     scores[pred_idx] = -cost_matrix[pred_idx, gt_idx]  # Convert back to positive IOU value
                
                # Add results
                for pred_idx, pred_bbox in enumerate(bboxes):
                    all_outputs.append({
                        "image_id": int(id_list[i]["image_id"]),
                        "ann_id": int(id_list[i]["ann_id"]),
                        "think": thinking,
                        "category_id": int(id_list[i]["cat_id"]),
                        "bbox": pred_bbox,
                        #"score": float(max(scores[pred_idx],0.0))  # Use the match score
                        "score": float((pred_bbox[2]-pred_bbox[0])*(pred_bbox[3]-pred_bbox[1])/(id_list[i]["img_width"]*id_list[i]["img_height"]))
                    })
            else:
                # If there are no ground truth boxes or predicted boxes, score is 0
                for pred_bbox in bboxes:
                    all_outputs.append({
                        "image_id": int(id_list[i]["image_id"]),
                        "ann_id": int(id_list[i]["ann_id"]),
                        "think": thinking,
                        "category_id": int(id_list[i]["cat_id"]),
                        "bbox": pred_bbox,
                        "score": 0.0
                    })
            
        except Exception as e:
            # raise
            print(f"Error processing result: {e}")
            # Skip this because the implementation is different from the original
            continue

if __name__ == "__main__":
    main()