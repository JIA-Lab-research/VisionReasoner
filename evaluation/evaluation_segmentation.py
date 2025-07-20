import argparse
import torch
import json
import numpy as np
import os
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import sys

# Add the parent directory to the Python path to import model module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_reasoner.models.vision_reasoner_model import VisionReasonerModel
from vision_reasoner.models.qwen_vl import QwenVLModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vision_reasoner")
    parser.add_argument("--model_path", type=str, default="pretrained_models/VisionReasoner-7B")
    parser.add_argument("--task_router_model_path", type=str, default="pretrained_models/TaskRouter-1.5B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)

    # for parallel evaluation
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    return parser.parse_args()

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0, 0
    return intersection, union

def compute_bbox_iou(bbox1, bbox2):
    # Calculate the intersection area of two bboxes
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of the two bboxes
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0
    
    return intersection / union

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
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    
    dataset = dataset.select(range(start_idx, end_idx))
    
    # Check if dataset has bbox information
    has_bbox = 'bbox' in dataset[0]
    
    all_outputs = []
    
    # Prepare batches
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
        batch_data = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
        
        batch_images = [item["image"].convert("RGB") for item in batch_data]
        batch_questions = [item["text"].lower().strip(".\"?!") for item in batch_data]
        id_list = [{
            "image_id": item["image_id"],
            "ann_id": item["ann_id"],
            "mask": item["mask"],
            "img_height": item["img_height"],
            "img_width": item["img_width"],
            "bbox": item["bbox"] if has_bbox else None
        } for item in batch_data]
        
        process_batch(model, batch_images, batch_questions, id_list, all_outputs, has_bbox)
    
    # Save results
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

def process_batch(model, batch_images, batch_questions, id_list, all_outputs, has_bbox):
    """Process a batch of images and questions"""
    batch_results = model.segment_objects_batch(batch_images, batch_questions)
    
    for i, result in enumerate(batch_results):
        try:
            thinking = result["thinking"]
            bboxes = result["bboxes"]
            mask_all = result["masks"]
            gt_mask = np.array(id_list[i]["mask"])
            
            intersection, union = compute_iou(mask_all, gt_mask)
            
            bbox_iou = 0.0
            if has_bbox:
                try:     
                    gt_bbox = id_list[i]["bbox"]
                    for pred_bbox in bboxes:
                        if compute_bbox_iou(pred_bbox, gt_bbox) > 0.5:
                            bbox_iou = 1.0
                            break
                except Exception as e:
                    print(f"Bbox error: {e}, Image ID: {id_list[i]['image_id']}, Ann ID: {id_list[i]['ann_id']}")
                    bbox_iou = 0.0
            
            all_outputs.append({
                "image_id": id_list[i]["image_id"],
                "ann_id": id_list[i]["ann_id"],
                "think": thinking,
                "intersection": int(intersection),
                "union": int(union),
                "bbox_iou": bbox_iou
            })
            
        except Exception as e:
            print(f"Error processing result: {e}")
            # Add penalty in this situation
            all_outputs.append({
                "image_id": id_list[i]["image_id"],
                "ann_id": id_list[i]["ann_id"],
                "think": "",
                "intersection": 0,
                "union": np.array(id_list[i]["mask"]).sum(),
                "bbox_iou": 0.0
            })
        

if __name__ == "__main__":
    main()