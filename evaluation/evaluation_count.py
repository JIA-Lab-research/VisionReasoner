import argparse
import torch
import json
import numpy as np
import os
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import sys

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
    all_outputs = []

    # Process in batches
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
        batch_data = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
        
        batch_images = [item["image"].convert("RGB") for item in batch_data]
        batch_questions = [item["text"].lower().strip(".\"?!") for item in batch_data]
        id_list = [{
            "image_id": item["image_id"],
            "ann_id": item["ann_id"],
            "img_height": item["img_height"],
            "img_width": item["img_width"],
            "gt_count": item["count"]
        } for item in batch_data]
        process_batch(model, batch_images, batch_questions, id_list, all_outputs)
    
    # Save results
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

def process_batch(model, batch_images, batch_questions, id_list, all_outputs):
    """Process a batch of images and questions"""
    batch_results = model.count_objects_batch(batch_images, batch_questions)
    
    for i, result in enumerate(batch_results):
        try:
            thinking = result["thinking"]
            bboxes = result["bboxes"]
            pred_count = result["count"]
            
            all_outputs.append({
                "image_id": id_list[i]["image_id"],
                "ann_id": id_list[i]["ann_id"],
                "think": thinking,
                "pred_count": pred_count,
                "gt_count": id_list[i]["gt_count"]
            })
            
        except Exception as e:
            # raise
            print(f"Error processing result: {e}")
            # Add penalty in this situation
            all_outputs.append({
                "image_id": id_list[i]["image_id"],
                "ann_id": id_list[i]["ann_id"],
                "think": "",
                "pred_count": 1,
                "gt_count": id_list[i]["gt_count"]
            })
    
    print(f"Processed batch of {len(batch_images)} images")
            

if __name__ == "__main__":
    main()