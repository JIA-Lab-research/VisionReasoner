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
from vision_reasoner.models.visurf_model import ViSurfModel

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
    elif args.model == "visurf":
        model = ViSurfModel(reasoning_model_path=args.model_path, 
                                    task_router_model_path=args.task_router_model_path, 
                                    segmentation_model_path=args.segmentation_model_path)
    # Load dataset
    dataset = load_from_disk(args.test_data_path)['test']
    # dataset = dataset.select(range(100))
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    
    dataset = dataset.select(range(start_idx, end_idx))
    
    # Check if dataset has bbox information    
    all_outputs = []
    
    # Prepare batches
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
        batch_data = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
        
        batch_images = [item["image"].convert("RGB") for item in batch_data]
        batch_questions = [item["text"].lower() for item in batch_data]
        id_list = [{
            "text": item["text"],
            "answer": item["answer"],
            "choices": item["choices"] if 'choices' in item else None
        } for item in batch_data]
        
        process_batch(model, batch_images, batch_questions, id_list, all_outputs)
    
    # Save results
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

def process_batch(model, batch_images, batch_questions, id_list, all_outputs):
    """Process a batch of images and questions"""
    batch_results = model.answer_questions_batch_math(batch_images, batch_questions)
    
    for i, result in enumerate(batch_results):
        try:
            thinking = result["thinking"]
            answer = result["answer"]
            
            all_outputs.append({
                "text": id_list[i]["text"],
                "think": thinking,
                "prediction": id_list[i]['choices'][ord(answer.strip().lower()[0]) - ord('a')] if id_list[i]['choices'] is not None else answer,
                "ground_truth": id_list[i]['answer'],
                "choices": id_list[i]['choices'] if 'choices' in id_list[i] else None
            })
            
        except Exception as e:
            print(f"Error processing result: {e}, Raw answer is {answer}")
            # Add penalty in this situation
            all_outputs.append({
                "text": id_list[i]["text"],
                "think": "",
                "prediction": "",
                "ground_truth": id_list[i]['answer'],
                "choices": id_list[i]['choices'] if 'choices' in id_list[i] else None
            })

if __name__ == "__main__":
    main()