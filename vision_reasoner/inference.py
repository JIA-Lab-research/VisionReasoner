# test_qwen_vl_model.py
import argparse
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models.vision_reasoner_model import VisionReasonerModel
from utils import visualize_results_enhanced

def main():
    parser = argparse.ArgumentParser(description="Test unified vision model on a single image")
    parser.add_argument("--model_path", type=str, default='pretrained_models/VisionReasoner-7B', help="Path to the model")
    parser.add_argument("--task_router_model_path", type=str, default="pretrained_models/TaskRouter-1.5B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--image_path", type=str, default="assets/airplanes.png", help="Path to the input image")
    parser.add_argument("--query", type=str, default="How many airplanes are there in this image?", help="Query/instruction for the model")
    parser.add_argument("--task", type=str, choices=["auto", "detection", "segmentation", "counting", "vqa", "generation", "depth_estimation"], 
                        default="auto", help="Task type (default: auto)")
    parser.add_argument("--output_path", type=str, default="result_visualization.png", help="Path to save the output visualization")
    parser.add_argument("--hybrid_mode", action="store_true", help="Whether to use YOLO for object detection")
    parser.add_argument("--yolo_model_path", type=str, default="yolov8x-worldv2.pt", help="Path to the YOLO model")
    parser.add_argument("--refer_image_path", type=str, default="", help="Path to the reference image")
    parser.add_argument("--image_prompt", type=str, default="", help="Prompt for image generation")
    parser.add_argument("--generation_mode", action="store_true", help="Whether to use generation model")
    parser.add_argument("--generation_model_name", type=str, default="gpt-image-1", help="Name of the generation model")
    args = parser.parse_args()
    
    # Determine task type
    if args.image_prompt != "":
        assert args.generation_mode, "Please set --generation_mode to True when using image prompt"
        task_type = "generation"
    else:
        task_type = args.task
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.generation_mode:
        model = VisionReasonerModel(reasoning_model_path=args.model_path, 
                                task_router_model_path=args.task_router_model_path, 
                                segmentation_model_path=args.segmentation_model_path,
                                generation_model_path=args.generation_model_name)
    elif args.hybrid_mode:
        model = VisionReasonerModel(reasoning_model_path=args.model_path, 
                                task_router_model_path=args.task_router_model_path, 
                                segmentation_model_path=args.segmentation_model_path,
                                yolo_model_path=args.yolo_model_path)
    else:
        model = VisionReasonerModel(reasoning_model_path=args.model_path, 
                                task_router_model_path=args.task_router_model_path, 
                                segmentation_model_path=args.segmentation_model_path)

        
    if task_type != "generation":
        # Load image
        print(f"Loading image from {args.image_path}...")
        image = Image.open(args.image_path).convert("RGB")
        
    if task_type == "auto":
        result, task_type = model.process_single_image(image, args.query, return_task_type=True)
    elif task_type == "detection":
        result = model.detect_objects(image, args.query)
    elif task_type == "segmentation":
        result = model.segment_objects(image, args.query)
    elif task_type == "counting":
        result = model.count_objects(image, args.query)
    elif task_type == "generation":
        result = model.generate_image(args.refer_image_path, args.image_prompt)
    elif task_type == "depth_estimation":
        result = model.depth_estimation(image, args.query)
    else:    # VQA
        result = model.answer_question(image, args.query)
    
    # Print results
    print("\n===== Results =====")
    print("Task type: ", task_type)
    if args.image_prompt != "":
        print("User prompt: ", args.image_prompt)
    else:
        print("User question: ", args.query)
        if 'thinking' in result and result['thinking'].strip() != "":
            print("Thinking process: ", result['thinking'])
    
    # print("Response: ", result)

    if task_type == "detection":
        print(f"Total number of detected objects: {len(result['bboxes'])}")
    elif task_type == "segmentation":
        print(f"Total number of segmented objects: {len(result['bboxes'])}")
    elif task_type == "counting":
        print(f"Total number of interested objects is: {result['count']}")
    elif task_type == "generation":
        result.save(args.output_path, format="PNG")
        print(f"The generated image is saved as '{args.output_path}'")
    elif task_type == "depth_estimation":
        result['bbox_result'].save(args.output_path.replace(".png", "_bbox.png"), format="PNG")
        result['depth_map'].save(args.output_path.replace(".png", "_full.png"), format="PNG")
        print(f"The depth map is saved as '{args.output_path.replace(".png", "_bbox.png")}' and '{args.output_path.replace(".png", "_full.png")}'")
    else:  # QA
        print(f"The answer is: {result['answer']}")
    
    if task_type != "generation" and task_type != "vqa" and task_type != "counting":
        # Visualize results with the new three-panel layout
        visualize_results_enhanced(image, result, task_type, args.output_path)
        print(f"\nResult visualization saved as '{args.output_path}'")



if __name__ == "__main__":
    main()