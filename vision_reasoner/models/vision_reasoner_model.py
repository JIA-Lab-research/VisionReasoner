import torch
import numpy as np
import re
import json
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image as PILImage
from ultralytics import YOLOWorld
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
from .vggt.models.vggt import VGGT
from .vggt.utils.load_fn import load_and_preprocess_images
from .base_model import (
    BaseVisionModel,
    DetectionModel,
    SegmentationModel,
    CountingModel,
    QAModel
)
from qwen_vl_utils import process_vision_info
from .task_router import TaskRouter

STOP_WORDS = {"is", "are", "find", "the", "segment", "all", "in", "image", 
              "how", "many", "there", "locate", "please"}
MAX_QUERY_WORDS = 2


class VisionReasonerModel(BaseVisionModel, DetectionModel, SegmentationModel, CountingModel, QAModel):
    """
    VisionReasoner model implementing all task interfaces
    """
    def __init__(self, 
                 reasoning_model_path="Ricky06662/VisionReasoner-7B", 
                 segmentation_model_path="facebook/sam2-hiera-large",
                 depth_estimation_model_path="facebook/VGGT-1B",
                 task_router_model_path="Ricky06662/TaskRouter-1.5B",
                 yolo_model_path=None,
                 generation_model_path=None):
        """
        Initialize the VisionReasoner model with reasoning and segmentation components
        
        Args:
            reasoning_model_path (str): Path to the reasoning model
            segmentation_model_path (str): Path to the segmentation model
        """
        self.resize_size = 840
        
        # Initialize reasoning model
        self.reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoning_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.reasoning_model.eval()
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")
        
        # Initialize segmentation model
        self.segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)

        # Initialize depth estimation model
        self.depth_estimation_model = VGGT.from_pretrained(depth_estimation_model_path).to("cuda")

        self.task_router = TaskRouter(task_router_model_path)
        
        # Template for detection/segmentation tasks
        self.DETECTION_TEMPLATE = \
            "Please find \"{Question}\" with bboxs and points." \
            "Compare the difference between object(s) and find the most closely matched object(s)." \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
            "i.e., <think> thinking process here </think>" \
            "<answer>{Answer}</answer>"
        
        # Template for QA tasks
        self.QA_TEMPLATE = "{Question}"

        # Initialize YOLO model
        self.use_hybrid_mode = False
        if yolo_model_path:
            self.use_hybrid_mode = True
            self.yolo_model = YOLOWorld(yolo_model_path)
        
        # Initialize generation model
        if generation_model_path:
            self.generation_model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", generation_model_path))

    
    def extract_bbox_points_think(self, output_text, x_factor, y_factor):
        """
        Extract bounding boxes, points, and thinking process from model output
        
        Args:
            output_text (str): Raw output text from the model
            x_factor (float): Scaling factor for x coordinates
            y_factor (float): Scaling factor for y coordinates
            
        Returns:
            tuple: (pred_bboxes, pred_points, think_text, pred_answer)
        """
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
        pred_bboxes = []
        pred_points = []
        pred_answer = None
        think_text = ""
        
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                pred_answer = data
                pred_bboxes = [[
                    int(item['bbox_2d'][0] * x_factor + 0.5),
                    int(item['bbox_2d'][1] * y_factor + 0.5),
                    int(item['bbox_2d'][2] * x_factor + 0.5),
                    int(item['bbox_2d'][3] * y_factor + 0.5)
                ] for item in data]
                pred_points = [[
                    int(item['point_2d'][0] * x_factor + 0.5),
                    int(item['point_2d'][1] * y_factor + 0.5)
                ] for item in data]
            except Exception as e:
                print(f"Error parsing JSON: {e}")
        
        think_pattern = r'<think>([^<]+)</think>'
        think_match = re.search(think_pattern, output_text)
        if think_match:
            think_text = think_match.group(1)
        
        return pred_bboxes, pred_points, think_text, pred_answer
    
    def extract_qa_answer(self, output_text):
        """
        Extract answer for QA tasks
        
        Args:
            output_text (str): Raw output text from the model
            
        Returns:
            dict: Result dictionary with answer and thinking (if available)
        """
        think_pattern = r'<think>([^<]+)</think>'
        think_match = re.search(think_pattern, output_text)
        thinking = think_match.group(1) if think_match else ""
        
        # Remove thinking tags from output to get cleaner answer
        clean_answer = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()
        
        return {
            "answer": clean_answer,
            "thinking": thinking,
            "full_response": output_text
        }
    
    def generate_masks(self, image, bboxes, points=None):
        """
        Generate segmentation masks for given image, bounding boxes and points
        
        Args:
            image (PIL.Image): Input image
            bboxes (list): List of bounding boxes
            points (list): List of points
            
        Returns:
            numpy.ndarray: Combined segmentation mask
        """
        img_height, img_width = image.height, image.width
        mask_all = np.zeros((img_height, img_width), dtype=bool)
        
        if not bboxes:
            return mask_all
        
        if points and len(points) != len(bboxes):
            return mask_all
        
        try:
            self.segmentation_model.set_image(image)
            if points:
                for bbox, point in zip(bboxes, points):
                    masks, scores, _ = self.segmentation_model.predict(
                        point_coords=[point],
                        point_labels=[1],
                        box=bbox
                    )
                    sorted_ind = np.argsort(scores)[::-1]
                    masks = masks[sorted_ind]
                    mask = masks[0].astype(bool)
                    mask_all = np.logical_or(mask_all, mask)
            else:
                for bbox in bboxes:
                    masks, scores, _ = self.segmentation_model.predict(
                        box=bbox
                    )
                    sorted_ind = np.argsort(scores)[::-1]
                    masks = masks[sorted_ind]
                
            return mask_all
        except Exception as e:
            print(f"Error generating masks: {e}")
            return mask_all
    
    def _generate_model_output(self, images, instructions, template, batch_mode=False):
        """
        Generate raw model output for images and instructions
        
        Args:
            images (PIL.Image or List[PIL.Image]): Input image(s)
            instructions (str or List[str]): Text instruction(s)/query(ies)
            template (str): Template to use for the prompt
            batch_mode (bool): Whether to process in batch mode
            
        Returns:
            tuple: (output_texts, scale_factors)
        """
        if not batch_mode:
            images = [images]
            instructions = [instructions]
        
        batch_messages = []
        scale_factors = []
        
        for image, instruction in zip(images, instructions):
            # Prepare image
            original_width, original_height = image.size
            x_factor, y_factor = original_width/self.resize_size, original_height/self.resize_size
            scale_factors.append((x_factor, y_factor))
            resized_image = image.resize((self.resize_size, self.resize_size), PILImage.BILINEAR)
            
            # Format text based on template
            if "{Question}" in template:
                formatted_text = template.format(
                    Question=instruction.lower().strip(".\"?!"),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                )
            else:
                formatted_text = template
                
            # Create message
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": resized_image
                    },
                    {   
                        "type": "text",
                        "text": formatted_text
                    }
                ]
            }]
            batch_messages.append(message)
        
        # Prepare for batch inference
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Generate output
        generated_ids = self.reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        if not batch_mode:
            return output_texts[0], scale_factors[0]
        return output_texts, scale_factors
    
    def route_task(self, instruction):
        """
        Route task based on instruction
        """
        task_type = self.task_router.route_task(instruction)
        return task_type
    
    # BaseVisionModel implementation
    def process_single_image(self, image, instruction, return_task_type=False):
        """
        Process a single image with given instruction
        
        Args:
            image (PIL.Image): Input image
            instruction (str): Text instruction or query
            
        Returns:
            dict: Results dictionary
        """
        # Determine task type based on instruction
        task_type = self.route_task(instruction)
        
        if task_type == "segmentation":
            result = self.segment_objects(image, instruction)
        elif task_type == "detection":
            result = self.detect_objects(image, instruction)
        elif task_type == "counting":
            result = self.count_objects(image, instruction)
        elif task_type == "depth_estimation":
            result = self.depth_estimation(image, instruction)
        elif task_type == "generation":
            result = self.generate_image(image, instruction)
        else:  # Default to VQA
            result = self.answer_question(image, instruction)
        
        if return_task_type:
            return result, task_type
        else:
            return result
    
    def process_batch(self, batch_images, batch_instructions):
        """
        Process a batch of images with given instructions
        
        Args:
            batch_images (list): List of PIL Images
            batch_instructions (list): List of text instructions or queries
            
        Returns:
            list: List of result dictionaries
        """
        results = []
        for image, instruction in zip(batch_images, batch_instructions):
            result = self.process_single_image(image, instruction)
            results.append(result)
        return results
    
    def detect_objects_yolo(self, image, query):
        """
        Detect objects in an image based on a query using YOLO model
        
        Args:
            image: Input image
            query: Text query describing what to detect
            
        Returns:
            dict: Results with bounding boxes and scores
        """
        # Initialize a YOLO model
        query = " ".join([word.strip(".\"?!'") for word in query.lower().strip(".\"?!").split() if word not in STOP_WORDS])
        names = [query]
        self.yolo_model.set_classes(names)

        # Run detection on the given image
        results = self.yolo_model.predict(image)
        
        # Get original image dimensions
        img_height, img_width = image.height, image.width
        
        # Get YOLO's input size
        yolo_input_size = results[0].orig_shape
        
        # Calculate scaling factors
        x_scale = img_width / yolo_input_size[1]
        y_scale = img_height / yolo_input_size[0]
        
        # Scale the bounding boxes back to original image size
        bboxes = results[0].boxes.xyxy
        scaled_bboxes = []
        for bbox in bboxes:
            scaled_bbox = [
                int(bbox[0] * x_scale),
                int(bbox[1] * y_scale),
                int(bbox[2] * x_scale),
                int(bbox[3] * y_scale)
            ]
            scaled_bboxes.append(scaled_bbox)

        return scaled_bboxes

    def if_yolo_condition(self, query):
        """
        Check if YOLO should be used for the given query
        
        Args:
            query (str): Text query describing what to detect
            
        Returns:
            bool: True if YOLO should be used, False otherwise
        """

        # trivial condition
        query_words = [word for word in query.lower().strip(".\"?!").split() if word not in STOP_WORDS]
        return len(query_words) <= MAX_QUERY_WORDS
    
    # DetectionModel implementation
    def detect_objects(self, image, query):
        """
        Detect objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to detect
            
        Returns:
            dict: Results with bounding boxes and scores
        """
        try:
            if self.use_hybrid_mode and self.if_yolo_condition(query):
                bboxes = self.detect_objects_yolo(image, query)
                scores = [1.0] * len(bboxes)
                # use middle point of bbox as point
                points = [[int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)] for bbox in bboxes]
                output_text, thinking, pred_answer = "", "", str(bboxes)
            else:
                output_text, (x_factor, y_factor) = self._generate_model_output(
                    image,
                    query,
                    self.DETECTION_TEMPLATE
                )
                
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
                
                # Assign confidence scores (all 1.0 as the model doesn't provide them)
                scores = [1.0] * len(bboxes)
            
            return {
                "bboxes": bboxes,
                "points": points,
                "scores": scores,
                "thinking": thinking,
                "full_response": output_text,
                "pred_answer": pred_answer
            }
        except Exception as e:
            print(f"Error in detection: {e}")
            return {
                "bboxes": [],
                "points": [],
                "scores": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            }
    
    def detect_objects_batch(self, images, queries):
        """
        Detect objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of detection results
        """
        try:
            # TODO: support yolo for batch

            output_texts, scale_factors = self._generate_model_output(
                images,
                queries,
                self.DETECTION_TEMPLATE,
                batch_mode=True
            )
            
            results = []
            for output_text, (x_factor, y_factor) in zip(output_texts, scale_factors):
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
                
                scores = [1.0] * len(bboxes)
                results.append({
                    "bboxes": bboxes,
                    "points": points,
                    "scores": scores,
                    "thinking": thinking,
                    "full_response": output_text,
                    "pred_answer": pred_answer
                })
            return results
        except Exception as e:
            print(f"Error in batch detection: {e}")
            return [{
                "bboxes": [],
                "points": [],
                "scores": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            } for _ in range(len(images))]
    
    # SegmentationModel implementation
    def segment_objects(self, image, query):
        """
        Segment objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to segment
            
        Returns:
            dict: Results with masks and bounding boxes
        """
        try:
            if self.use_hybrid_mode and self.if_yolo_condition(query):
                #bboxes, masks = self.segment_objects_yolo(image, query)
                bboxes = self.detect_objects_yolo(image, query)
                # use middle point of bbox as point
                points = [[int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)] for bbox in bboxes]
                output_text, thinking, pred_answer = "", "", str(bboxes)
            else:
                output_text, (x_factor, y_factor) = self._generate_model_output(
                    image,
                    query,
                    self.DETECTION_TEMPLATE
                )
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
            masks = self.generate_masks(image, bboxes, points)
            
            return {
                "masks": masks,
                "bboxes": bboxes,
                "points": points,
                "thinking": thinking,
                "full_response": output_text,
                "pred_answer": pred_answer
            }
        except Exception as e:
            raise
            print(f"Error in segmentation: {e}")
            img_height, img_width = image.height, image.width
            return {
                "masks": np.zeros((img_height, img_width), dtype=bool),
                "bboxes": [],
                "points": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            }
    
    def segment_objects_batch(self, images, queries):
        """
        Segment objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of segmentation results
        """
        try:
            # TODO: support yolo for batch
            output_texts, scale_factors = self._generate_model_output(
                images,
                queries,
                self.DETECTION_TEMPLATE,
                batch_mode=True
            )
            
            results = []
            for image, output_text, (x_factor, y_factor) in zip(images, output_texts, scale_factors):
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
                
                masks = self.generate_masks(image, bboxes, points)
                results.append({
                    "masks": masks,
                    "bboxes": bboxes,
                    "points": points,
                    "thinking": thinking,
                    "full_response": output_text,
                    "pred_answer": pred_answer
                })
            return results
        except Exception as e:
            print(f"Error in batch segmentation: {e}")
            return [{
                "masks": np.zeros((img.height, img.width), dtype=bool),
                "bboxes": [],
                "points": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            } for img in images]
    
    # CountingModel implementation
    def count_objects(self, image, query):
        """
        Count objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to count
            
        Returns:
            dict: Results with count and bounding boxes
        """
        try:
            if self.use_hybrid_mode and self.if_yolo_condition(query):
                bboxes = self.detect_objects_yolo(image, query)
                # use middle point of bbox as point
                points = [[int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)] for bbox in bboxes]
                output_text, thinking, pred_answer = "", "", str(bboxes)
            else:
                output_text, (x_factor, y_factor) = self._generate_model_output(
                    image,
                    query,
                    self.DETECTION_TEMPLATE
                )
                
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
            
            count = len(bboxes)
            
            return {
                "count": count,
                "bboxes": bboxes,
                "points": points,
                "thinking": thinking,
                "full_response": output_text,
                "pred_answer": pred_answer
            }
        except Exception as e:
            print(f"Error in counting: {e}")
            return {
                "count": 0,
                "bboxes": [],
                "points": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            }
    
    def count_objects_batch(self, images, queries):
        """
        Count objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of counting results
        """
        try:
            # TODO: support yolo for batch
            output_texts, scale_factors = self._generate_model_output(
                images,
                queries,
                self.DETECTION_TEMPLATE,
                batch_mode=True
            )
            
            results = []
            for output_text, (x_factor, y_factor) in zip(output_texts, scale_factors):
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
                
                count = len(bboxes)
                results.append({
                    "count": count,
                    "bboxes": bboxes,
                    "points": points,
                    "thinking": thinking,
                    "full_response": output_text,
                    "pred_answer": pred_answer
                })
            return results
        except Exception as e:
            print(f"Error in batch counting: {e}")
            return [{
                "count": 0,
                "bboxes": [],
                "points": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            } for _ in range(len(images))]
    
    # QAModel implementation
    def answer_question(self, image, question):
        """
        Answer a question about an image
        
        Args:
            image: Input image
            question: Text question
            
        Returns:
            dict: Results with answer and thinking (if available)
        """
        try:
            output_text, _ = self._generate_model_output(
                image,
                question,
                self.QA_TEMPLATE
            )
            
            result = self.extract_qa_answer(output_text)
            return result
        except Exception as e:
            print(f"Error in QA: {e}")
            return {
                "answer": "",
                "thinking": "",
                "full_response": ""
            }
    
    def answer_questions_batch(self, images, questions):
        """
        Answer questions about a batch of images
        
        Args:
            images: List of input images
            questions: List of text questions
            
        Returns:
            list: List of QA results
        """
        try:
            output_texts, _ = self._generate_model_output(
                images,
                questions,
                self.QA_TEMPLATE,
                batch_mode=True
            )
            
            results = []
            for output_text in output_texts:
                result = self.extract_qa_answer(output_text)
                results.append(result)
            return results
        except Exception as e:
            print(f"Error in batch QA: {e}")
            return [{
                "answer": "",
                "thinking": "",
                "full_response": ""
            } for _ in range(len(images))]
            
    def generate_image(self, refer_image_path, image_prompt):
        """
        Generate an image based on a query
        
        Args:
            refer_image_path: Path to the reference image
            image_prompt: Text prompt describing what to generate
            
        Returns:
            dict: Results with generated image and thinking (if available)
        """
        if self.generation_model is None or image_prompt is None:
            raise ValueError("Do not have generation model or query")
        
        try:
            if refer_image_path == "":
                # Generate the image
                output = self.generation_model.images.generate(
                    model="gpt-image-1",
                    prompt=image_prompt,
                )
                image_base64 = output.data[0].b64_json
                
            else:
                output = self.generation_model.images.edit(
                    model="gpt-image-1",
                    image=[open(refer_image_path, "rb")], 
                    prompt=image_prompt,
                )
                image_base64 = output.data[0].b64_json
            
            image = PILImage.open(BytesIO(base64.b64decode(image_base64)))
            return image
        except Exception as e:
            print(f"Error in image generation: {e}")
            return None
        
    def depth_estimation(self, image, query):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = load_and_preprocess_images([image], mode="pad").to(device)
        print(images[0].shape)

        pil_image = images[0].cpu().numpy()
        pil_image = np.transpose(pil_image, (1, 2, 0))  # 从 (C,H,W) 转换为 (H,W,C)
        pil_image = ((pil_image - pil_image.min()) / (pil_image.max() - pil_image.min()) * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(pil_image)

        print(pil_image.size)
        output_text, (x_factor, y_factor) = self._generate_model_output(
            pil_image,
            query,
            self.DETECTION_TEMPLATE
        )
        bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
            output_text, 
            x_factor, 
            y_factor
        )
        
    
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Load and preprocess example images
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.depth_estimation_model.aggregator(images)

            # Predict Depth Maps
            depth_map, _ = self.depth_estimation_model.depth_head(aggregated_tokens_list, images, ps_idx)
            
            # Convert depth map to numpy array and normalize to 0-255 range
            depth_map = depth_map.squeeze().cpu().numpy()
            depth_map = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            
            # 获取原始图像尺寸
            original_image = image
            orig_w, orig_h = original_image.size
            
            # 将深度图转换为PIL图像以便调整大小
            depth_map_pil = PILImage.fromarray(depth_map)
            
            # 计算填充区域和缩放比例
            target_size = max(orig_w, orig_h)
            pad_w = (target_size - orig_w) // 2
            pad_h = (target_size - orig_h) // 2
            
            # 调整深度图大小并裁剪填充区域
            depth_map_resized = depth_map_pil.resize((target_size, target_size), PILImage.BILINEAR)
            depth_map_cropped = depth_map_resized.crop((
                pad_w,                    # left
                pad_h,                    # top
                target_size - pad_w,      # right
                target_size - pad_h       # bottom
            ))
            
            # 转回numpy数组
            depth_map = np.array(depth_map_cropped)
            
            # 对bboxes进行相同的坐标转换
            # 1. 从正方形图像坐标转换到target_size坐标
            square_size = pil_image.size[0]  # 正方形图像的尺寸
            scale_factor = target_size / square_size
            
            # 2. 应用缩放和裁剪变换
            transformed_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                # 缩放到target_size
                x1 = int(x1 * scale_factor)
                y1 = int(y1 * scale_factor)
                x2 = int(x2 * scale_factor)
                y2 = int(y2 * scale_factor)
                
                # 减去padding偏移
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(orig_w, x2 - pad_w)
                y2 = min(orig_h, y2 - pad_h)
                
                transformed_bboxes.append([x1, y1, x2, y2])
            
            # Create RGB depth map
            depth_map_rgb = np.stack([depth_map] * 3, axis=-1)
            
            # Convert original image to numpy array
            original_image = np.array(image)
            
            # Create combined images with two types of masks
            # 1. Using transformed bounding box mask
            bbox_mask = np.zeros_like(original_image, dtype=bool)
            for bbox in transformed_bboxes:
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:  # 确保bbox有效
                    bbox_mask[y1:y2, x1:x2] = True
            
            bbox_combined = np.where(bbox_mask, depth_map_rgb, original_image)
            bbox_result = Image.fromarray(bbox_combined)
            
            return {
                'bbox_result': bbox_result,  # Result using transformed bounding box mask
                'depth_map': Image.fromarray(depth_map),      # Original depth map
                'bboxes': transformed_bboxes,  # Transformed bounding boxes
            }
        