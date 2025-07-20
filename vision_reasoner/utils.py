import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import pdb

def visualize_results_enhanced(image, result, task_type, output_path):
    """
    Enhanced visualization with three-panel layout
    """
    # Create a figure with 3 subplots
    plt.figure(figsize=(15, 5))
    
    # First panel: Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Second panel: Image with bounding boxes
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    
    if 'bboxes' in result and result['bboxes']:
        for bbox in result['bboxes']:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
    
    if 'points' in result and result['points']:
        for point in result['points']:
            plt.plot(point[0], point[1], 'go', markersize=8)  # Green point
    
    plt.title('Image with Bounding Boxes')
    plt.axis('off')
    
    # Third panel: Mask overlay (for segmentation tasks)
    plt.subplot(1, 3, 3)
    plt.imshow(image, alpha=0.6)
    
    if task_type == 'segmentation' and 'masks' in result and result['masks'] is not None:
        mask = result['masks']
        if np.any(mask):
            mask_overlay = np.zeros_like(np.array(image))
            mask_overlay[mask] = [255, 0, 0]  # Red color for mask
            plt.imshow(mask_overlay, alpha=0.4)
    
    if task_type == 'detection' or task_type == 'counting':
        # For non-segmentation tasks, just show bounding boxes again
        if 'bboxes' in result and result['bboxes']:
            for bbox in result['bboxes']:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 fill=True, edgecolor='red', facecolor='red', alpha=0.3)
                plt.gca().add_patch(rect)
    
    task_title = {
        'detection': 'Detection Overlay',
        'segmentation': 'Segmentation Mask',
        'counting': 'Counting Results',
        'qa': 'Visual QA'
    }
    
    plt.title(task_title.get(task_type, 'Results Overlay'))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def draw_points(image, keypoints, scores, pose_keypoint_color, keypoint_score_threshold, radius, show_keypoint_weight):
    if pose_keypoint_color is not None:
        assert len(pose_keypoint_color) == len(keypoints)
    for kid, (kpt, kpt_score) in enumerate(zip(keypoints, scores)):
        x_coord, y_coord = int(kpt[0]), int(kpt[1])
        if kpt_score > keypoint_score_threshold:
            color = tuple(int(c) for c in pose_keypoint_color[kid])
            if show_keypoint_weight:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)
                transparency = max(0, min(1, kpt_score))
                cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
            else:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)

def draw_links(image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold, thickness, show_keypoint_weight, stick_width = 2):
    height, width, _ = image.shape
    if keypoint_edges is not None and link_colors is not None:
        assert len(link_colors) == len(keypoint_edges)
        for sk_id, sk in enumerate(keypoint_edges):
            x1, y1, score1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]), scores[sk[0]])
            x2, y2, score2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]), scores[sk[1]])
            if (
                x1 > 0
                and x1 < width
                and y1 > 0
                and y1 < height
                and x2 > 0
                and x2 < width
                and y2 > 0
                and y2 < height
                and score1 > keypoint_score_threshold
                and score2 > keypoint_score_threshold
            ):
                color = tuple(int(c) for c in link_colors[sk_id])
                if show_keypoint_weight:
                    X = (x1, x2)
                    Y = (y1, y2)
                    mean_x = np.mean(X)
                    mean_y = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    polygon = cv2.ellipse2Poly(
                        (int(mean_x), int(mean_y)), (int(length / 2), int(stick_width)), int(angle), 0, 360, 1
                    )
                    cv2.fillConvexPoly(image, polygon, color)
                    transparency = max(0, min(1, 0.5 * (keypoints[sk[0], 2] + keypoints[sk[1], 2])))
                    cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
                else:
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)

def visualize_pose_estimation_results_enhanced(image, result, task_type, output_path):
    palette = np.array(
        [
            [255, 128, 0],
            [255, 153, 51],
            [255, 178, 102],
            [230, 230, 0],
            [255, 153, 255],
            [153, 204, 255],
            [255, 102, 255],
            [255, 51, 255],
            [102, 178, 255],
            [51, 153, 255],
            [255, 153, 153],
            [255, 102, 102],
            [255, 51, 51],
            [153, 255, 153],
            [102, 255, 102],
            [51, 255, 51],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0],
            [255, 255, 255],
        ]
    )

    link_colors = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
    keypoint_colors = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

    numpy_image = np.array(image)
    
    keypoint_edges = result["keypoint_edges"]
    # pdb.set_trace()

    for pose_result in result["pose_results"]:
        scores = np.array(pose_result["scores"])
        keypoints = np.array(pose_result["keypoints"])

        # draw each point on image
        draw_points(numpy_image, keypoints, scores, keypoint_colors, keypoint_score_threshold=0.3, radius=4, show_keypoint_weight=False)

        # draw links
        draw_links(numpy_image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold=0.3, thickness=1, show_keypoint_weight=False)

    # pose_image = Image.fromarray(numpy_image)
    
    # Create a figure with 3 subplots
    plt.figure(figsize=(15, 5))
    
    # First panel: Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Second panel: Image with bounding boxes
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    
    if 'bboxes' in result and result['bboxes']:
        for bbox in result['bboxes']:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
    plt.title('Image with Bounding Boxes')
    plt.axis('off')
    
    
    # Third panel: Image with bounding boxes
    plt.subplot(1, 3, 3)
    plt.imshow(numpy_image)
    plt.title('Pose Estimation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def visualize_depth_estimation_results_enhanced(image, result, task_type, output_path):
        # Create a figure with 3 subplots
    plt.figure(figsize=(15, 5))
    
    # First panel: Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Second panel: Image with bounding boxes
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    
    if 'bboxes' in result and result['bboxes']:
        for bbox in result['bboxes']:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
    plt.title('Image with Bounding Boxes')
    plt.axis('off')
    
    
    # Third panel: Image with bounding boxes
    plt.subplot(1, 3, 3)
    plt.imshow(result['bbox_depth_map'])
    plt.title('Depth Estimation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()