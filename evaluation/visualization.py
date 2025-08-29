import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_result(image, mask_all, gt_mask, thinking, question, save_dir="visualization_results"):
    """保存一行三列的可视化图像"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建一行三列的子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 第一列：原图
    axes[0].imshow(image)
    axes[0].set_title('raw image', fontsize=14)
    axes[0].axis('off')
    
    # 第二列：预测掩码叠加原图
    axes[1].imshow(image)
    if mask_all.sum() > 0:
        overlay = np.zeros_like(np.array(image))
        overlay[mask_all] = [255, 0, 0]  # 红色
        axes[1].imshow(overlay, alpha=0.5)
    axes[1].set_title('predicted mask', fontsize=14)
    axes[1].axis('off')
    
    # 第三列：真实掩码叠加原图
    axes[2].imshow(image)
    if gt_mask.sum() > 0:
        overlay = np.zeros_like(np.array(image))
        overlay[gt_mask] = [0, 255, 0]  # 绿色
        axes[2].imshow(overlay, alpha=0.5)
    axes[2].set_title('ground truth mask', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    filename = f"vis_{np.random.randint(10000, 99999)}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # txt save thinking
    with open(save_path.replace(".png", ".txt"), "w") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Thinking: {thinking}")
        
def visualize_result_with_bboxes_and_points(image, boxes, points, gt_bbox, thinking, question, save_dir="visualization_results"):
    """保存一行三列的可视化图像"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建一行三列的子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 第一列：原图
    axes[0].imshow(image)
    axes[0].set_title('raw image', fontsize=14)
    axes[0].axis('off')
    
    # 第二列：预测掩码叠加原图
    axes[1].imshow(image)
    for box in boxes:
        axes[1].add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2))
    for point in points:
        axes[1].plot(point[0], point[1], 'ro', markersize=5)
    axes[1].set_title('predicted bboxes and points', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(image)
    axes[2].add_patch(plt.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1], fill=False, edgecolor='green', linewidth=2))
    for point in points:
        axes[2].plot(point[0], point[1], 'ro', markersize=5)
    axes[2].set_title('ground truth bbox', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    filename = f"vis_{np.random.randint(10000, 99999)}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # txt save thinking
    with open(save_path.replace(".png", ".txt"), "w") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Thinking: {thinking}")