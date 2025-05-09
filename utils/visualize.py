import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from models.custom_cnn import CustomCNN
import models.vgg16_finetuned

def visualize_features(model, layer_name, data_loader, num_images=5, device="cpu"):
    
    if isinstance(model, CustomCNN):
        model_prefix = "custom_cnn"
    elif hasattr(model, 'features'):
        model_prefix = "vgg16"
    else:
        model_prefix = "unknown_model"

    save_dir = os.path.join(os.getcwd(), 'visualizations')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    features = []
    input_images = []

    if isinstance(model, CustomCNN):
        if layer_name == 'conv1':
            layer = model.conv1
        elif layer_name == 'conv3':
            layer = model.conv3
        elif layer_name == 'conv5':
            layer = model.conv5
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
            
    elif hasattr(model, 'features'):
        if layer_name == 'conv1':
            layer = model.features[0]
        elif layer_name == 'conv3':
            layer = model.features[7]
        elif layer_name == 'conv5':
            layer = model.features[14]
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
    
    else:
        raise ValueError("Unknown model")

    def hook_fn(module, input, output):
        features.append(output.detach())

    hook = layer.register_forward_hook(hook_fn)

    images, _ = next(iter(data_loader))
    input_images = images[:num_images].clone()

    model.eval()
    with torch.no_grad():
        images = images.to(device)
        _ = model(images)

    feature_maps = features[0][:num_images]

    plt.figure(figsize=(15, num_images * 3))
    for img_idx in range(num_images):
        plt.subplot(num_images, num_images + 1, img_idx * (num_images + 1) + 1)
        img = input_images[img_idx].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')

        feature_map = feature_maps[img_idx].cpu().numpy()
        for i in range(feature_map.shape[0]):
            subplot_idx = img_idx * (num_images + 1) + i + 2
            if subplot_idx <= num_images * (num_images + 1):
                plt.subplot(num_images, num_images + 1, subplot_idx)
                feature = feature_map[i]
                feature_min = feature.min()
                feature_max = feature.max()
                if feature_max > feature_min:
                    feature = (feature - feature_min) / (feature_max - feature_min)
                plt.imshow(feature, cmap='viridis')
                plt.axis('off')

    save_path = os.path.join(save_dir, f'{model_prefix}_{layer_name}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visuals saved: {save_path}")
    plt.close()

    hook.remove()