# train_vgg16_finetuned.py
from utils.data_loader import get_train_loader
from models.vgg16_finetuned import vgg16_finetuned
from utils.evaluate import evaluate_model
from utils.visualize import visualize_features 
from utils.save_model import save_model
from utils.train import train_model  
import torch
import os
import time  

DATA_DIR = os.path.join(os.getcwd(), "datasets", "flowers")
train_loader, class_names = get_train_loader(DATA_DIR, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# VGG16 Fine-Tuned Model
start_time = time.time()
model_vgg16_finetuned = vgg16_finetuned(train_loader, num_epochs=10, device=device)

end_time = time.time()
train_time = end_time - start_time

accuracy, precision, recall, f1 = evaluate_model(model_vgg16_finetuned, train_loader, device=device)
print(f"VGG16 Fine-Tuned Accuracy: {accuracy:.2f}%")
print(f"VGG16 Fine-tuned Precision: {precision:.4f}")
print(f"VGG16 Fine-tuned Recall: {recall:.4f}")
print(f"VGG16 Fine-tuned F1-score: {f1:.4f}")

visualize_features(model_vgg16_finetuned, 'conv1', train_loader, num_images=5, device=device)
visualize_features(model_vgg16_finetuned, 'conv3', train_loader, num_images=5, device=device)
visualize_features(model_vgg16_finetuned, 'conv5', train_loader, num_images=5, device=device)

save_model(model_vgg16_finetuned, "outputs/saved_models/vgg16_finetuned.pth")

print(f"VGG16 Fine-Tuned Train Time: {train_time:.2f} seconds")
