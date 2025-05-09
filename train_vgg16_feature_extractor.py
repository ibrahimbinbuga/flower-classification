from utils.data_loader import get_train_loader
from models.vgg16_feature_extractor import vgg16_feature_extractor
from utils.evaluate import evaluate_model
from utils.save_model import save_model
import torch
import os
import time  

DATA_DIR = os.path.join(os.getcwd(), "datasets", "flowers")
train_loader, class_names = get_train_loader(DATA_DIR, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# VGG16 Feature Extractor Model
start_time = time.time()
model_vgg16_feature = vgg16_feature_extractor(train_loader, num_epochs=10, device=device)

end_time = time.time()
train_time = end_time - start_time

accuracy, precision, recall, f1 = evaluate_model(model_vgg16_feature, train_loader, device=device)
print(f"VGG16 Feature Extractor Accuracy: {accuracy:.2f}%")
print(f"VGG16 Feature Extractor Precision: {precision:.4f}")
print(f"VGG16 Feature Extractor Recall: {recall:.4f}")
print(f"VGG16 Feature Extractor F1-Score: {f1:.4f}")

save_model(model_vgg16_feature, "outputs/saved_models/vgg16_feature_extractor.pth")

print(f"VGG16 Feature Extractor Train Time: {train_time:.2f} seconds")