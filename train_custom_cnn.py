from utils.data_loader import get_train_loader
from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.visualize import visualize_features
from utils.save_model import save_model
from models.custom_cnn import CustomCNN
import torch
import os
import time  

DATA_DIR = os.path.join(os.getcwd(), "datasets", "flowers")
train_loader, class_names = get_train_loader(DATA_DIR, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Custom CNN Model
model_custom_cnn = CustomCNN(num_classes=5).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_custom_cnn.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()

model_custom_cnn = train_model(model_custom_cnn, train_loader, criterion, optimizer, num_epochs=15, device=device)

end_time = time.time()
train_time = end_time - start_time  

accuracy, precision, recall, f1 = evaluate_model(model_custom_cnn, train_loader, device=device)
print(f"Custom CNN Accuracy: {accuracy:.2f}%")
print(f"Custom CNN Precision: {precision:.4f}")
print(f"Custom CNN Recall: {recall:.4f}")
print(f"Custom CNN F1-Score: {f1:.4f}")

visualize_features(model_custom_cnn, 'conv1', train_loader, num_images=5, device=device)
visualize_features(model_custom_cnn, 'conv3', train_loader, num_images=5, device=device)
visualize_features(model_custom_cnn, 'conv5', train_loader, num_images=5, device=device)

save_model(model_custom_cnn, "outputs/saved_models/custom_cnn.pth")

print(f"Custom CNN Train Time: {train_time:.2f} seconds")