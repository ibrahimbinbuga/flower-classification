import torch

def save_model(model, model_path):
    
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
