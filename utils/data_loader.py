import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def get_train_loader(data_dir, batch_size=32, shuffle=True):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir, "train")
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, train_dataset.classes

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path
    
def get_test_loader(data_dir, batch_size=32, shuffle=False):
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_path = os.path.join(data_dir, "test")
    test_dataset = TestDataset(test_path, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader
