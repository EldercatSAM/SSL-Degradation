from torch.utils.data import Dataset
from torchvision import datasets
import torchvision
import os

def load_dataset(dataset_name, root = '/path/to/your/imagenet/', split="train"):
    
    if dataset_name == "imagenet-1k":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=3),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), 
                                            (0.229, 0.224, 0.225)),
        ])
        root = os.path.join(root, 'imagenet-1k')
        return ReturnIndexDataset(os.path.join(root, split), transform=transform)
    
    else:
        raise ValueError("Unsupported dataset. Choose from 'imagenet-1k'.")


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, _ = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx
