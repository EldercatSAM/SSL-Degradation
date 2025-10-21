import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T

class EncoderwithProjection(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone
        base_encoder = models.__dict__['resnet50'](pretrained=False)
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])

        self.input_norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.input_norm(x)
        x = self.encoder(x)
        return x

    def get_embeddings(self, x):
        x = self.encoder(x)
        B, d, h, w = x.shape
        x = x.reshape(B, d, h*w).permute(0,2,1)
        return [None, None, None, None], [None, None, None, x]


