import os

import torch
import torch.nn as nn

MODELS_DIR = os.path.join(os.path.dirname(__file__), '../data/models')


class ModelLandmark(nn.Module):
    def __init__(self, model, n_classes):
        super(ModelLandmark, self).__init__()
        self.model = model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def main():
    device = 'cpu'
    model_alignment_path = os.path.join(MODELS_DIR, 'model_alignment.bin')
    model_align_faces = torch.load(model_alignment_path, map_location=torch.device(device))
    print(model_align_faces)


if __name__ == '__main__':
    main()
