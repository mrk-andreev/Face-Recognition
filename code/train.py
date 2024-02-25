import math
import os
import random
import xml.etree.ElementTree as ET

import cv2
import imutils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
from tqdm import tqdm

PATH_TO_DATA = os.path.join(os.path.dirname(__file__), '../data/ibug_300W_large_face_landmark_dataset')
PATH_TO_MODELS = os.path.join(os.path.dirname(__file__), '../models')


class FaceLandmarksDataset(Dataset):

    def __init__(self, transform=None, mode=None, val_split=None):
        val_split = val_split or None

        tree = ET.parse(f"{PATH_TO_DATA}/labels_ibug_300W_train.xml")
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = PATH_TO_DATA
        self.mode = mode

        for filename in root[2]:
            if self.mode == 'val':
                if filename.attrib['file'] in val_split:
                    self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

                    self.crops.append(filename[0].attrib)

                    landmark = []
                    for num in range(68):
                        x_coordinate = int(filename[0][num].attrib['x'])
                        y_coordinate = int(filename[0][num].attrib['y'])
                        landmark.append([x_coordinate, y_coordinate])
                    self.landmarks.append(landmark)

            else:
                if filename.attrib['file'] not in val_split:
                    self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

                    self.crops.append(filename[0].attrib)

                    landmark = []
                    for num in range(68):
                        x_coordinate = int(filename[0][num].attrib['x'])
                        y_coordinate = int(filename[0][num].attrib['y'])
                        landmark.append([x_coordinate, y_coordinate])
                    self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # print(self.image_filenames[index])
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks


class Transforms():
    def __init__(self, mode):
        self.mode = mode

    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+math.cos(math.radians(angle)), -math.sin(math.radians(angle))],
            [+math.sin(math.radians(angle)), +math.cos(math.radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        if self.mode == 'train':
            image, landmarks = self.color_jitter(image, landmarks)
            image, landmarks = self.rotate(image, landmarks, angle=10)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        # image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, landmarks


class ModelLandmark(nn.Module):
    def __init__(self, model, n_classes):
        super(ModelLandmark, self).__init__()
        self.model = model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def train_alignment():
    """
    Dataset:
      * https://ibug.doc.ic.ac.uk/resources/300-W/
      * http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
    """

    tree = ET.parse(f"{PATH_TO_DATA}/labels_ibug_300W_train.xml")
    root = tree.getroot()
    all_images = []
    for filename in root[2]:
        # print(filename.attrib['file'])
        all_images.append(filename.attrib['file'])

    val_split = set(random.sample(all_images, int(len(all_images) * 0.1)))
    dataset = FaceLandmarksDataset(Transforms(mode='val'), val_split=val_split)

    train_dataset = FaceLandmarksDataset(Transforms(mode='train'), val_split=val_split, mode='train')
    valid_dataset = FaceLandmarksDataset(Transforms(mode='val'), val_split=val_split, mode='val')

    # shuffle and batch the datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=False, drop_last=True)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Freeze all layers except the last two residual blocks and the fully connected layer
    for param in model.parameters():
        param.requires_grad = True

    model = ModelLandmark(model, 68 * 2)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-8,
                            betas=(0.95, 0.999))

    n_epochs = 10
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Use device {device}')

    model = model.to(device)

    train_losses = []
    val_losses = []
    loss_min = np.inf

    for epoch in tqdm(range(n_epochs), total=n_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for step in range(1, len(train_loader) + 1):
            inputs, landmarks = next(iter(train_loader))
            inputs = inputs.to(device)
            landmarks = landmarks.view(landmarks.size(0), -1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += landmarks.size(0)

        average_loss = total_loss / len(train_loader)

        train_losses.append(average_loss)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {average_loss:.4f}")

        # Validation phase
        model.eval()
        validation_loss = 0.0
        val_total_samples = 0

        with torch.no_grad():
            for step in range(1, len(valid_loader) + 1):
                inputs, landmarks = next(iter(valid_loader))
                inputs = inputs.to(device)
                landmarks = landmarks.view(landmarks.size(0), -1).to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, landmarks)

                validation_loss += val_loss.item()
                val_total_samples += landmarks.size(0)

        avg_val_loss = validation_loss / len(valid_loader)

        val_losses.append(avg_val_loss)

        if avg_val_loss < loss_min:
            loss_min = avg_val_loss
            torch.save(model.state_dict(), f'{PATH_TO_MODELS}/face_landmarks.pth')
        print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {avg_val_loss:.4f}")

    torch.save(model, f'{PATH_TO_MODELS}/model_alignment.bin')
    torch.save(model.state_dict(), f'{PATH_TO_MODELS}/model_weights_alignment.bin')


def main():
    train_alignment()


if __name__ == '__main__':
    main()
