"""
!wget <ENTR URL FROM https://disk.yandex.ru/d/S8f03spLIA1wrw> -O data/celebA_train_500.zip
!cd data/ && unzip celebA_train_500.zip
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

RESCALE_SIZE = 100  # 100 #160 #224
DATASET_DIR = os.path.join(os.path.dirname(__file__), '../data/')
PATH_TO_MODELS = os.path.join(os.path.dirname(__file__), '../data/models')
os.makedirs(PATH_TO_MODELS, exist_ok=True)


class CustomDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """

    def __init__(self, files, labels, mode):
        super().__init__()
        # список файлов для загрузки
        self.files = files
        self.labels = labels
        # режим работы
        self.mode = mode

        self.len_ = len(self.files)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = np.array(cv2.imread(f'{DATASET_DIR}/celebA_train_500/celebA_imgs/{file}'))
        image = Image.fromarray(image[77:-41, 45:-50])
        image.load()
        return image

    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomRotation(degrees=30),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        aug = transforms.Compose([
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip()
        ])
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        # x = np.array(x / 255, dtype='float32')
        x = transform(x)
        if self.mode == 'train':
            x = aug(x)

        # label = self.labels[index]
        label = self.labels[self.files[index]]
        return x, label

    def _prepare_sample(self, image):
        #         padding = transforms.Pad(padding=((RESCALE_SIZE - image.size[1]) // 2, (RESCALE_SIZE - image.size[0]) // 2))
        #         image = padding(image)
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)


def main():
    # read attrs
    df_attrs = pd.read_csv(f"{DATASET_DIR}/celebA_train_500/celebA_anno.txt", header=None, sep=' ')
    df_attrs.columns = ['name_img', 'id']
    df_attrs.head()

    split = pd.read_csv(f"{DATASET_DIR}/celebA_train_500/celebA_train_split.txt", header=None, sep=' ')
    split.columns = ['name_img', 'class']

    train_files = list(split[split['class'] == 0]['name_img'])
    val_files = list(split[split['class'] == 1]['name_img'])
    test_files = list(split[split['class'] == 2]['name_img'])

    labels = dict(zip(df_attrs['name_img'], df_attrs['id']))

    # read images
    train_dataset = CustomDataset(train_files, labels, mode='train')
    val_dataset = CustomDataset(val_files, labels, mode='val')
    test_dataset = CustomDataset(test_files, labels, mode='test')

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set the seed for PyTorch
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_classes = 500

    model = models.resnet50(weights='IMAGENET1K_V1')

    # Freeze all layers except the last two residual blocks and the fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer2.parameters():
        param.requires_grad = True

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, n_classes),
        nn.BatchNorm1d(n_classes),
        nn.ReLU(inplace=True),
        nn.Linear(n_classes, n_classes),
    )

    loss_fn = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-8, betas=(0.95, 0.999))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-8,
                            betas=(0.95, 0.999))
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=24)

    n_epochs = 12
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in tqdm(range(n_epochs), total=n_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        average_loss = total_loss / len(train_loader)
        accuracy = correct / total_samples

        train_losses.append(average_loss)
        train_acc.append(accuracy)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {average_loss:.4f}, Train Accuracy: {accuracy:.4f}")

        # Validation phase
        model.eval()
        validation_loss = 0.0
        val_correct = 0
        val_total_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                validation_loss += val_loss.item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_labels).sum().item()
                val_total_samples += val_labels.size(0)

        avg_val_loss = validation_loss / len(val_loader)
        val_accuracy = val_correct / val_total_samples

        val_losses.append(avg_val_loss)
        val_acc.append(val_accuracy)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    torch.save(model, f'{PATH_TO_MODELS}/model_ce.bin')
    torch.save(model.state_dict(), f'{PATH_TO_MODELS}/model_ce_weights.bin')


if __name__ == '__main__':
    main()
