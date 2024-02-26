import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from facenet_pytorch import MTCNN

MODELS_DIR = os.path.join(os.path.dirname(__file__), '../data/models')
PATH_TO_IMAGE = os.path.join(
    os.path.dirname(__file__),
    '../data/example.webp'
)


class ModelLandmark(nn.Module):
    def __init__(self, model, n_classes):
        super(ModelLandmark, self).__init__()
        self.model = model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ModelWithoutLastLayer(nn.Module):
    def __init__(self, model):
        super(ModelWithoutLastLayer, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.features(x)


def main():
    device = 'cpu'
    model_alignment_path = os.path.join(MODELS_DIR, 'model_alignment.bin')
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=200)
    model_align_faces = torch.load(model_alignment_path, map_location=torch.device(device))
    model_embeddings = ModelWithoutLastLayer(
        torch.load(os.path.join(MODELS_DIR, 'model_ce.bin'), map_location=torch.device(device)))

    def detect_faces(image):
        img = Image.fromarray(image)

        boxes, _ = mtcnn.detect(img)
        boxes = np.array(boxes, dtype=np.float32)

        crop_image_list = []

        if boxes.shape:
            for box in boxes:
                x1 = int(box[1])
                x2 = int(box[3])
                y1 = int(box[0])
                y2 = int(box[2])
                crop_img = image[x1:x2, y1:y2:]
                crop_image_list.append(crop_img)

        return crop_image_list

    def align_face(crop_image_list):
        def get_landmarks(img):
            img = TF.to_tensor(img)
            img = TF.resize(img, (224, 224))
            img = TF.normalize(img, [0.5], [0.5])
            img = img[None, :, :, :]

            model_align_faces.eval()
            with torch.no_grad():
                inputs = img
                outputs = model_align_faces(inputs)
                outputs = (outputs + 0.5) * 224
                outputs = outputs.view(-1, 68, 2)
            return outputs[0].numpy()

        def get_align_img(img, landmarks):
            def find_angle(eyePoints):
                leftEyeX, leftEyeY = eyePoints[0]
                rightEyeX, rightEyeY = eyePoints[1]
                angle = math.atan((leftEyeY - rightEyeY) / (leftEyeX - rightEyeX)) * (180 / math.pi)
                return angle

            def rotate_image(image, angle):
                image_center = tuple(np.array(image.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
                return result

            eyePoints = (landmarks[39], landmarks[42])
            angle = find_angle(eyePoints)
            align_img = rotate_image(img, angle)
            return align_img

        align_img_list = []
        for img in crop_image_list:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            landmarks = get_landmarks(gray_img)
            final_align_img = get_align_img(img, landmarks)
            align_img_list.append(final_align_img)
        return align_img_list

    def compute_embeddings(align_img_list):
        model_embeddings.eval()
        embeddings = []
        with torch.no_grad():
            for img in align_img_list:
                img = TF.to_tensor(img)
                img = TF.resize(img, (224, 224))
                img = TF.normalize(img, [0.5], [0.5])
                img = img[None, :, :, :]

                img = img.to(device)
                outputs = model_embeddings(img)
                if len(outputs) > 1:
                    outputs = outputs[1]
                outputs = outputs.detach().cpu().numpy()
                outputs = [list(i.flatten()) for i in outputs]
                embeddings.extend(outputs)

        return embeddings

    image = cv2.imread(PATH_TO_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_image_list = detect_faces(image)
    align_img_list = align_face(crop_image_list)
    embeddings = compute_embeddings(align_img_list)
    print(embeddings)


if __name__ == '__main__':
    main()
