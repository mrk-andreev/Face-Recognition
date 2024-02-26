import argparse
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from PIL import Image
from facenet_pytorch import MTCNN

MODELS_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), '../data/models')


class ModelLandmark(nn.Module):
    def __init__(self, model, n_classes):
        super(ModelLandmark, self).__init__()
        self.model = model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class FaceDetector:
    def __init__(self, device, min_face_size=200):
        self._mtcnn = MTCNN(keep_all=True, device=device, min_face_size=min_face_size)

    def detect_faces(self, image):
        boxes, _ = self._mtcnn.detect(Image.fromarray(image))
        boxes = np.array(boxes, dtype=np.float32)

        faces_crops = []
        if boxes.shape:
            for box in boxes:
                x1 = int(box[1])
                x2 = int(box[3])
                y1 = int(box[0])
                y2 = int(box[2])
                crop_img = image[x1:x2, y1:y2:]
                faces_crops.append(crop_img)

        return faces_crops


class FaceAligner:
    def __init__(self, model_alignment_path, device):
        self._model_align_faces = torch.load(model_alignment_path, map_location=torch.device(device))
        self._model_align_faces.eval()

    def align_faces(self, images):
        aligned_images = []
        for image in images:
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            landmarks = self._get_landmarks(gray_img)
            aligned_image = self._get_align_img(image, landmarks)
            aligned_images.append(aligned_image)
        return aligned_images

    def _get_landmarks(self, img):
        img = tf.to_tensor(img)
        img = tf.resize(img, [224, 224])
        img = tf.normalize(img, [0.5], [0.5])
        img = img[None, :, :, :]

        with torch.no_grad():
            inputs = img
            outputs = self._model_align_faces(inputs)
            outputs = (outputs + 0.5) * 224
            outputs = outputs.view(-1, 68, 2)
        return outputs[0].numpy()

    @classmethod
    def _find_angle(cls, eye_points):
        (left_eye_x, left_eye_y), (right_eye_x, right_eye_y) = eye_points
        return math.atan((left_eye_y - right_eye_y) / (left_eye_x - right_eye_x)) * (180 / math.pi)

    @classmethod
    def _rotate_image(cls, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _get_align_img(self, img, landmarks):
        eye_points = (landmarks[39], landmarks[42])
        angle = self._find_angle(eye_points)
        align_img = self._rotate_image(img, angle)
        return align_img


class ModelWithoutLastLayer(nn.Module):
    def __init__(self, model):
        super(ModelWithoutLastLayer, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.features(x)


class FaceEmbedding:
    def __init__(self, model_embeddings_path, device):
        self._model_embeddings = ModelWithoutLastLayer(
            torch.load(model_embeddings_path, map_location=torch.device(device)))
        self._device = device
        self._model_embeddings.eval()

    def compute_embeddings(self, align_img_list):
        embeddings = []
        with torch.no_grad():
            for img in align_img_list:
                img = tf.to_tensor(img)
                model_input_size = [224, 224]
                img = tf.resize(img, model_input_size)
                normalize_mean = [0.5]
                normalize_std = [0.5]
                img = tf.normalize(img, normalize_mean, normalize_std)
                img = img[None, :, :, :]

                img = img.to(self._device)
                outputs = self._model_embeddings(img)
                if len(outputs) > 1:
                    outputs = outputs[1]
                outputs = outputs.detach().cpu().numpy()
                outputs = [list(i.flatten()) for i in outputs]
                embeddings.extend(outputs)

        return embeddings


def main():
    parser = argparse.ArgumentParser(
        prog='Face-Recognition Predict Pipeline'
    )
    parser.add_argument('--input_image_path', required=True)
    parser.add_argument(
        '--model_alignment_path',
        default=os.path.join(MODELS_DEFAULT_DIR, 'model_alignment.bin')
    )
    parser.add_argument(
        '--model_embeddings_path',
        default=os.path.join(MODELS_DEFAULT_DIR, 'model_ce.bin')
    )
    parser.add_argument(
        '--device',
        default='cpu'
    )
    parser.add_argument(
        '--min_face_size',
        default=200,
        type=int
    )
    args = parser.parse_args()

    device = args.device
    face_detector = FaceDetector(device, min_face_size=args.min_face_size)
    face_aligner = FaceAligner(args.model_alignment_path, device)
    face_embedding = FaceEmbedding(args.model_embeddings_path, device)
    input_image_path = args.input_image_path

    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_image_list = face_detector.detect_faces(image)
    align_img_list = face_aligner.align_faces(crop_image_list)
    embeddings = face_embedding.compute_embeddings(align_img_list)
    print(embeddings)


if __name__ == '__main__':
    main()
