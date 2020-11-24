import torchvision.transforms as transforms
import torch
from torchvision import models
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import cv2
import tqdm
import os


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()
root_source = "bird_dataset"
root_target = "croped_bird_dataset"


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def predict_boxes(img, threshold):
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def get_bird_box(boxes, classes):
    for i, c in enumerate(classes):
        if c == "bird":
            box = boxes[i]
            x = min(int(box[1][1]), int(box[0][1]))
            y = min(int(box[1][0]), int(box[0][0]))
            h = abs(int(box[1][1] - box[0][1]))
            w = abs(int(box[1][0] - box[0][0]))
            return x, y, h, w


def crop_bird(img_path, threshold):
    img = Image.open(img_path)
    img = transforms.ToTensor()(img).to(device)
    pred_boxes, pred_class = predict_boxes(img, threshold)
    x, y, h, w = get_bird_box(pred_boxes, pred_class)
    img = transforms.functional.crop(img, x,y , h, w)
    return img


def showim(image):
    image = image.permute(1,2,0).cpu().numpy()
    plt.imshow(image)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


    
if __name__ == '__main__':
    errors = []
    for subroot in os.listdir(root_source):
        create_folder(os.path.join(root_target, subroot))
        for category in os.listdir(os.path.join(root_source, subroot)):
            create_folder(os.path.join(root_target, subroot, category))
            images = os.listdir(os.path.join(root_source, subroot, category))
            print(f"transfering : {subroot}/{category}")
            for image in tqdm.tqdm(images):
                source_im_path = os.path.join(root_source, subroot, category, image)
                target_im_path = os.path.join(root_target, subroot, category, image)
                try:
                    img = crop_bird(source_im_path, threshold=0.8)
                except:
                    img = Image.open(source_im_path)
                    img = transforms.ToTensor()(img)
                    print(f"ERROR : {source_im_path}")
                    errors.append(source_im_path)
                save_image(img, target_im_path)