import torch
from models import *
from utils.utils import *
from utils.datasets import *
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

dataset_path = "D:/buaa/机器学习/测试脚本/core_coreless_test.txt"
anno_dir = "D:/buaa/机器学习/测试脚本/Anno_test/"
out_dir = "predicted_file_level1"

core_real = 0
coreless_real = 0
core_right = 0
coreless_right = 0

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def test(img_dir_path,anno_dir_path):
    global core_real,coreless_real,core_right,coreless_right
    core_reuslt = open(out_dir + "/det_test_带电芯充电宝.txt","w")
    coreless_reuslt = open(out_dir + "/det_test_不带电芯充电宝.txt","w")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet("config/yolov3.cfg", img_size=416).to(device)
    # model.load_state_dict(torch.load("/home/yg/Desktop/ckp-result/checkpoints/yolov3_ckpt_499.pth"))
    model.load_state_dict(torch.load("D:/buaa/机器学习/测试脚本/model.pth"))
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    with open(dataset_path) as f:
        for item in f.readlines():
            # print(item)
            item = item.replace("\n","")
            if len(item) < 4:
                continue
            img_path = img_dir_path + item + ".jpg"
            # img_path = "/home/yg/Desktop/workspace/yolo/PyTorch-YOLOv3/data/xray/samples/core_battery00005647.jpg"
            # Extract image as PyTorch tensor
            img = transforms.ToTensor()(Image.open(img_path))
            # Pad to square resolution
            img, _ = pad_to_square(img, 0)
            # Resize
            img = resize(img, 416)
            # input_imgs = Variable(img.type(Tensor))
            # print()
            # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            # x = torch.from_numpy(x)
            x = img
            x = Variable(x.type(Tensor))
            x = x.reshape((-1,3,416,416))
            with torch.no_grad():
                detections = model(x)
                detections = non_max_suppression(detections)
                for detection in detections:
                    if detection is not None:
                        img = Image.open(img_path)
                        x,y = img.size
                        detection = rescale_boxes(detection, 416, [y,x])
                        for d in detection:
                            type = int(d[-1])
                            # print(type)
                            if type == 0:
                                core_reuslt.write(item)
                                core_reuslt.write(" ")
                                core_reuslt.write(str(float(d[-2])))
                                core_reuslt.write(" ")
                                core_reuslt.write(str(float(d[0])))
                                core_reuslt.write(" ")
                                core_reuslt.write(str(float(d[1])))
                                core_reuslt.write(" ")
                                core_reuslt.write(str(float(d[2])))
                                core_reuslt.write(" ")
                                core_reuslt.write(str(float(d[3])))
                                core_reuslt.write("\n")
                            else:
                                coreless_reuslt.write(item)
                                coreless_reuslt.write(" ")
                                coreless_reuslt.write(str(float(d[-2])))
                                coreless_reuslt.write(" ")
                                coreless_reuslt.write(str(float(d[0])))
                                coreless_reuslt.write(" ")
                                coreless_reuslt.write(str(float(d[1])))
                                coreless_reuslt.write(" ")
                                coreless_reuslt.write(str(float(d[2])))
                                coreless_reuslt.write(" ")
                                coreless_reuslt.write(str(float(d[3])))
                                coreless_reuslt.write("\n")
                                # coreless_reuslt.write(item,float(d[-2]),float(d[0]),float(d[1]),float(d[2]),float(d[3]))


    core_reuslt.close()
    coreless_reuslt.close()
if __name__ == "__main__":
    test("D:/buaa/机器学习/测试脚本/Image_test/","D:/buaa/机器学习/测试脚本/Anno_test/")