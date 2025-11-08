import os
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize
import random
from random import sample, shuffle
from turtle import clear
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, dataset_dir, annotation_lines, input_shape, num_classes, epoch_length, mosaic, train, mosaic_ratio=0.7, use_dce=False):
        super(YoloDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.train = train
        self.mosaic_ratio = mosaic_ratio
        self.use_dce = use_dce  # Fixed: was self.use_dc = use_dc

        self.image_dir = 'VOC_Clean/train/CleanImages' if self.train else 'VOC_Clean/test/CleanImages'

        # self.dataset_dict = {"Haze": "HazyImages", "Rain": "RainyImages", "Snow": "SnowyImages"}

        self.epoch_now = -1
        self.length = len(self.annotation_lines)
        
        # Initialize clearimage_lines for mosaic augmentation
        if self.train:
            self.clearimage_lines = [line.replace('HazyImages', 'CleanImages').replace('RainyImages', 'CleanImages').replace('SnowyImages', 'CleanImages') 
                                   for line in self.annotation_lines]
        else:
            self.clearimage_lines = self.annotation_lines

        # CLIP preprocessing transform for context images
        if self.use_dce:
            self.context_transform = Compose([
                ToPILImage(),
                Resize(224, interpolation=Image.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

    def __len__(self):
        return self.length

    def _get_context_pair(self, annotation_line):
        """Get a random context pair from the same degradation type"""
        # Get all lines with same degradation type (simplified - you may need to adapt)
        # For now, randomly select another line
        context_idx = random.randint(0, self.length - 1)
        context_line = self.annotation_lines[context_idx]
        
        # Parse context line
        line_content = context_line.split()
        degrad_context_path = os.path.join(self.dataset_dir, line_content[0])
        clean_context_path = os.path.join(self.dataset_dir, self.image_dir, line_content[0].split('/')[-1])
        
        # Load and preprocess context images
        degrad_context_img = Image.open(degrad_context_path).convert('RGB')
        clean_context_img = Image.open(clean_context_path).convert('RGB')
        
        # Convert to numpy for transform
        degrad_context_np = np.array(degrad_context_img)
        clean_context_np = np.array(clean_context_img)
        
        # Apply CLIP transform
        degrad_context = self.context_transform(degrad_context_np)
        clean_context = self.context_transform(clean_context_np)
        
        return degrad_context, clean_context


    def __getitem__(self, index):
        index = index % self.length

        if self.mosaic:
            if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
                lines = sample(self.annotation_lines, 3)
                lines.append(self.annotation_lines[index])
                clearlines = sample(self.clearimage_lines, 3)
                clearlines.append(self.clearimage_lines[index])
                shuffle(lines)
                shuffle(clearlines)
                image, box, clearimg = self.get_random_data_with_Mosaic(lines, self.input_shape, clearlines)
            else:
                image, box, clearimg = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        else:
            image, box, clearimg = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        clearimg = np.transpose(preprocess_input(np.array(clearimg, dtype=np.float32)), (2, 0, 1))
        
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        
        if self.use_dce:
            degrad_context, clean_context = self._get_context_pair(self.annotation_lines[index])
            return image, box, clearimg, degrad_context, clean_context
        else:
            return image, box, clearimg

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()

        image_path = os.path.join(self.dataset_dir, line[0])
        image = Image.open(image_path)
        image = cvtColor(image)

        clean_path = os.path.join(self.dataset_dir, self.image_dir, image_path.split('/')[-1])
        # print(image_path, '||', clean_path)
        clearimg = Image.open(clean_path)
        clearimg = cvtColor(clearimg)

        iw, ih = image.size
        h, w = input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            '''clear'''
            clearimg = clearimg.resize((nw, nh), Image.BICUBIC)
            new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
            new_clearimg.paste(clearimg, (dx, dy))
            clear_image_data = np.array(new_clearimg, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box, clear_image_data

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        clearimg = clearimg.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        '''clear'''
        new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
        new_clearimg.paste(clearimg, (dx, dy))
        clearimg = new_clearimg

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            clearimg = clearimg.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        clear_image_data = np.array(clearimg, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        hue1, sat1, val1 = cv2.split(cv2.cvtColor(clear_image_data, cv2.COLOR_RGB2HSV))
        dtype1 = clear_image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        x1 = np.arange(0, 256, dtype=r.dtype)
        lut_hue1 = ((x1 * r[0]) % 180).astype(dtype)
        lut_sat1 = np.clip(x1 * r[1], 0, 255).astype(dtype)
        lut_val1 = np.clip(x1 * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        clear_image_data = cv2.merge((cv2.LUT(hue1, lut_hue1), cv2.LUT(sat1, lut_sat1), cv2.LUT(val1, lut_val1)))
        clear_image_data = cv2.cvtColor(clear_image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box, clear_image_data

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:

            line_content = line.split()

            image = Image.open(line_content[0])
            image = cvtColor(image)

            iw, ih = image.size

            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    clearimg = []
    degrad_contexts = []
    clean_contexts = []
    
    has_context = len(batch[0]) == 5  # Check if context pairs are present
    
    for item in batch:
        images.append(item[0])
        bboxes.append(item[1])
        clearimg.append(item[2])
        if has_context:
            degrad_contexts.append(item[3])
            clean_contexts.append(item[4])
    
    images = np.array(images)
    clearimg = np.array(clearimg)
    
    if has_context:
        degrad_contexts = torch.stack(degrad_contexts, dim=0)
        clean_contexts = torch.stack(clean_contexts, dim=0)
        return images, bboxes, clearimg, degrad_contexts, clean_contexts
    else:
        return images, bboxes, clearimg
