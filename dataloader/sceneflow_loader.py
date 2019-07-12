# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:13:35 2019

@author: Administrator
"""

import numpy as np
import random
import cv2
import os
from utils import readPFM
import dataloader.listflowfile as lt


class DataLoaderSceneFlow(object):
    def __init__(self, data_path, batch_size, patch_size=(256, 512), max_disp=129, val_size=500):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_disp = max_disp
        self.val_size = val_size
        all_left_img, all_right_img, all_left_disp, \
        test_left_img, test_right_img, test_left_disp = lt.get_sceneflow_img(data_path)
        self.data = list(zip(all_left_img, all_right_img, all_left_disp))
        self.train_size = len(all_left_img) - self.val_size

    def generator(self, is_training=True):

        train_left = [x[0] for x in self.data[self.val_size:]]
        train_right = [x[1] for x in self.data[self.val_size:]]
        train_labels = [x[2] for x in self.data[self.val_size:]]

        val_left = [x[0] for x in self.data[:self.val_size]]
        val_right = [x[1] for x in self.data[:self.val_size]]
        val_labels = [x[2] for x in self.data[:self.val_size]]

        index = [i for i in range(len(train_labels))]  # 8664*7/8
        random.shuffle(index)
        shuffled_labels = []
        shuffled_left_data = []
        shuffled_right_data = []

        for i in index:
            shuffled_left_data.append(train_left[i])
            shuffled_right_data.append(train_right[i])
            shuffled_labels.append(train_labels[i])
        if is_training:
            for j in range(len(train_labels) // self.batch_size):
                left, right, label = self.load_batch(shuffled_left_data[j * self.batch_size: (j + 1) * self.batch_size],
                                                     shuffled_right_data[
                                                     j * self.batch_size: (j + 1) * self.batch_size],
                                                     shuffled_labels[j * self.batch_size: (j + 1) * self.batch_size],
                                                     is_training)
                left = np.array(left)
                right = np.array(right)
                label = np.array(label)
                yield left, right, label
        else:
            for j in range(self.val_size // self.batch_size):
                left, right, label = self.load_batch(val_left[j * self.batch_size: (j + 1) * self.batch_size],
                                                     val_right[j * self.batch_size: (j + 1) * self.batch_size],
                                                     val_labels[j * self.batch_size: (j + 1) * self.batch_size],
                                                     is_training)
                left = np.array(left)
                right = np.array(right)
                label = np.array(label)
                yield left, right, label

    def load_batch(self, left, right, labels, is_training):
        batch_left = []
        batch_right = []
        batch_label = []
        for x, y, z in zip(left, right, labels):
            # print(x)
            if is_training:
                crop_x = random.randint(0, 540 - 1 - self.patch_size[0])
                crop_y = random.randint(0, 960 - 1 - self.patch_size[1])
            else:
                crop_x = (540 - 1 - self.patch_size[0]) // 2
                crop_y = (960 - 1 - self.patch_size[1]) // 2

            x = cv2.imread(x)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x[crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1], :]
            x = self.mean_std(x)
            batch_left.append(x)

            y = cv2.imread(y)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            y = y[crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1], :]
            y = self.mean_std(y)
            batch_right.append(y)

            z = readPFM(z)
            # z = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
            z = z[crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1]]
            z[z > (self.max_disp - 1)] = self.max_disp - 1
            batch_label.append(z)
        return batch_left, batch_right, batch_label

    @staticmethod
    def mean_std(inputs):
        inputs = np.float32(inputs) / 255.
        inputs[:, :, 0] -= 0.485
        inputs[:, :, 0] /= 0.229
        inputs[:, :, 1] -= 0.456
        inputs[:, :, 1] /= 0.224
        inputs[:, :, 2] -= 0.406
        inputs[:, :, 2] /= 0.225
        return inputs


if __name__ == '__main__':
    loader = DataLoaderSceneFlow(data_path='../dataset/', batch_size=10, max_disp=192)
