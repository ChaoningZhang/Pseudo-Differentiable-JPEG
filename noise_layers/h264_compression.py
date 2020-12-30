import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import random
import string

def generate_random_key(l=30):
    s=string.ascii_lowercase+string.digits
    return ''.join(random.sample(s,l))

class H264(nn.Module):
    def __init__(self):
        super(H264, self).__init__()
        self.key = generate_random_key()

    def forward(self, noised_and_cover):
        video_folder_path = "./h264/" + self.key
        if not os.path.exists(video_folder_path):
            os.makedirs(video_folder_path)
        fourcc = cv2.VideoWriter_fourcc(*'hvc1') 
        video = cv2.VideoWriter(video_folder_path + '/video.mov', fourcc, 30, (128, 128))

        noised_image = noised_and_cover[0]

        container_img_copy = noised_image.clone()
        containers_ori = container_img_copy.detach().cpu().numpy()
        
        containers = np.transpose(containers_ori, (0, 2, 3, 1))
        N, _, _, _ = containers.shape
        # containers = (containers + 1) / 2 # transform range of containers from [-1, 1] to [0, 1]
        containers = (np.clip(containers, 0.0, 1.0)*255).astype(np.uint8)

        # containers = (np.clip(containers, 0.0, 1.0)*255).astype(np.uint8)

        for i in range(N):
            img = containers[i]
            video.write(img)

        cv2.destroyAllWindows()
        video.release()

        containers_loaded = np.copy(containers)
        video_saved = cv2.VideoCapture(video_folder_path + '/video.mov')

        for i in range(N):
            _, img = video_saved.read()
            containers_loaded[i] = img

        # containers_loaded = np.transpose(containers_loaded, (0, 3, 1, 2)).astype(np.float32) / 255

        containers_loaded = containers_loaded.astype(np.float32) / 255
        # containers_loaded = containers_loaded * 2 - 1 # transform range of containers from [0, 1] to [-1, 1]
        containers_loaded = np.transpose(containers_loaded, (0, 3, 1, 2))

        container_gap = containers_loaded - containers_ori
        container_gap = torch.from_numpy(container_gap).float().cuda()

        container_img_noised_mpeg = noised_image + container_gap

        noised_and_cover[0] = container_img_noised_mpeg

        return noised_and_cover