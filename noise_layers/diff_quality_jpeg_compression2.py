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

class DiffQFJpegCompression2(nn.Module):
    def __init__(self):
        super(DiffQFJpegCompression2, self).__init__()
        self.qualities = [10, 25, 50, 75, 90]
        self.key = generate_random_key()

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        res_noised_image = torch.zeros_like(noised_image)
        batch_size = noised_image.shape[0]

        for q in range(len(self.qualities)):
            quality = self.qualities[q]

            jpeg_folder_path = "./jpeg_" + str(quality) + "/" + self.key
            if not os.path.exists(jpeg_folder_path):
                os.makedirs(jpeg_folder_path)

            noised_image_q = noised_image[q*batch_size//len(self.qualities):(q+1)*batch_size//len(self.qualities)]
            container_img_copy = noised_image_q.clone()
            containers_ori = container_img_copy.detach().cpu().numpy()
            
            containers = np.transpose(containers_ori, (0, 2, 3, 1))
            N, _, _, _ = containers.shape
            # containers = (containers + 1) / 2 # transform range of containers from [-1, 1] to [0, 1]
            containers = (np.clip(containers, 0.0, 1.0)*255).astype(np.uint8)

            # containers = (np.clip(containers, 0.0, 1.0)*255).astype(np.uint8)

            for i in range(N):
                img = cv2.cvtColor(containers[i], cv2.COLOR_RGB2BGR)
                folder_imgs = jpeg_folder_path + "/jpg_" + str(i).zfill(2) + ".jpg"
                cv2.imwrite(folder_imgs, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

            containers_loaded = np.copy(containers)
            
            for i in range(N):
                folder_imgs = jpeg_folder_path + "/jpg_" + str(i).zfill(2) + ".jpg"
                img = cv2.imread(folder_imgs)
                containers_loaded[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # containers_loaded = np.transpose(containers_loaded, (0, 3, 1, 2)).astype(np.float32) / 255

            containers_loaded = containers_loaded.astype(np.float32) / 255
            # containers_loaded = containers_loaded * 2 - 1 # transform range of containers from [0, 1] to [-1, 1]
            containers_loaded = np.transpose(containers_loaded, (0, 3, 1, 2))

            container_gap = containers_loaded - containers_ori
            container_gap = torch.from_numpy(container_gap).float().cuda()

            container_img_noised_jpeg = noised_image_q + container_gap

            res_noised_image[q*batch_size//len(self.qualities):(q+1)*batch_size//len(self.qualities)] = container_img_noised_jpeg

        noised_and_cover[0] = res_noised_image

        return noised_and_cover
