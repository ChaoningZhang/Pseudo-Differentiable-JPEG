import os
import cv2
from moviepy.editor import *
import numpy as np
import torch
from  torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

def read_video(video_path):
    # video_path = "/workspace/smb_ssd_data1/Data/oops_dataset/oops_video/train/27 Hilarious Cooking Fail Nominees - FailArmy Hall of Fame (July 2017)11.mp4"

    # sample = cv2.VideoCapture(video_path)
    # length = int(sample.get(cv2.CAP_PROP_FRAME_COUNT))

    sample = VideoFileClip(video_path)
    length = sample.reader.nframes

    frames = np.zeros((length, 128, 128, 3), dtype=np.uint8)

    # for i in range(length):
    #     success, fr = sample.read()
    #     H, W, _ = fr.shape
    #     S = min(H, W)
    #     fr = cv2.resize(fr[H//2-S//2:H//2+S//2, W//2-S//2:W//2+S//2], (128, 128))
    #     frames[i] = fr

    for i, fr in enumerate(sample.iter_frames()):
        H, W, _ = fr.shape
        S = min(H, W)
        fr = cv2.resize(fr[H//2-S//2:H//2+S//2, W//2-S//2:W//2+S//2], (128, 128))
        frames[i] = fr

    # sample.release()
    # cv2.destroyAllWindows()

    return frames, length

class OOPSDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.dataset = os.listdir(data_path)
        self.dataset.sort()

        # # Filter problematic entries
        # bad_entries = []
        # print(len(self.dataset))
        # for i in range(len(self.dataset)):
        #     if i % 1000 == 0:
        #         print(i)
        #     s = self.dataset[i]
        #     if 'ata Party' in s:
        #         bad_entries.append(s)
        #     else:
        #         try:
        #             sample = VideoFileClip(self.data_path + "/" + s)
        #         except OSError:
        #             print("problematic video")
        #             bad_entries.append(s)

        # for entry in bad_entries:
        #     os.remove(self.data_path + "/" + entry)

        # for s in self.dataset:
        #     if 'ata Party' in s:
        #         print(s)
        #         import pdb; pdb.set_trace()

        self.transform = transform

    def __getitem__(self, idx):
        frames, l = read_video(self.data_path + "/" + self.dataset[idx])

        if (l < 32):
            frames = frames.transpose(0, 3, 1, 2).astype(np.float32)
            frames = torch.from_numpy(frames / 255)
        else:
            start_ind = np.random.randint(l - 31)
            frames = frames[start_ind:start_ind + 32]
            frames = frames.transpose(0, 3, 1, 2).astype(np.float32)
            frames = torch.from_numpy(frames / 255)
        
        return (frames, 0)

    def __len__(self):
        return len(self.dataset)