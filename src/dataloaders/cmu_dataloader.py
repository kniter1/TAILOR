# %%
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import json
import random
import time
import pickle


"""
CMU-MOSEI info
Train 16326 samples
Val 1871 samples
Test 4659 samples
CMU-MOSEI feature shapes
visual: (60, 35)
audio: (60, 74)
text: GLOVE->(60, 300)
label: (6) -> [happy, sad, anger, surprise, disgust, fear] 
    averaged from 3 annotators
unaligned:
text: (50, 300)
visual: (500, 35)
audio: (500, 74)    
"""

emotion_dict = {4:0, 5:1, 6:2, 7:3, 8:4, 9:5}
class AlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        data = torch.load(self.data_path)

        data = data[data_type]
        visual = data['src-visual']
        audio = data['src-audio']
        text = data['src-text']
        labels = data['tgt']      
        return visual, audio, text, labels
    
    
    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]

        text_mask = np.array(text_mask)

        return text, text_mask
    
    def _get_visual(self, index):
        visual = self.visual[index]
        visual_mask = [1] * visual.shape[0]
        visual_mask = np.array(visual_mask)

        return visual, visual_mask
    
    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * audio.shape[0]

        audio_mask =  np.array(audio_mask)

        return audio, audio_mask
    
    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] =  1

        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label


class UnAlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        label_data = torch.load(self.data_path)
        label_data = label_data[data_type]
        with open('/amax/cmy/mosei_senti_data_noalign.pkl', 'rb') as f:
            data = pickle.load(f)
        data = data[data_type]
        visual = data['vision']
        audio = data['audio']
        text = data['text']
        audio = np.array(audio)
        labels = label_data['tgt']      
        return visual, audio, text, labels
    
    
    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]

        text_mask = np.array(text_mask)

        return text, text_mask
    
    def _get_visual(self, index):
        visual = self.visual[index]
        visual_mask = [1] * 50
        visual_mask = np.array(visual_mask)

        return visual, visual_mask
    
    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * 50

        audio_mask =  np.array(audio_mask)

        return audio, audio_mask
    
    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] =  1

        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label