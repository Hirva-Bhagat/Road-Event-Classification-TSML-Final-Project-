import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from os.path import *
import numpy as np
import random
from glob import glob
import csv
from utils import load_value_file

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.png'.format(i))
        #print(image_path)
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(image_path)
            #print("here")
            return video

    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def load_annotation_data(data_file_path, fold):
    database = {}
    data_file_path = os.path.join(data_file_path, 'fold%d.csv'%fold)
    print('Load from %s'%data_file_path)
    with open(data_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            value = {}
            value['subset'] = row[3]
            value['label'] = row[1]
            value['n_frames'] = int(row[2])
            database[row[0]] = value
    return database

def get_class_labels():
#### define the labels map
    class_labels_map = {}
    class_labels_map['end_action'] = 0
    class_labels_map['lchange'] = 1
    class_labels_map['lturn'] = 2
    class_labels_map['rchange'] = 3
    class_labels_map['rturn'] = 4
    return class_labels_map

def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data.items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['label']
            video_names.append(key)    ### key = 'rturn/20141220_154451_747_897'
            annotations.append(value)

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video, end_second,
                 sample_duration, fold):

    data = load_annotation_data(annotation_path, fold)
    #print("ALL DATA")
    #print(data)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    #print(video_names)
    class_to_idx = get_class_labels()
    #print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 100 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i]).replace('\\','/')
        if not os.path.exists(video_path):
            print('File does not exists: %s'%video_path)
            continue

#        n_frames = annotations[i]['n_frames']
        # count in the dir
        l = os.listdir(video_path)
        # If there are other files (e.g. original videos) besides the images in the folder, please abstract.
        n_frames = len(l)

        if n_frames < 16 + 25*(end_second-1):
            print('Video is too short: %s'%video_path)
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,

            'video_id': video_names[i].split('\\')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(0, n_frames ))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                for j in range(0, n_samples_for_each_video):
                    sample['frame_indices'] = list(range(0, n_frames))
                    sample_j = copy.deepcopy(sample)
                    dataset.append(sample_j)
    #print(dataset[1:10])
    return dataset, idx_to_class


class Without_gaze(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold, 
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        h_flip = False

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        
        if self.horizontal_flip is not None:
            p = random.random()
            if p < 0.5:
                h_flip = True
                clip = [self.horizontal_flip(img) for img in clip]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        if (h_flip == True) and (target != 0):
            if target == 1:
                target = 3
            elif target == 3:
                target = 1
            elif target == 2:
                target = 4
            elif target == 4:
                target = 2

        return clip, target
    def __len__(self):
        return len(self.data)

class Brain4cars_Outside(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold,
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=5,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is an image.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        h_flip = False

        if self.temporal_transform is not None:
            frame_indices,target_idc = self.temporal_transform(frame_indices)
        #print(path)
        clip = self.loader(path, frame_indices)
        target = self.loader(path, target_idc)

        if self.horizontal_flip is not None:
            p = random.random()
            if p < 0.5:
                clip = [self.horizontal_flip(img) for img in clip]
                target = [self.horizontal_flip(img) for img in target]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        #print("clip:")
        #print(path)
        #print(clip)
        if clip:
            clip = torch.stack(clip,0)
        #else:
        #    clip = torch.tensor([])
        #clip = torch.stack(clip, 0)

        print(target_idc)
        if self.target_transform is not None:
            target = [self.target_transform(img) for img in target]
        if target:
            print(target)
            target = torch.stack(target, 0).permute(1, 0, 2, 3).squeeze()
        #else:
        #    target = torch.tensor([])
        #target = torch.stack(target, 0).permute(1, 0, 2, 3).squeeze()
            
        #print(target)
        return clip, target
    def __len__(self):
        return len(self.data)
    
class With_gaze(data.Dataset):
    def __init__(self,
                 root_path,
                 gaze_maps_path,
                 annotation_path,
                 subset,
                 nfold, 
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)

        self.gaze_maps_path=gaze_maps_path
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        h_flip = False
        base=os.path.split(path)[1]
        label=os.path.split(os.path.split(path)[0])[1]
       
        path2=os.path.join(self.gaze_maps_path,label,base).replace("\\","/")
                                                                   
        #print(path2)
        
        #path2=gaze_maps_path+fname
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        clip2 = self.loader(path2, frame_indices)
        
        
        
        if self.horizontal_flip is not None:
            p = random.random()
            if p < 0.5:
                h_flip = True
                clip = [self.horizontal_flip(img) for img in clip]
                clip2 = [self.horizontal_flip(img) for img in clip2]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip2 = [self.spatial_transform(img) for img in clip2]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip2 = torch.stack(clip2, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        if (h_flip == True) and (target != 0):
            if target == 1:
                target = 3
            elif target == 3:
                target = 1
            elif target == 2:
                target = 4
            elif target == 4:
                target = 2

        return (clip,clip2), target
    def __len__(self):
        return len(self.data)
    
