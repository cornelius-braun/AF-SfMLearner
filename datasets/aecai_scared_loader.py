import os
from PIL import Image, ImageFile
from glob import glob
from torch.utils.data import Dataset
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

class SCAREDLoader(Dataset):
    def __init__(self, root_dir, split, transform=None, n_val_samples=600):
        if split =='train' or split=='val':
            self.left_paths_all= sorted(glob('{}/train/**/**/frames_resized/left/*.png'.format(root_dir)))
            self.right_paths_all= sorted(glob('{}/train/**/**/frames_resized/right/*.png'.format(root_dir)))
            
            if split=='train':
                self.left_paths= self.left_paths_all[0:len(self.left_paths_all)-n_val_samples]
                self.right_paths= self.right_paths_all[0:len(self.right_paths_all)-n_val_samples]
    
            elif split=='val':
                self.left_paths= self.left_paths_all[len(self.left_paths_all)-n_val_samples::]
                self.right_paths= self.right_paths_all[len(self.right_paths_all)-n_val_samples::]

        elif split=='test':
            self.left_paths= sorted(glob('{}/test/**/**/frames_resized/left/*.png'.format(root_dir)))
            self.right_paths= sorted(glob('{}/test/**/**/frames_resized/right/*.png'.format(root_dir)))
            
        
        self.transform = transform

    def __getitem__(self, idx):
        #print(self.left_paths[idx])
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return len(self.left_paths)


class SCAREDTestLoader(Dataset):
    def __init__(self, dataset, keyframe, transform=None):
        
        root_dir = "/home/cornelius/Documents/Bibliothek/ICL/IndependentStudyOption/scared_accurate_3ds" #r'/mnt/398441BC350EAA68/Data/CV/endoscopic_challenge_1280x1024'
        
        self.left_paths_all= sorted(glob('{}/dataset{}_keyframe{}/left_disp_rect.png'.format(root_dir, dataset, keyframe)))
        self.right_paths_all= sorted(glob('{}/dataset{}_keyframe{}/right_disp_rect.png'.format(root_dir, dataset, keyframe)))
        if len(self.right_paths_all)!= len(self.left_paths_all):
            self.right_paths_all=self.left_paths_all

        self.transform = transform
        print("loaded")
    
    def load_gt_disp_scared(self, path):
        #all_disps= sorted(glob('{}/*.png'.format(path)))
        #gt_disparities = []
        #for i in range(len(all_disps)):
        disp = cv2.imread(path, -1)
        disp = disp.astype(np.float32) / 256
        return disp


    def __getitem__(self, idx):
        #print(self.left_paths[idx])
        left_image = Image.open(self.left_paths_all[idx]).convert('RGB')
        right_image = Image.open(self.right_paths_all[idx]).convert('RGB')
        #left_disp = self.load_gt_disp_scared(self.left_disp_all[idx])

        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample#, left_disp
    
    def __len__(self):
        return len(self.left_paths_all)

def prepare_dataloader3(data_dir, data_type, scared_d, keyframe, mode, augment_parameters, do_augmentation, batch_size, size, num_workers):
    
    data_transform = False
    if data_type=='kitti':
        dataset = KittiStereo2015Loader(data_dir, mode, transform=data_transform)
    
    elif data_type=='eigen':
        dataset= EigenKittiLoader(data_dir, mode, transform=data_transform)
    elif data_type=='video_gen':
        dataset= VideoLoader(data_dir, mode, transform= data_transform)
    elif data_type=='scared':
        #dataset, keyframe, transform=None
        dataset= SCAREDTestLoader(scared_d, keyframe, transform=data_transform)
    elif data_type=='heart':
        #dataset, keyframe, transform=None
        #root_dir, heart_data=5, transform=None
        dataset= HeartLoaderTest('', scared_d, transform=data_transform)
    elif data_type=='hamlyn':
        dataset= HamlynLoader(data_dir, mode, transform=data_transform)
    else:
        raise Exception('the data type is not implemented, please create a data loader for {}'.format(data_type))
                            
    #dataset = ConcatDataset(datasets)
    n_img = len(dataset)
    if mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=False)
    return n_img, loader, dataset #dataset.left_img_files, dataset.right_img_files