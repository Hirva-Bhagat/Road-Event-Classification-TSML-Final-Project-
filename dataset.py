from datasets.Brain4cars import With_gaze, Without_gaze
import torch.utils.data as data

class MyDataset(data.Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB
        
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        return xA, xB
    
    def __len__(self):
        return len(self.datasetA)
    
def get_training_set(opt, spatial_transform, horizontal_flip, temporal_transform,
                     target_transform):
    
    assert opt.dataset in ['With_gaze', 'Without_gaze']

    if opt.dataset == 'With_gaze':
        training_data = With_gaze(
                opt.video_path,
                opt.gaze_path,
                opt.annotation_path,
                'training',
                opt.n_fold,
                opt.end_second,
                1,
                spatial_transform=spatial_transform,
                horizontal_flip=horizontal_flip,
                temporal_transform=temporal_transform,
                target_transform=target_transform)
    if opt.dataset == 'Without_gaze':
        training_data = Without_gaze(
                opt.video_path,
                opt.annotation_path,
                'training',
                opt.n_fold,
                opt.end_second,
                1,
                spatial_transform=spatial_transform,
                horizontal_flip=horizontal_flip,
                temporal_transform=temporal_transform,
                target_transform=target_transform)
            
    

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['With_gaze', 'Without_gaze']

    if opt.dataset == 'With_gaze':
        validation_data = With_gaze(
                opt.video_path,
                opt.gaze_path,
                opt.annotation_path,
                'validation',
                opt.n_fold,
                opt.end_second,
                opt.n_val_samples,
                spatial_transform,
                None,
                temporal_transform,
                target_transform,
                sample_duration=opt.sample_duration)

    if opt.dataset == 'Without_gaze':
        validation_data = Without_gaze(
                opt.video_path,
                opt.annotation_path,
                'validation',
                opt.n_fold,
                opt.end_second,
                opt.n_val_samples,
                spatial_transform,
                None,
                temporal_transform,
                target_transform,
                sample_duration=opt.sample_duration)
   

    return validation_data
