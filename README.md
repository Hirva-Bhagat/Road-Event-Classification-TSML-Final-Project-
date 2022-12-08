# Road-Event-Classification-TSML-Final-Project-
Final project for TSML Course. oad event detection is associated with identifying lane changes\cite{scheel2019attention}, pedestrian interactions\cite{MIANI2022432}, maneuver events\cite{gao2022attention}, etc. from sensory or telemetry vehicle information. If treated as just a visual task it could be done in the form on event/action recognition. However, in order to understand driver's decision-making process, this definition could be modified. In order to correlate driver's attention with the road events, we propose to fuse driver's gaze information in the learning architecture. This project implements multi source learning by using augmentation and fusing external features to modify the learning process.oad event detection is associated with identifying lane changes\cite{scheel2019attention}, pedestrian interactions\cite{MIANI2022432}, maneuver events\cite{gao2022attention}, etc. from sensory or telemetry vehicle information. If treated as just a visual task it could be done in the form on event/action recognition. However, in order to understand driver's decision-making process, this definition could be modified. In order to correlate driver's attention with the road events, we propose to fuse driver's gaze information in the learning architecture. This project implements multi source learning by using augmentation and fusing external features to modify the learning process.

## Acknowledgements
The code was modified from "https://github.com/yaorong0921/Driver-Intention-Prediction"
This project is thankful to them

## Dataset
Dataset Brain4cars link: https://github.com/asheshjain399/ICCV2015_Brain4Cars

Please download the dataset and extract it in a seperate folder. 

## Dataset Preparation
In the ipynb file there are instructions given for code blocks and their specific uses.

###1. Extract frames from outside videos (road_camera): 

This can be done by specifying the file path and running cell 2.

###2. Generate attention maps

This can be done by running cell block 3.
For data preparation we need attention maps. These are generated based on PoG Estimation.
For PoG estimation we use this github repo: https://github.com/VTTI/gaze-fixation-and-object-saliency
It could be tedious to setup for PoG Estimation, instead use this link to download the resulting CSV files (gaze_results): https://drive.google.com/file/d/1OsOn3hsKjK0lCAxpQXDq3kSnL-iCn7mM/view?usp=sharing


###3. Augment attention maps on outside frames (for Early fusion)

This can be done by running cell block 4

## Training
We train 3 models and parameters for them are defined in ipynb file.

### Baseline
3D ResNet model. Run cell block 6 and 7. The code generates evaluation and accuracy for the best epoch and saves it.

### EF-3D ResNet
3D ResNet model with Early Fusion. You have to change the video_path parameter in opt to your augmented dataset folder. Run cell block 8 and 9. 

### LF-3D ResNet
3D ResNet model with Late Fusion. You have to change the video_path parameter in opt to your attention maps folder and gaze_path to outside frames folder. Run cell block 10 and 11.
