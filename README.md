## Saratan (Arabic : سرطان)##

This repo contains the code used for the following paper.
```
@Inbook{Christ2016,
title="Automatic Liver and Lesion Segmentation in CT Using Cascaded Fully Convolutional Neural Networks and 3D Conditional Random Fields",
author="Christ, Patrick Ferdinand and Elshaer, Mohamed Ezzeldin A. and Ettlinger, Florian and Tatavarty, Sunil and Bickel, Marc and Bilic, Patrick and Rempfler, Markus and Armbruster, Marco and Hofmann, Felix and D'Anastasi, Melvin and Sommer, Wieland H. and Ahmadi, Seyed-Ahmad and Menze, Bjoern H.",
editor="Ourselin, Sebastien and Joskowicz, Leo and Sabuncu, Mert R. and Unal, Gozde and Wells, William",
bookTitle="Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II",
year="2016",
publisher="Springer International Publishing",
address="Cham",
pages="415--423",
isbn="978-3-319-46723-8",
doi="10.1007/978-3-319-46723-8_48",
url="http://dx.doi.org/10.1007/978-3-319-46723-8_48"
}
```
The offical repo referred to in the paper is found in this repo : https://github.com/IBBM/Cascaded-FCN/

### Training the network on your own data using UNet pretrained models ###
To retrain the network on your data, find the training notebook and the training prototxt in `notebooks/realtime_train.ipynb`. The notebook trains the network using the pretrained UNet model as initialization. To download the pretrained UNet model:

 1. Go to http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
 2. Download u-net-release-2015-10-02.tar.gz from the download section at the bottom of the page
 3. Extract the file and move phseg_v5.caffemodel to the same directory of the notebook 

### Using our pretrained network for inference or for further training ###
This is what you will need :
 1. Pretrained models for liver/lesion segmentation : In https://github.com/IBBM/Cascaded-FCN/tree/master/models/cascadedfcn/step1 (and /step2) a link to our models is included in the README.md
 2. Deploy prototxt : In https://github.com/IBBM/Cascaded-FCN/tree/master/models/cascadedfcn/step1 (and /step2)
 3. (for training) Training code : In this repo under notebooks/realtime_train.ipynb
 4. (for training) Training and solver prototxt : In this repo under notebooks/solver_unet.prototxt and notebooks/unet-overfit-python.prototxt . The net prototxt works for step1 and step2 training.
 
### Notes ###
Important code locations :

 1. *data/layers* :  Caffe python data layer used for training from numpy volumes (numpy_data_layer.py) and it's confguration (config.py). nifti_to_numpy.py converts Nifti volumes to .Npy files, we found this very useful to load these large files from memory mapped locations on the disk, allowing for large scale training with adequate memory requirements.
 1. *deploy/* : files needed to deploy a machine with Nvidia drivers and a ready caffe docker container
 1. *validation/pipeline* : Scripts to do End-to-End validation a given model, reporting all desired scores.
 1. *notebooks/* : Notebook containing training code with realtime monitoring of the loss. The solver and train prototxt are also found here.
 1. *crf/* : scripts used to run or train crf. 
 1. *diagnostics/* : (mostly obselete) tools used to do sanity checks, for example to verify lmdb databases have the correct images.

Further README.md files are in the mentioned directories
