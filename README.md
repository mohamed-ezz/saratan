## Saratan (Arabic : سرطان)##

This is a newer version of caffe-tools repository. It replaces both caffe-tools and caffe-helper repos.
It contains the code used for the paper `Automatic Liver and Lesion Segmentation in CT Using Cascaded Fully Convolutional Neural Networks and 3D Conditional Random Fields`


Folder description :

 1. *data* : everything related to data creation and processing, and also Caffe python layers
 1. *deploy* : files needed to deploy a machine with Nvidia drivers and a ready caffe container
 1. *validation* : Scripts to do End-to-End validation a given model, reporting all desired scores.
 1. *notebooks* : template notebooks. For example realtime training notebook to train a network and see realtime curves and image prediction examples.
 1. *crf* : scripts used to run or train crf. 
 1. *diagnostics* : tools used to do sanity checks, for example to verify lmdb databases have the correct images.

Further README.md files are in the mentioned directories