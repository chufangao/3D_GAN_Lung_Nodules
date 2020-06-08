# 3D_GAN_Lung_Nodules
This is the code for an undegraduate research program at DePaul University.

It contains code for generating 3D lung nodule ct scans From LIDC data through a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP). 

The data can be found at https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI (this is large, ~100 gb).

Some data preprocessing files are not present in this repository, so you may have to write your own (getting and sorting the dicom images into a dictionary).

The paper is published at SPIE Medical Imaging: Augmenting LIDC Dataset Using 3D Generative Adversarial Networks to Improve Lung Nodule Detection
(pdf available on [ResearchGate](https://www.researchgate.net/profile/Chufan_Gao/publication/331723419_Augmenting_LIDC_dataset_using_3D_generative_adversarial_networks_to_improve_lung_nodule_detection/links/5d2357a6299bf1547ca34e48/Augmenting-LIDC-dataset-using-3D-generative-adversarial-networks-to-improve-lung-nodule-detection.pdf))

If you do decide to use this, please cite:
```
@inproceedings{gao2019augmenting,
  title={Augmenting LIDC dataset using 3D generative adversarial networks to improve lung nodule detection},
  author={Gao, Chufan and Clark, Stephen and Furst, Jacob and Raicu, Daniela},
  booktitle={Medical Imaging 2019: Computer-Aided Diagnosis},
  volume={10950},
  pages={109501K},
  year={2019},
  organization={International Society for Optics and Photonics}
}
````
