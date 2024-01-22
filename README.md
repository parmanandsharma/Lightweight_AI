# AI based medical image analysis and disease diagnosis based on LWBNA_Unet based models

## Overview
LWBNA_unet is a lightweight deep learning model developed for biological image analysis, specifically designed to address challenges in processing large, complex biological images. It is designed with a focus on efficiency and reduced computational resources. It is approximately 10 times lighter than the conventional Unet model. This lightweight design makes it suitable for deployment on devices (such as mobile phone, Nvidia’s Jetson, Raspberry pi) with limited memory and processing capabilities, which is particularly advantageous for tele-screening and real-time medical image analysis.

## Model Architecture
The LWBNA_unet is based on the Unet architecture, which is a convolutional neural network (CNN) widely used for image segmentation tasks. It consists of an encoder-decoder structure with skip connections to capture both local and global image features. LWBNA-Unet introduces an innovative concept at the bottleneck of the network. This concept involves channel narrowing, where the number of channels progressively reduces, akin to controlling the flow of information through a channel by varying its width. This channel-wise attention in both the encoder and decoder paths contributes to improved discrimination of unwanted image features. Despite its lightweight design, LWBNA-Unet achieves high segmentation accuracy. It can accurately delineate and segment features within medical images, including complex or noisy ones.

![LWBNA_Unet Model Architecture](https://github.com/parmanandsharma/LWBNA_Models/blob/master/LWBNA_unet_architecture.png?raw=true)

LWBNA-Unet model's versatility allows for easy modification to create a DL model for binary or multi-class classification tasks. Additionally it can be configured in a GAN architecture for image generation as well as pixel to pixel image translation. 


## Applications
The model has been applied in various biological imaging scenarios, demonstrating its capability to handle diverse data types and complex image structures. It shows promising results in enhancing the understanding of biological processes and structures. For example, LWBNA-Unet model has demonstrated exceptional glaucoma detection capabilities, outperforming commercial software. Features such as the foveal avascular zone (FAZ) area and perimeter, measured from optical coherence tomography angiography (OCTA) images, have been successfully used as biomarkers for glaucoma detection. When trained as an image denoiser, it can remove salt and paper noise from the medical images, which results in more accurate analysis of vascular structure such as vessel density, vessel length etc., and its relationship with diseases. 

## Reference
For more detailed information about LWBNA_unet and its applications, please refer to the following article: Sharma et.al., [A lightweight deep learning model for automatic segmentation and analysis of ophthalmic images,](https://www.nature.com/articles/s41598-022-12486-w) published in Scientific Reports, vol. 12, Article number: 8508 (year 2022).

## Patent
Toru Nakazawa and Parmanand Sharma, Biological image processing program, biological image processing device, and biological image processing method, WO2023062764A1, 2023 (https://patentimages.storage.googleapis.com/84/ed/cd/c728e12db9a034/WO2023062764A1.pdf)


## License
This project is licensed under the terms of the MIT license.
The use of LWBNA_unet models for commercial purposes is subject to the terms and conditions of the patent ["WO2023062764A1"], held by ["Parmanand Sharma and Toru Nakazawa"]. Any use of the Software for commercial purposes without proper licensing under the aforementioned patent is strictly prohibited. For information on obtaining a commercial license, please contact ["inventors"].
For research and non-commercial use, the Software is available under the terms of the MIT License.

## Usage
Install Python >=3.8

Setup GPU, if available,

This package will install,

'numpy>=1.24',

'matplotlib>=3.5',

'opencv-python>=4.5.1', 

'pandas>=1.4.0',

'tensorflow-gpu>=2.5',

Setting dataset directory structure:

To train image segmentation model based on LWBNA_unet, at first it is required to create a dataset. Dataset has two folders, training and validation. Both the folders have two folder, one is image and another is GT or Mask. Filename of image in image folder and it's mask in mask folder has the same name. For example the dataset directory tree may looks like:
    
training

   |----images
   
      |--img_01.png 
      
      |-------
      
   |----mask
   
      |--img_01.png
      
      |-------
        
validation

   |----val_images
   
       |--val_01.png
       
       |------
       
  |----val_mask
  
       |---val_01.png
       
       |-------
                   
calling segmentation model--------

from Lightweight_AI import LWBNA_unet_segmentation_model  

model = LWBNA_unet_segmentation_model(img_shape=(512,512,3))

------rest you can writing your loss function, optimizer etc. -----------
      
EASY STYLE: Training image segmentation LWBNA_unet model in a very easy way. it will ask you to just open image file from training and validation image directories. Rest it will do everything automatically. Please follow the pop-up windows, if image windows appear just press anykey ---------
       
from Lightweight_AI import Train_LWBNA_segmentation_model 

path_for_saving = 'c:/Trained_LWBNA_unet' # set your saving path  

Train_LWBNA_segmentation_model(image_size=(320,320),Resize=False,img_preprocess=False,total_epochs=500,batch_size=4, model_name='LWBNA_unet',save_path=path_for_saving, load_validation_data = False, early_stop=True, patience=30)
     
In the above function, if you want to resize image and mask input it and make Resize=True, if you want to equalize image intensity make img_preprocess=True, set total training epochs, batch_size, model_name. if load_validation=False, it will use 20% data from training, you can use early_stoping to avoid overfitting and set patience value. This function uses Dice_loss and adam optimizer.
   
