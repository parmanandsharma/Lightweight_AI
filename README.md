# AI based medical image analysis and disease daignosis based on LWBNA_Unet based models

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



