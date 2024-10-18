# ROP visualization
This project proposed a lightweight UNet model for localizing ROP that incorporates multi-scale dilated convolution and attention mechanism. First, the LDA-UNet lightweight network is designed. The model has small parameters and strong detail segmentation ability, which alleviates the issue of information loss and effectively improves the segmentation accuracy of the ROP demarcation/ridge. Then, LDA-UNet is combined with a contour detection algorithm to form an object visualization model, which can accurately locate the lesion area and delineate the lesion contour, thereby achieving visualization of CNN-assisted ROP diagnosis, and improve the interpretability and credibility of the model.


## 1. Dataset and preprocessing (Folder: Image Preprocessing).
 
This project used data from the publicly available HVDROPDB dataset([DOI:10.17632/xw5xc7xrmp.3](https://data.mendeley.com/datasets/xw5xc7xrmp/3)), including images of demarcation line/ridge and their corresponding ground truths.In order to improve the generalization ability and robustness of the model, we perform image enhancement and image augmentation on the original image.

### Step 1: Img_enhancement.py.
We used image enhancement techniques such as normalization,Contrast Limited Adaptive Histogram Equalization (CLAHE), gamma transformation, and weighted average grayscale conversion. 

### Step 2: Img_augment.py. 
we use the ImageDataGenerator function from the Tensorflow library to augment both images and masks by 20 times. To minimize image contamination and ensure that each image is unique, the augmentation methods included random rotation, horizontal and vertical shifting, scaling, cropping, horizontal and vertical flipping, and adjusting channel pixel values to simulate different lighting conditions. Ultimately, 1,000 images were obtained, each with a size of 512 x 512 pixels, and the dataset was split into training and testing sets in an 8:2 ratio. 
![image](https://github.com/user-attachments/assets/07b0f5d0-fc02-42cf-b3f3-5ea246c81a45)

1000 preprocessed images and 1000 mask images are stored in [DOI: 10.6084/m9.figshare.27229680.](https://doi.org/10.6084/m9.figshare.27229680.v2)divided into a training set of 800 images and a testing set of 200 images.

![image](https://github.com/user-attachments/assets/e4b566ce-81e1-4e1f-b262-e9d0e07cd8db)

## 2. Model(LDA-UNet) design and training 

### Step 3: dataset.py
Get training set and validation set data.

### Step 4: model.LDA_UNet.py
 Design LDA-UNet model:
 lightweighted unet: Replace standard convolution with depthwise separable convolution.
 Dilated convolution: downsampling and upsampling dilated convolution are symmetric[1, 2, 4, 2, 1].
 Attention Mechanism: CBAM follows the convolutional block in each encoding stage and is outputted via an addition operation.
![image](https://github.com/user-attachments/assets/018b2ec2-4e42-41ad-ba8b-f2043f412b2a)

### Step 5: train.py
The experiment was based on Python 3.8.19, Torch 1.11.0, Cuda113 to build a deep learning framework. The platform configuration for model training is: NVIDIA GeForce RTX 4060. During the training process, the optimizer uses RMSprop (Root Mean Square Propagation), the initial learning rate is 1e-5, the momentum is 0.9, the epoch is 20, the batch size is 2, the image input size is 512×512, and the loss function is the binary cross entropy loss function (BCEWithLogitsLoss). 
![Training and Validation LOSS](https://github.com/user-attachments/assets/60830d19-6ecd-4b58-b831-012b7bd85198)

## 3. Model evaluation

### Step 6: test.py
Evaluate the model's mIoU, recall, precision, accuracy, etc. in the test set.
![mIoU](https://github.com/user-attachments/assets/854cd875-fa97-4dc9-b9a7-9dd30d70ca3f) ![mPA](https://github.com/user-attachments/assets/7198db11-e737-4476-934c-7c85de3170b1) ![Precision](https://github.com/user-attachments/assets/d67dab07-e254-4a3d-a514-a7c4e10cf5ce) ![Recall](https://github.com/user-attachments/assets/5ccc9218-348d-44c5-8edb-f519e37dad1b)


## 4. ROP visalization

### Step 7: pred.py
Predicting mask image based on fundus image.
![Figure_1](https://github.com/user-attachments/assets/47991fcb-d59b-408b-9c16-713dd83cc2bf)

### Step 8: ROP_Visualization.py
According to the predicted mask image, the contour of the ROP is delineated and the rectangular box of the ROP is located, thereby realizing the visualization of the ROP.
Visualization of ROP,Generate target bounding boxes and draw lesion contours.
![Figure_1](https://github.com/user-attachments/assets/93219539-0d64-4782-abbb-194548d5d5b9)

## 5. Comparative experiment (Folder: model)
To validate the effectiveness of LDA-UNet, we selected four current typical lightweight UNet models for comparison. 

The selected models include: 

unet.py: Unet is a lightweight version of the UNet model with the number of channels in the decoder reduced by half. 

Mobile_UMet.py: Mobile_UNet is a typical representative of lightweight CNN using DSConv. 

GhostU_Net.py(class GhostU_Net): GhostU_Net is a model that employs a ghost module to generate more features with fewer parameters and reduce computation. 

GhostU_Net.py(class GhostU_Net2): GhostU_Net2 is an upgraded version of GhostU_Net with even fewer parameters. 

![image](https://github.com/user-attachments/assets/8ff35268-ec9d-4b32-bf65-9d445371d8fb)




