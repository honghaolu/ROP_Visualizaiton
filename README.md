# ROP visualization
This project proposed a lightweight UNet model for localizing ROP that incorporates multi-scale dilated convolution and attention mechanism. First, the LDA-UNet lightweight network is designed. The model has small parameters and strong detail segmentation ability, which alleviates the issue of information loss and effectively improves the segmentation accuracy of the ROP demarcation/ridge. Then, LDA-UNet is combined with a contour detection algorithm to form an object visualization model, which can accurately locate the lesion area and delineate the lesion contour, thereby achieving visualization of CNN-assisted ROP diagnosis, and improve the interpretability and credibility of the model.

## 1. Dataset and preprocessing (Folder: Image Preprocessing).
 
This project used data from the publicly available HVDROPDB dataset([DOI:10.17632/xw5xc7xrmp.3](https://data.mendeley.com/datasets/xw5xc7xrmp/3)), including images of demarcation line/ridge and their corresponding ground truths.In order to improve the generalization ability and robustness of the model, we perform image enhancement and image augmentation on the original image.

### Step 1: Img_enhancement.py.
We used image enhancement techniques such as normalization,Contrast Limited Adaptive Histogram Equalization (CLAHE), gamma transformation, and weighted average grayscale conversion. 

### Step 2: Img_augment.py. 
we use the ImageDataGenerator function from the Tensorflow library to augment both images and masks by 20 times. To minimize image contamination and ensure that each image is unique, the augmentation methods included random rotation, horizontal and vertical shifting, scaling, cropping, horizontal and vertical flipping, and adjusting channel pixel values to simulate different lighting conditions. Ultimately, 1,000 images were obtained, each with a size of 512 x 512 pixels, and the dataset was split into training and testing sets in an 8:2 ratio. 

1000 preprocessed images and 1000 mask images are stored in [DOI: 10.6084/m9.figshare.27229680.](https://doi.org/10.6084/m9.figshare.27229680.v2)
## 2. Model(LDA-UNet) training 

### Step 3: dataset.py
Get training set and validation set data.

### Step 4: model.LDA_UNet.py
 Design LDA-UNet model:
 lightweighted unet: Replace standard convolution with depthwise separable convolution.
 Dilated convolution: downsampling and upsampling dilated convolution are symmetric[1, 2, 4, 2, 1]
 Attention Mechanism: CBAM follows the convolutional block in each encoding stage and is outputted via an addition operation.
![image](https://github.com/user-attachments/assets/018b2ec2-4e42-41ad-ba8b-f2043f412b2a)

## Phase 3: Model evaluation

Step 5: test, Evaluate the model's mIoU, recall, precision, accuracy, etc. in the test set.

## Phase 4: ROP visalization

Step 6: ROP visalization, Visualization of ROP,Generate target bounding boxes and draw lesion contours.
