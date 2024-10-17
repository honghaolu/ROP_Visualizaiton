# ROP visualization
This project proposed a lightweight UNet model for localizing ROP that incorporates multi-scale dilated convolution and attention mechanism. First, the LDA-UNet lightweight network is designed. The model has small parameters and strong detail segmentation ability, which alleviates the issue of information loss and effectively improves the segmentation accuracy of the ROP demarcation/ridge. Then, LDA-UNet is combined with a contour detection algorithm to form an object visualization model, which can accurately locate the lesion area and delineate the lesion contour, thereby achieving visualization of CNN-assisted ROP diagnosis, and improve the interpretability and credibility of the model.

## Phase 1: Image Preprocessing (Folder: Image Preprocessing).

Step 1: Image enhancement,(norm、clahe、gamma、gray).

Step 2: Image augmentation using the ImageDataGenerator function. trainset and testset(validation set) 8:2.


## Phase 2: Model(LDA-UNet) training 

Step 3: dataset, Get training set and validation set data.

Step 4: model.LDA-UNet, Design LDA-UNet model.

## Phase 3: Model evaluation

Step 5: test, Evaluate the model's mIoU, recall, precision, accuracy, etc. in the test set.

## Phase 4: ROP visalization

Step 6: ROP visalization, Visualization of ROP,Generate target bounding boxes and draw lesion contours.
