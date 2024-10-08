# ROP visualization

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
