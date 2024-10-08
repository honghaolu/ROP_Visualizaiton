#Step 3：dataset

import torch
import cv2
import os
import glob
from torch.utils.data import Dataset

#dataset
class Train_Loader(Dataset):
    # Initialization function, read all pictures under data_path
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'train/images/*.png'))

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('images', 'masks')

        # Read training images and label images
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # change the pixel value from 255 to 1
        if label.max() > 1:
            label = label / 255

        #Move the channel forward and convert HWC to CHW, which is more suitable for CNN
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label

    def __len__(self):
        # Return the size of the test set
        return len(self.imgs_path)

#Validation set
class Val_Loader(Dataset):
    def __init__(self, data_path):
        # Initialization function, read all pictures under data_path
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'test/images/*.png'))

    def __getitem__(self, index):
        # Read the image according to the index
        image_path = self.imgs_path[index]
        # Generate label_path based on image_path
        label_path = image_path.replace('images', 'masks')


        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)


        # change the pixel value from 255 to 1
        if label.max() > 1:
            label = label / 255


        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label

    def __len__(self):
        # Return the size of the test set
        return len(self.imgs_path)

# Can be executed separately to view dataset information
if __name__ == "__main__":
    train_loader = Train_Loader("dataset")
    print("Number of training set data：", len(train_loader))
    val_loader = Val_Loader("dataset")
    print("Number of validation set data：", len(val_loader))
    train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                               batch_size=4,
                                               shuffle=True)

    for image, label in train_loader:
        print(image.shape)

    val_loader = torch.utils.data.DataLoader(dataset=val_loader,
                                             batch_size=1,
                                             shuffle=False)
    for image, label in val_loader:
        print(image.shape)