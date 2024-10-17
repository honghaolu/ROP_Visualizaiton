#Step 6: test

from tqdm import tqdm
from utils_metrics import compute_mIoU, show_results
import numpy as np
import torch
import os
import cv2
import shutil
from torchsummary import summary
from model.LDA_UNet import U_Net_CBAM

def cal_miou(test_dir="dataset/test/images",
             pred_dir="dataset/test/result",
             gt_dir="dataset/test/masks"):

    # miou_mode = 0 represents the entire miou calculation process, including obtaining prediction results and calculating miou.
    miou_mode = 0

    # Number of categories+1
    num_classes = 2
    name_classes = ["Background", "Ridge"]

    if miou_mode == 0 or miou_mode == 1:
        # If the saved folder exists, delete it first and then create it again
        if os.path.exists(pred_dir):
            shutil.rmtree(pred_dir)
        os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load network
        net = U_Net_CBAM(img_ch=1, output_ch=1)

        # Copy the network to deivce
        net.to(device=device)

        summary(net, input_size=(1, 512, 512))
        # Load model parameters
        net.load_state_dict(torch.load('best_model/best_model_ridge_LDAUNet_0731.pth', map_location=device)) # todo

        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".png")
            img = cv2.imread(image_path)
            origin_shape = img.shape

            # Convert to grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Convert to batch 1, channel 1, array size 512 * 512
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            # Convert to Tensor
            img_tensor = torch.from_numpy(img)

            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # prediction
            pred = net(img_tensor)
            # Extract results
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision= compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # Execute the function to calculate mIoU

        print("Get miou done.")
        miou_out_path = pred_dir
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == '__main__':
    cal_miou()
