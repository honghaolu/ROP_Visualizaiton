#Step 7: Predicting a single photo

import numpy as np
import torch
import cv2
import os
from model.LDA_UNet import U_Net_CBAM
from torchsummary import summary
import matplotlib.pyplot as plt
def pred(img_path,result_path):
    print("Load model.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = U_Net_CBAM(img_ch=1, output_ch=1)

    net.to(device=device)

    summary(net, input_size=(1, 512, 512))

    net.load_state_dict(torch.load('best_model/best_model_ridge_LDAUNet_0731.pth', map_location=device))  # todo

    net.eval()
    print("Load model done.")
    print("Get predict result.")
    origin_img = cv2.imread(img_path)
    origin_shape = origin_img.shape
    print("Origin image shape:",origin_shape)
    # transform gray image
    img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (512, 512))
    # Convert to batch 1, channel 1, array size 512 * 512
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    print("Predicting the shape of a graph:",img.shape)

    # convert to tensor
    img_tensor = torch.from_numpy(img)

    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    # predit
    pred = net(img_tensor)
    # extract result
    pred = np.array(pred.data.cpu()[0])[0]
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)

    # The complete save path of the predicted image
    output_file_path = os.path.join(result_path, 'pred.png')

    cv2.imwrite(output_file_path, pred)

    print(f"Image saved to: {output_file_path}")

    # show result
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB))
    plt.title('Fundus image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
    plt.title('Mask image')

    plt.show()

if __name__ == '__main__':
    img_path = 'dataset/predit/48.png'
    result_path = 'dataset/predit/'

    pred(img_path,result_path)
