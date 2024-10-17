#Step 5: Model Training
#Evaluate the loss and IOU of the training set and validation set. IOU can more intuitively monitor the training effect of the model

from model.LDA_UNet import U_Net_CBAM
from dataset import Train_Loader, Val_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt
from utils_iou import calculate_iou
def train_net(net, device, data_path, epochs=40, batch_size=2, lr=1e-5):
    # Loading training set
    train_loader = Train_Loader(data_path)
    per_epoch_num = len(train_loader) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                               batch_size=batch_size,
                                               shuffle=True)
    # Load the validation set (this project uses the test set as the validation set)
    val_loader = Val_Loader(data_path)
    val_loader = torch.utils.data.DataLoader(dataset=val_loader,
                                               batch_size=1,
                                               shuffle=False)

    #RMSprop
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # Binary Cross Entropy Loss Function
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    # Training epochs
    with tqdm(total=epochs*per_epoch_num) as pbar:
        # Record the train loss and val loss of each epoch
        train_losses = []
        train_ious = []
        val_losses = []
        val_ious = []
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0
            running_iou = 0.0
            # Start training according to batch_size
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # Use network parameters to output prediction results
                pred = net(image)
                # print(pred)

                # Calculate loss
                loss = criterion(pred, label)
                running_loss += loss.item()  # Accumulated losses

                # Save the network parameters with the smallest loss value
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model/best_model_ridge_LDAUNet_20241002.pth')
                    #torch.save(net.state_dict(),'best_model/best_model_test.pth')
                # Back propagation and optimization, updating parameters
                loss.backward()
                optimizer.step()
                pbar.update(1)

                # Calculate the positive sample IoU
                iou = calculate_iou(pred,label)
                running_iou += iou

            # Calculate the average loss of the training set for this epoch
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)  # record train loss

            train_iou = running_iou / len(train_loader) # Calculate the average iou of this epoch
            train_ious.append(train_iou)# record train iou

            #Calculate the loss and iou of the validation set
            net.eval()
            val_running_loss = 0.0
            val_running_iou = 0.0
            with torch.no_grad():
                for image, label in val_loader:
                    # Copy data to the device
                    image = image.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)
                    pred = net(image)  # Get predicted values

                    loss = criterion(pred, label) # Calculating Losses
                    val_running_loss += loss.item()

                    #Calculate the iou of the validation set
                    iou = calculate_iou(pred, label)
                    val_running_iou += iou  # 累加IoU

            # Calculate the average loss on the validation set
            val_loss = val_running_loss / len(val_loader)  # Calculate the average loss
            val_losses.append(val_loss)  # record val loss

            # Calculate the average iou of the validation set
            val_iou = val_running_iou / len(val_loader)  # Calculate average iou
            val_ious.append(val_iou)  # record val loss

            print('Epoch %d loss: %.3f, val_loss: %.3f,iou: %.3f,val_iou: %.3f' % (epoch + 1, train_loss, val_loss,train_iou,val_iou))

        # Use the matplotlib library to plot the train loss and val loss curves.
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss ')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Use the matplotlib library to plot the train IOU and val IOU curves.
        plt.figure(figsize=(10, 5))
        plt.plot(train_ious, label='Train IoU')
        plt.plot(val_ious, label='Validation IoU')
        plt.title('Training and Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.show()


if __name__ == "__main__":

    # Select the device. If cuda is available, use cuda. If not, use cpu.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The device you are running is：",device)
    # n_classes：The probability number of each pixel you want to obtain. For one class and background, use n_classes = 1. The output here is a black and white comparison, so use 1
    net = U_Net_CBAM(img_ch=1, output_ch=1)

    #Copy the network to deivce
    net.to(device=device)
    # Print model structure
    summary(net, input_size=(1, 512, 512))

    # Specify the training set and start training
    data_path = "dataset"  # todo Modify to your dataset location
    print("Training started, calculating, please wait patiently")
    train_net(net, device, data_path, epochs=20, batch_size=4, lr=1e-5)
