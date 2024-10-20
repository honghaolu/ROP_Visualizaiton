
from model.ghostnet import *

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class mix_conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(mix_conv_block,self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//4, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out//4),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//4, kernel_size=5,stride=1,padding=2,bias=True),
            nn.BatchNorm2d(ch_out//4),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//4, kernel_size=7,stride=1,padding=3,bias=True),
            nn.BatchNorm2d(ch_out//4),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x33 = self.conv3(x)
        x = torch.cat((x3,x5,x7,x33),dim=1)
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out, t=2, kernel_size=3, padding=1):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=kernel_size,stride=1,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class mix_RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(mix_RRCNN_block,self).__init__()

        self.RCNN3 = nn.Sequential(
            Recurrent_block(ch_out//4,t=t, kernel_size=3),
            Recurrent_block(ch_out//4,t=t, kernel_size=3)
        )

        self.RCNN5 = nn.Sequential(
            Recurrent_block(ch_out//4,t=t, kernel_size=5, padding=2),
            Recurrent_block(ch_out//4,t=t, kernel_size=5, padding=2)
        )

        self.RCNN7 = nn.Sequential(
            Recurrent_block(ch_out//4,t=t, kernel_size=7, padding=3),
            Recurrent_block(ch_out//4,t=t, kernel_size=7, padding=3)
        )

        #self.RCNN3 = Recurrent_block(ch_out//4,t=t, kernel_size=3)
        #self.RCNN5 = Recurrent_block(ch_out//4,t=t, kernel_size=5, padding=2)
        #self.RCNN7 = Recurrent_block(ch_out//4,t=t, kernel_size=7, padding=3)
        self.RCNN = Recurrent_block(ch_out,t=t)
        self.Conv_1x1_0 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_1 = nn.Conv2d(ch_out,ch_out//4,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x0 = self.Conv_1x1_0(x)
        x = self.Conv_1x1_1(x0)
        x3 = self.RCNN3(x)
        x5 = self.RCNN5(x)
        x7 = self.RCNN7(x)
        x33 = self.RCNN3(x)
        x_cat = torch.cat((x3,x5,x7,x33),dim=1)
        #x1 = self.RCNN(x_cat)
        return x0+x_cat


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class GhostU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(GhostU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        #self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv1 = GhostBottleneck(img_ch, 64, 64)

        #self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv2 = GhostBottleneck(64, 128, 128)

        #self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv3 = GhostBottleneck(128, 256, 256)

        #self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv4 = GhostBottleneck(256, 512, 512)

        #self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Conv5 = GhostBottleneck(512, 1024, 1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class GhostU_Net2(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(GhostU_Net2,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        #self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv1 = GhostBottleneck(img_ch, 64, 64)

        #self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv2 = GhostBottleneck(64, 128, 128)

        #self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv3 = GhostBottleneck(128, 256, 256)

        #self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv4 = GhostBottleneck(256, 512, 512)

        #self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Conv5 = GhostBottleneck(512, 1024, 1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        #self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up_conv5 = GhostBottleneck(1024, 512, 512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        #self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up_conv4 = GhostBottleneck(512, 256, 256)        

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        #self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up_conv3 = GhostBottleneck(256, 128, 128)          

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        #self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Up_conv2 = GhostBottleneck(128, 64, 64) 

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



