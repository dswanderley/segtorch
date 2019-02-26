"""
Train network
"""

import os
import torch
import torchvision

import numpy as np
import torch.nn as nn
#import matplotlib.pyplot as plt

from torch import optim
from PIL import Image #,transform

from torch.utils.data import Dataset, DataLoader

#from torchvision import transforms, utils
from unet import Unet
#from dice import dice_loss

class UltrasoundDataset(Dataset):
    """B-mode ultrasound dataset"""

    def __init__(self, im_dir='im', gt_dir='gt', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.im_dir = im_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.images_name = os.listdir(self.im_dir)

    def __len__(self):
        """
        @
        """
        return len(self.images_name)

    def __getitem__(self, idx):
        """
        @
        """
        im_name = self.images_name[idx]

        im_path = os.path.join(self.im_dir, im_name)    # PIL image in [0,255], 3 channels
        gt_path = os.path.join(self.gt_dir, im_name)    # PIL image in [0,255], 3 channels
                
        image = Image.open(im_path)
        gt_im = Image.open(gt_path)

        # Image to array
        im_np = np.array(image).astype(np.float32) / 255.
        if (len(im_np.shape) > 2):
            im_np = im_np[:,:,0]

        # Grouth truth to array
        gt_np = np.array(gt_im).astype(np.float32)
        if (len(gt_np.shape) > 2):
            gt_np = gt_np[:,:,0]

        # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
        gray_mask = (gt_np / 255.).astype(np.float32)
            
        # Multi mask - background (R = 1) / ovary (G = 1) / follicle (B = 1) 
        t1 = 128./2.
        t2 = 255. - t1
        multi_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 3))
        # Background mask
        aux_b = multi_mask[:,:,0]
        aux_b[gt_np < t1] = 255.
        multi_mask[...,0] = aux_b
        # Ovary mask
        aux_o = multi_mask[:,:,1]
        aux_o[(gt_np >= t1) & (gt_np <= t2)] = 255.
        multi_mask[...,1] = aux_o
        # Follicle mask
        aux_f = multi_mask[:,:,2]
        aux_f[gt_np > t2] = 255.
        multi_mask[...,2] = aux_f
        # Convert to float
        multi_mask = (multi_mask / 255.).astype(np.float32)
                
        # Print data if necessary
        #Image.fromarray(gray_mask.astype(np.uint8)).save("gt.png")      
        #toprint = Image.fromarray(mask_rgb.astype(np.uint8))
        #toprint.save("multi_mask.png")

        # Apply transformations
        if self.transform:
            im_np, gray_mask, multi_mask = self.transform(im_np, gray_mask, multi_mask)
        
        # Convert to torch (to be used on DataLoader)
        return im_name, torch.from_numpy(im_np), torch.from_numpy(gray_mask), torch.from_numpy(multi_mask)


'''
Transformation parameters
'''
#rotate_range = 25
#translate_range = (10.0, 10.0)
#scale_range = (0.90, 1.50)
#shear_range = 0.0
#im_size = (512,512)


#LOSS FUNCTION
class DiceLoss (nn.Module):
    '''
    Dice Loss (Ignore background - channel 0)
    
    Arguments:
        @param prediction: tensor with predictions classes
        @param groundtruth: tensor with ground truth mask
    '''

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward (self, pred, gt):

        SMOOTH = 0.0001


        #torchvision.utils.save_image(gt[0,0,...], "gt_bkg.png")
        #torchvision.utils.save_image(gt[0,1,...], "gt_ov.png")
        #torchvision.utils.save_image(gt[0,2,...], "gt_fol.png")
        
        #torchvision.utils.save_image(pred[0,0,...], "results_bkg.png")
        #torchvision.utils.save_image(pred[0,1,...], "results_ov.png")
        #torchvision.utils.save_image(pred[0,2,...], "results_fol.png")

        # Ignorne background
        prediction = pred[:,1:,...].contiguous()
        groundtruth = gt[:,1:,...].contiguous()

        iflat = prediction.view(-1)
        tflat = groundtruth.view(-1)

        intersection = (iflat * tflat).sum()
        union = iflat.sum() + tflat.sum()
        dsc = ((2. * intersection + SMOOTH) / (union + SMOOTH))

        loss_dsc = 1. - dsc
        return loss_dsc


def saveweights(state):
    '''
    Save network weights.

    Arguments:
    @state: parameters of the network
    '''
    path = ''
    filename = path + 'weights.pth.tar'
    
    torch.save(state, filename)


def train_net(net, epochs=30, batch_size=3, lr=0.1):
    '''
    Train network function

    Arguments:
        @param net: network model
        @param epochs: number of training epochs (int)
        @param batch_size: batch size (int)
        @param lr: learning rate
    '''

    # Load Dataset
    ovary_dataset = UltrasoundDataset(im_dir='Dataset/im/', gt_dir='Dataset/gt/')
    data_len = len(ovary_dataset)

    train_data = DataLoader(ovary_dataset, batch_size=batch_size, shuffle=True)
    # dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=threads, drop_last=True, pin_memory=True)
    
    # Define parameters
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) #optim.Adam(net.parameters())
    criterion =  DiceLoss() # nn.CrossEntropyLoss()
    best_loss = 1000    # Init best loss with a too high value

    # Run epochs
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # Active train
        net.train()
        # Init loss count
        loss_train_sum = 0
        
        for batch_idx, (im_name, image, gray_mask, multi_mask) in enumerate(train_data):
            
            # Active GPU train
            if torch.cuda.is_available():
                net = net.to(device)
                image = image.to(device)
                gray_mask = gray_mask.to(device)
                multi_mask = multi_mask.to(device)
            
            # Handle with ground truth
            if type(criterion) is type(nn.CrossEntropyLoss()):
                groundtruth = gray_mask.long()
            else:
                multi_mask = multi_mask.permute(0, 3, 1, 2).contiguous()
                groundtruth = multi_mask

            # Run prediction
            image.unsqueeze_(1) # add a dimension to the tensor, respecting the network input on the first postion (tensor[0])
            pred_masks = net(image)

            # Print output preview
            if batch_idx == len(train_data):
                torchvision.utils.save_image(image[0,...], "input.png")
                torchvision.utils.save_image(groundtruth[0,...], "groundtruth.png")
                torchvision.utils.save_image(pred_masks[0,...], "output.png")
            
            # Calculate loss for each batch
            loss = criterion(pred_masks, groundtruth)
            #loss = criterion(pred_masks[-1,...], groundtruth[-1,...])
            loss_train_sum += len(image) * loss.item()

            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()           

        # Calculate average loss per epoch
        avg_loss_train = loss_train_sum / data_len
        print('loss: {:f}'.format(avg_loss_train))
        
        # To evaluate on validation set
        # XXXXXXXXXXXXXXXXXXXXX
        # call train()
        # epoch of training on the training set
        # call eval()
        # evaluate your model on the validation set
        # repeat
        # XXXXXXXXXXXXXXXXXXXXX

        # Save weights
        if best_loss > avg_loss_train:
            best_loss = avg_loss_train

            saveweights({
                        'epoch': epoch,
                        'arch': 'unet',
                        'state_dict': net.state_dict(),
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()
                        })



# if __name__ == '__main__':


# Load Unet
net = Unet(n_channels=1, n_classes=3)
print(net)

# Load CUDA if exist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_net(net)

