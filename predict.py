import torch
from torch.utils.data import DataLoader
from utils.losses import DiceCoefficients

class Inference():

    def __init__(self, model, device, weights_path):

        self.model = model
        self.device = device
        self.weights_path = weights_path
        self._load_network()
        self.criterion = DiceCoefficients()

    def _load_network(self):
        state = torch.load(self.weights_path)
        self.model.load_state_dict(state['state_dict'])


    def predict(self, images, save=True):
        '''
        Predict segmentation function

        Arguments:
            @param images: Testset
            @param save: save images (True) - not implemented
        '''

        self.model.eval()

        data_loader = DataLoader(images, batch_size=1, shuffle=False)
        for idx, (im_name, image, gt_mask, ov_mask, fol_mask) in enumerate(data_loader):

            # Active GPU
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
                image = image.to(self.device)
                gt_mask = gt_mask.to(self.device)
                #ov_mask = ov_mask.to(self.device)
                #fol_mask = fol_mask.to(self.device)

            # Handle with ground truth
            if len(gt_mask.size()) < 4:
                target = gt_mask.long()
            else:
                groundtruth = gt_mask.permute(0, 3, 1, 2).contiguous()

            # Prediction
            image.unsqueeze_(1) # add a dimension to the tensor, respecting the network input on the first postion (tensor[0])
            pred = self.model(image)
            # Handle multiples outputs
            if type(pred) is list:
                pred = pred[0]

            dsc = self.criterion(pred, groundtruth)

            print(im_name)
            print('Stroma DSC:    {:f}'.format(dsc[1]))
            print('Follicle DSC:  {:f}'.format(dsc[2]))