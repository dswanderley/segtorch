
import torch
import torchvision.models as models
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from torch.autograd import Variable
from nets.unet import *



# Load CUDA if exist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = models.vgg16(pretrained=True).to(device)
model.eval()

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        if device.type == 'cpu':
            self.features = output.clone().detach().requires_grad_(True)
            #torch.tensor(output, requires_grad=True)
        else:
            self.features = output.clone().detach().requires_grad_(True).cuda()
            #torch.tensor(output, requires_grad=True).cuda()
    def close(self):
        self.hook.remove()



layer = 29
fltr = 180

activations = SaveFeatures(list(model.children())[0][layer])  # register hook


img = torch.rand(1,3,40,40)

img_var = Variable(img)  # convert image to Variable that requires grad
optimizer = torch.optim.Adam([img_var])

for n in range(500):  # optimize pixel values for opt_steps times
    optimizer.zero_grad()
    model(img_var)
    loss = -activations.features[0, fltr].mean()
    loss.backward()
    optimizer.step()

    img_np = img_var.data.cpu().numpy()[0].transpose(1,2,0)


plt.imshow(img_np)
plt.show()

print('')