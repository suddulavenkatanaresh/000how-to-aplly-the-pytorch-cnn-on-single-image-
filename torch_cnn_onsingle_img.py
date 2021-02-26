import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy
from torchvision import transforms 

from PIL  import Image 


im=Image.open('/home/vihal_venkat/Desktop/car.jpeg')


transforms=transforms.Compose([  transforms.Resize(250), 
                               transforms.CenterCrop(224),
                               
                               transforms.Grayscale(num_output_channels=3),
                               transforms.ToTensor()
                               ])


im=transforms(im)
#plt.imshow(im)

im=im.unsqueeze(0)
#v=im.shape[0]


#plt.imshow(im.numpy().transpose(1,2,0))

#print(im.size())
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5,stride=1,padding=0)
        
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(10*110*110, 50)
        self.fc2 = nn.Linear(50, 10)
        self.A=nn.Sigmoid()

    def forward(self, x):
        x=self.conv1(x)
        #x=F.relu(F.max_pool2d(x,2))
        #x=self.A(x)
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x=self.conv2(x)
        x=self.conv2_drop(x)
        x=F.max_pool2d(x,2)
        x=F.relu(x)
       # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        #x = x.view(-1,)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        ##x=F.log_softmax(x)
        return x
        
    
    
model=Net()
model.train()
out=model(im)
print(out.size())
