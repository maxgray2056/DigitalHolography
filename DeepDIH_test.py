# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:05:04 2021

@author: MaxGr
"""

import torch  
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image, ImageOps

import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"




print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print(torch.cuda.get_device_name(device_default))
device = torch.device("cuda")
print(torch.version.cuda)
print(torch.__version__)
print(torch.cuda.get_arch_list())



img_size = 1000

'''
Spherical light function
Nx, Ny : hologram size
z : object-sensor distance 
wavelength: wavelength of light
deltaX, deltaY : sensor size
'''
# um
Nx = 1000
Ny = 1000
z = 90000
wavelength = 0.520
deltaX = 2.0
deltaY = 2.0




'''

'''
img = Image.open('./samples/90mm.tif')
img = ImageOps.grayscale(img)

h,w = img.size
if h != 1000 or w != 1000:
    img = img.resize((1000,1000), PIL.Image.LANCZOS)

img.show()

# pytorch provides a function to convert PIL images to tensors.
pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

tensor_img = pil2tensor(img)

g = tensor_img.numpy()
g = np.sqrt(g)
g = (g-np.min(g))/(np.max(g)-np.min(g))

plt.figure(figsize=(20,15))
plt.imshow(np.squeeze(g), cmap='gray')





'''
Phase Unwap and fft
'''
def unwrap(x):
    y = x % (2 * np.pi)
    return torch.where(y > np.pi, 2*np.pi - y, y)

def fft2dc(x):
    return np.fft.fftshift(np.fft.fft2(x))
  
def ifft2dc(x):
    return np.fft.ifft2(np.fft.fftshift(x))

def Phase_unwrapping(in_):
    f = np.zeros((1000,1000))
    for ii in range(1000):
        for jj in range(1000):
            x = ii - 1000/2
            y = jj - 1000/2
            f[ii,jj] = x**2 + y**2
    a = ifft2dc(fft2dc(np.cos(in_)*ifft2dc(fft2dc(np.sin(in_))*f))/(f+0.000001))
    b = ifft2dc(fft2dc(np.sin(in_)*ifft2dc(fft2dc(np.cos(in_))*f))/(f+0.000001))
    out = np.real(a - b)
    return out




def propagator(Nx,Ny,z,wavelength,deltaX,deltaY):
    k = 1/wavelength
    x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0)
    y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1)
    y_new = np.repeat(y,Nx,axis=1)
    x_new = np.repeat(x,Ny,axis=0)
    kp = np.sqrt(y_new**2+x_new**2)
    term=k**2-kp**2
    term=np.maximum(term,0) 
    phase = np.exp(1j*2*np.pi*z*np.sqrt(term))
    return phase


'''
Back-propogation
'''
phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
eta = np.fft.ifft2(np.fft.fft2(g)*np.fft.fftshift(np.conj(phase)))
plt.figure(figsize=(20,15))
plt.imshow(np.squeeze(np.abs(eta)), cmap='gray')


new_holo = ifft2dc(np.fft.fft2(eta)*np.fft.fftshift(phase))
plt.figure(figsize=(20,15))
plt.imshow(np.squeeze(np.abs(new_holo)), cmap='gray')

bp = np.squeeze(np.abs(eta))
bp = bp/(np.max(bp)-np.min(bp)) *255


cv2.imwrite('./bp.png',bp)

# a = np.fft.fftshift(np.conj(phase))
# b = np.conj(phase)
# c = np.squeeze(np.abs(np.fft.fft2(g)))
# c = c/(np.max(c)-np.min(c)) * 255

# plt.imshow(np.abs(a))
# plt.imshow(np.abs(b))
# plt.imshow(np.squeeze(g))

# plt.imshow(kp)
# plt.imshow(np.squeeze(np.abs(phase)), cmap='gray')




'''
Define loss function
'''
class RECLoss(nn.Module):
    def __init__(self):
        super(RECLoss,self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.z = z
        self.wavelength = wavelength
        self.deltaX = deltaX
        self.deltaY = deltaY
        self.prop = self.propagator(self.Nx,self.Ny,self.z,self.wavelength,self.deltaX,self.deltaY)
        self.prop = self.prop.cuda()

    def propagator(self,Nx,Ny,z,wavelength,deltaX,deltaY):
        k = 1/wavelength
        x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0)
        y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1)
        y_new = np.repeat(y,Nx,axis=1)
        x_new = np.repeat(x,Ny,axis=0)
        kp = np.sqrt(y_new**2+x_new**2)
        term=k**2-kp**2
        term=np.maximum(term,0) 
        phase = np.exp(1j*2*np.pi*z*np.sqrt(term))
        return torch.from_numpy(np.concatenate([np.real(phase)[np.newaxis,:,:,np.newaxis], np.imag(phase)[np.newaxis,:,:,np.newaxis]], axis = 3))
   

    def roll_n(self, X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def batch_fftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim)//2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def batch_ifftshift2d(self,x):
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim)//2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim)//2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
    
    def complex_mult(self, x, y):
        real_part = x[:,:,:,0]*y[:,:,:,0]-x[:,:,:,1]*y[:,:,:,1]
        real_part = real_part.unsqueeze(3)
        imag_part = x[:,:,:,0]*y[:,:,:,1]+x[:,:,:,1]*y[:,:,:,0]
        imag_part = imag_part.unsqueeze(3)
        return torch.cat((real_part, imag_part), 3)

    def forward(self,x,y):
        batch_size = x.size()[0]
        
        x = x.squeeze(2)
        y = y.squeeze(2)
        x = x.permute([0,2,3,1])
        y = y.permute([0,2,3,1])
        
        cEs = self.batch_fftshift2d(torch.fft(x,3,normalized=True))
        cEsp = self.complex_mult(cEs,self.prop)
        
        # forward propogate
        # reconstrut_freq = torch.log(torch.abs(self.batch_fftshift2d(torch.fft(x,3,normalized=True)) )+1e-5)
        reconstrut_freq = cEsp
        reconstrut_freq=(reconstrut_freq-torch.min(reconstrut_freq))/(torch.max(reconstrut_freq)-torch.min(reconstrut_freq))
        
        capture_freq =  torch.log( torch.abs(self.batch_fftshift2d(torch.fft(y,3,normalized=True) ))+1e-5)
        capture_freq=(capture_freq-torch.min(capture_freq))/(torch.max(capture_freq)-torch.min(capture_freq))
        
        h_x = x.size()[1]
        w_x = x.size()[2]
        
        h_tv_x = torch.pow((reconstrut_freq[:,1:,:,:]-reconstrut_freq[:,:h_x-1,:,:]),2).sum()
        w_tv_x = torch.pow((reconstrut_freq[:,:,1:,:]-reconstrut_freq[:,:,:w_x-1,:]),2).sum()
        
        #print(reconstrut_freq.shape)
        h_tv_y = torch.pow((capture_freq[:,1:,:,:]-capture_freq[:,:h_x-1,:,:]),2).sum()
        w_tv_y = torch.pow((capture_freq[:,:,1:,:]-capture_freq[:,:,:w_x-1,:]),2).sum()
        
        count_h = self._tensor_size(x[:,1:,:,:])
        count_w = self._tensor_size(x[:,:,1:,:])
        
        tv_diff = 2*(h_tv_x/count_h+w_tv_x/count_w)/batch_size - 2*(h_tv_y/count_h+w_tv_y/count_w)/batch_size
        print(0.01*tv_diff)
        
        S = torch.ifft(self.batch_ifftshift2d(cEsp),3,normalized=True)
        Se = S[:,:,:,0]
        
        mse = torch.mean(torch.abs(Se-y[:,:,:,0]))/2-0.01*tv_diff
        return mse

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


'''
discrete wavelet transform
'''
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


'''
Define Network
'''
# finish the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
        )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 16, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
        
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
        )
        
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            #nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            #nn.BatchNorm2d(16),
            nn.Conv2d(16, 2, 3, stride=1, padding=1),
        )
        
    
    def forward(self,x):
        x = x.float()
        x = self.conv_init(x)
        x = dwt_init(x)
        x = self.conv_1(x)
        x = dwt_init(x)
        x = self.conv_2(x)
        x = dwt_init(x)
        x = self.conv_nonlinear(x)
        
        x = self.deconv_1(x)
        x = iwt_init(x)
        x = self.deconv_2(x)
        x = iwt_init(x)
        x = self.deconv_3(x)
        x = iwt_init(x)
        x = self.deconv_4(x)
        return x


from torchsummary import summary
criterion_1 = RECLoss()
model = Net().cuda()
optimer_1 = optim.Adam(model.parameters(), lr=5e-3)


device = torch.device("cuda")
epoch_1 = 5000
epoch_2 = 2000
period = 100
eta = torch.from_numpy(np.concatenate([np.real(eta)[np.newaxis,:,:], np.imag(eta)[np.newaxis,:,:]], axis = 1))
holo = torch.from_numpy(np.concatenate([np.real(g)[np.newaxis,:,:], np.imag(g)[np.newaxis,:,:]], axis = 1))


for i in range(epoch_1):
    in_img = eta.to(device)
    target = holo.to(device)
    
    out = model(in_img) 
    l1_loss = criterion_1(out, target)
    loss = l1_loss
    
    optimer_1.zero_grad()
    loss.backward()
    optimer_1.step()
    
    print('epoch [{}/{}]     Loss: {}'.format(i+1, epoch_1, l1_loss.cpu().data.numpy()))
    if ((i+1) % period) == 0:
        outtemp = out.cpu().data.squeeze(0).squeeze(1)
        outtemp = outtemp
        plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
        plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
        plt.figure(figsize=(20,15))
        plt.imshow(tensor2pil(plotout), cmap='gray')
        plt.show()
        
        plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
        plotout_p = Phase_unwrapping(plotout_p)
        plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
        plt.figure(figsize=(20,15))
        plt.imshow((plotout_p), cmap='gray')
        plt.show()
        
        
outtemp = out.cpu().data.squeeze(0).squeeze(1)
outtemp = outtemp
plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
plt.figure(figsize=(30,30))
plt.imshow(tensor2pil(plotout), cmap='gray')
plt.show()


plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
plotout_p = Phase_unwrapping(plotout_p)
plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
plt.figure(figsize=(30,30))
plt.imshow((plotout_p), cmap='gray')
plt.show()        





#cv2.imwrite("./penalty_1/1_amp.png",tensor2pil(plotout))

torch.__version__


type(tensor2pil(plotout))

amp =tensor2pil(plotout)
amp.save("./penalty_1/1_amp.png")



import cv2
cv2.imwrite("phase.png",plotout_p)









