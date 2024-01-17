

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"


import torch  
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from skimage.io import imsave
# from skimage.measure import compare_ssim, compare_psnr, compare_mse
from skimage.io import imsave
# from skimage.measure import compare_mse, compare_ssim, compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse

import time
# import common
# import ops
# from models import  ENET
from GANmodels import Generator, Discriminator, Generator_SPA, Generator_SR

from sklearn.mixture import GaussianMixture as GMM
from torchsummary import summary

# Nx = 500
# Ny = 500
# z = 12000
# wavelength = 0.532
# deltaX = 4
# deltaY = 4
# mylamda = -5e-4  #5e-4#for background tvloss

image_size = 1000

Nx = image_size
Ny = image_size
# z = z
wavelength = 0.520
deltaX = 2
deltaY = 2

device = torch.device("cuda")


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def image_norm(image_path, image_size):
    # image_path = './samples/TS-20220224113123034.tif'
    image = cv2.imread(image_path)
    image = rgb2gray(image)
    h,w = image.shape
    min_len = h if h<=w else w 
    image_crop = image[(h//2-min_len//2):(h//2+min_len//2),
                       (w//2-min_len//2):(w//2+min_len//2)]
    image_500 = cv2.resize(image_crop, (image_size,image_size), interpolation = cv2.INTER_AREA)
    return image_500

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def unwrap(x):
    y = x % (2 * np.pi)
    return torch.where(y > np.pi, 2*np.pi - y, y)

def fft2dc(x):
    return np.fft.fftshift(np.fft.fft2(x))
  
def ifft2dc(x):
    return np.fft.ifft2(np.fft.fftshift(x))

def Phase_unwrapping(in_, s=image_size):
    f = np.zeros((s,s))
    for ii in range(s):
        for jj in range(s):
            x = ii - s/2
            y = jj - s/2
            f[ii,jj] = x**2 + y**2
    a = ifft2dc(fft2dc(np.cos(in_)*ifft2dc(fft2dc(np.sin(in_))*f))/(f+0.000001))
    b = ifft2dc(fft2dc(np.sin(in_)*ifft2dc(fft2dc(np.cos(in_))*f))/(f+0.000001))
    out = np.real(a - b)
    return out
    
    
    
def propagator_1(Nx,Ny,z,wavelength,deltaX,deltaY):
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


# def generate_holo(imge):
#     phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
#     E = np.ones((Nx,Ny))  # illumination light
#     E = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(np.conj(phase)))

#     Es = imge*E
#     S = np.fft.ifft2(np.fft.fft2(Es)*np.fft.fftshift(phase))

#     S1 = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(phase))

#     s=(S+1)*np.conj(S+1);
#     s1=(S1+1)*np.conj(S1+1);
#     g = s/s1

#     hologram = np.abs(g)
# #     plt.figure(figsize=(20,10))
# #     plt.imshow(hologram, cmap='gray')
#     gen_holo = (hologram-np.min(hologram))/(np.max(hologram)-np.min(hologram))
#     return gen_holo


def seg(img):
    critera = (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER,10,0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    data = np.float32(img.reshape(-1,1))
    r,best,center = cv2.kmeans(data,2,None,criteria=critera,attempts=10,flags=flags)
    # print(r)
    # print(best.shape)
    # print(center)
    center = np.uint8(center)

    if best.ravel()[0] == 0:
        data[best.ravel()==1] = (0)
        data[best.ravel()==0] = (255) 
    else:
        data[best.ravel()==1] = (255)
        data[best.ravel()==0] = (0) 
    # data[best.ravel()==2] = (0,0,255)
    # data[best.ravel()==3] = (0,255,0) 
    # data[best.ravel()==2] = (255)
    # data[best.ravel()==3] = (0) 

    data = np.uint8(data)
    mask = data.reshape((img.shape))
    mask = mask/255.
    # plt.imshow('img',img)
    # plt.imshow('res',oi)
    return mask
    
    
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

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def complex_mult(x, y):
    real_part = x[:,:,:,0]*y[:,:,:,0]-x[:,:,:,1]*y[:,:,:,1]
    real_part = real_part.unsqueeze(3)
    imag_part = x[:,:,:,0]*y[:,:,:,1]+x[:,:,:,1]*y[:,:,:,0]
    imag_part = imag_part.unsqueeze(3)
    return torch.cat((real_part, imag_part), 3)


def forward_propogate(x, z):
    x = x.squeeze(2)
#     y = y.squeeze(2)
    x = x.permute([0,2,3,1])
#     y = y.permute([0,2,3,1])
    prop = torch.tensor(propagator(Nx,Ny,z,wavelength,deltaX,deltaY)).to(device, dtype=torch.float)
    prop = prop.unsqueeze(0).unsqueeze(0)
    
    temp_x=torch.view_as_complex(x.contiguous())
    cEs = batch_fftshift2d(torch.view_as_real(torch.fft.fftn(temp_x, dim=(0,1,2), norm="ortho")))
    cEsp = complex_mult(cEs,prop)

    temp = torch.view_as_complex(batch_ifftshift2d(cEsp).contiguous())
    S = torch.view_as_real(torch.fft.ifftn(temp, dim=(0,1,2), norm="ortho") )
    Se = torch.sqrt(S[:,:,:,0]**2+S[:,:,:,1]**2)
    
    Se = Se.unsqueeze(0)
    return Se


    
class RECLoss(nn.Module):
    def __init__(self, z):
        super(RECLoss,self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.z = z
        self.wavelength =wavelength
        self.deltaX = deltaX
        self.deltaY = deltaY
        self.prop = self.propagator(self.Nx,self.Ny,self.z,self.wavelength,self.deltaX,self.deltaY)
        self.prop = self.prop.cuda()

    def propagator(self,Nx,Ny,z,wavelength,deltaX,deltaY):
        k = 1/wavelength
        # x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0)
        x =torch.unsqueeze(torch.arange(\
                                        torch.ceil(-torch.tensor(Nx)/2),torch.ceil(torch.tensor(Nx)/2),1)*(1/(Nx*deltaX)),dim=0)
        # y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1)
        y = torch.unsqueeze(torch.arange(torch.ceil(-torch.tensor(Ny)/2),torch.ceil(torch.tensor(Ny)/2),1)*(1/(Ny*deltaY)),dim=1)
        
        # print(x.shape)
        # print(y.shape)
        # y_new = np.repeat(y,Nx,axis=1)
        y_new = y.repeat(1, Nx)
        # x_new = np.repeat(x,Ny,axis=0)
        x_new = x.repeat(Ny,1)
        # print(y_new.shape)
        # print(x_new.shape)
        
        kp = torch.sqrt(y_new**2+x_new**2)
        term=k**2-kp**2
        term=np.maximum(term,0) 
        phase = torch.exp(1j*2*torch.pi*z*np.sqrt(term))
        # return torch.from_numpy(np.concatenate([np.real(phase)[np.newaxis,:,:,np.newaxis], np.imag(phase)[np.newaxis,:,:,np.newaxis]], axis = 3))
        return torch.cat([torch.real(phase).reshape(1,phase.shape[0],phase.shape[1],1), torch.imag(phase).reshape(1,phase.shape[0],phase.shape[1],1)], dim = 3)

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
    
    def TV(self, x, mask):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,1:,:,:])
        count_w = self._tensor_size(x[:,:,1:,:])
        h_tv = torch.pow((x[:,1:,:,:]-x[:,:h_x-1,:,:]),2).sum() #gradient in horizontal axis
        w_tv = torch.pow((x[:,:,1:,:]-x[:,:,:w_x-1,:]),2).sum() #gradient in vertical axis
        return 0.01*2*(h_tv/count_h+w_tv/count_w)/batch_size
    
    
#     def TV(self,x,mask):
#         batch_size = x.size()[0]
#         mask_tensor = torch.zeros((x.size())).to(device)
#         for i in range(batch_size):
#             mask_tensor[i,:,:,0] = mask
#             mask_tensor[i,:,:,1] = mask
#         h_x = x.size()[2]
#         w_x = x.size()[3]
        
#         count_h = self._tensor_size(x[:,1:,:,:])
#         count_w = self._tensor_size(x[:,:,1:,:])
#         x = torch.mul(x,mask_tensor)
#         amp = torch.sqrt(torch.pow(x[:,:,:,0],2)+torch.pow(x[:,:,:,1],2))
#         phase = torch.atan2(x[:,:,:,0],x[:,:,:,1])
# #         phase = (phase-torch.min(phase))/(torch.max(phase)-torch.min(phase))
# #         h_tv = torch.pow(phase[:,1:,:]-phase[:,:h_x-1,:],2).sum() #gradient in horizontal axis
# #         w_tv = torch.pow(phase[:,:,1:]-phase[:,:,:w_x-1],2).sum() #gradient in vertical axis
        
        
#         h_tv = torch.pow(x[:,1:,:,:]-x[:,:h_x-1,:,:],2).sum() #gradient in horizontal axis
#         w_tv = torch.pow(x[:,:,1:,:]-x[:,:,:w_x-1,:],2).sum() #gradient in vertical axis

# #         h_tv = 1*torch.pow(x[:,1:,:,0]-x[:,:h_x-1,:,0],2).sum()-torch.pow(x[:,1:,:,1]-x[:,:h_x-1,:,1],2).sum()   #gradient in horizontal axis
# #         w_tv = 1*torch.pow(x[:,:,1:,0]-x[:,:,:w_x-1,0],2).sum()-torch.pow(x[:,:,1:,1]-x[:,:,:w_x-1,1],2).sum() #gradient in vertical axis

#         return -2*(h_tv/count_h+w_tv/count_w)/batch_size #0.005 for cs prior
# #         return torch.sum(amp)/(batch_size*h_x*w_x)+torch.sum(phase)/(batch_size*h_x*w_x) #0.005 for cs prior
    
    
    def forward(self,x,y,mask,mylambda=0):
        x = x.squeeze(2)
        y = y.squeeze(2)
        x = x.permute([0,2,3,1])
        y = y.permute([0,2,3,1])
        
        # self.z = z.squeeze().cpu()
        # self.z = self.z.cpu()
        self.prop = self.propagator(self.Nx,self.Ny,self.z,self.wavelength,self.deltaX,self.deltaY)
        self.prop = self.prop.cuda()
        
        temp_x=torch.view_as_complex(x.contiguous())
          
       
        
        # cEs = self.batch_fftshift2d(torch.fft(x,3,normalized=True))
        
        cEs = self.batch_fftshift2d(torch.view_as_real (torch.fft.fftn(temp_x, dim=(0,1,2), norm="ortho")))
        
        cEsp = self.complex_mult(cEs,self.prop)
        
        # S = torch.ifft(self.batch_ifftshift2d(cEsp),3,normalized=True)
        
        temp = torch.view_as_complex(self.batch_ifftshift2d(cEsp).contiguous())
        S = torch.view_as_real(torch.fft.ifftn(temp, dim=(0,1,2), norm="ortho") )
        
        
        # Se = S[:,:,:,0]
        
        
        Se = torch.sqrt(S[:,:,:,0]**2+S[:,:,:,1]**2)
        # loss = torch.mean(torch.abs(Se-torch.sqrt(y[:,:,:,0])))/2#torch.mean(torch.abs(Se-y[:,:,:,0]))/2#
        loss = torch.mean(torch.abs(Se-torch.sqrt(y[:,:,:,0])))/2 + mylambda*self.TV(x,mask)#torch.mean(torch.abs(Se-y[:,:,:,0]))/2#

        # Se = S[:,:,:,:]
        
        # loss = torch.mean(torch.abs(Se-torch.sqrt(y[:,:,:,:])))/2#torch.mean(torch.abs(Se-y[:,:,:,0]))/2#
        
        
        return loss


    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
 




class BCELosswithLogits(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(BCELosswithLogits, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        logits = torch.sigmoid(logits)
        loss = - self.pos_weight * target * torch.log(logits) - \
               (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss 
        
    
    
    
    
# image_path = './samples/TS-20220314121315355.tif'
# z = 1650
# mylamda = 0
# epoch = 5000
    



def DIH_v32(image_path, z, mylamda, epoch):
    tl=time.localtime()
    format_time = time.strftime("%Y-%m-%d %H_%M_%S", tl) 
    print(format_time)

    time_start_all = time.time()
    time_start = time.time()
    
    
    # setup_seed(120)
    
    image_size = 1000
    
    Nx = image_size
    Ny = image_size
    z = z
    wavelength = 0.520
    deltaX = 2
    deltaY = 2
    mylamda = mylamda  #5e-4#for background tvloss
    # print(mylamda)
        
    
    # img = image_norm('./samples/TS-20220310163406351.tif', 1000)
    # # img = rgb2gray(np.array(Image.open('./target_mask3.bmp')))
    # # img = (np.array(Image.open('./target_mask3.bmp')))
    
    # img = np.sqrt(img)
    # img = (img-np.min(img))/(np.max(img)-np.min(img))
    # imsave('./gray.bmp',np.squeeze(img))
    # plt.figure(figsize=(20,10))
    # plt.imshow(np.squeeze(img), cmap='gray')
    
    phase = propagator_1(Nx,Ny,z,wavelength,deltaX,deltaY)
        
    # plt.figure(figsize=(20,10))
    # plt.imshow(np.abs(phase), cmap='gray')
    # np.fft.fftshift(phase)
    # plt.imshow(np.abs(np.fft.fftshift(phase)), cmap='gray')
    
    
        
    # E = np.ones((Nx,Ny))  # illumination light
    # E = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(np.conj(phase)))
    
    # Es = img*E
    # S = np.fft.ifft2(np.fft.fft2(Es)*np.fft.fftshift(phase))
    
    # S1 = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(phase))
    
    # s=(S+1)*np.conj(S+1);
    # s1=(S1+1)*np.conj(S1+1);
    # g = s/s1
    
    '''
    Holo
    '''
    hologram = image_norm(image_path, 1000)
    cv2.imwrite('./results/holo.bmp',hologram)

    hologram = (hologram-np.min(hologram))/(np.max(hologram)-np.min(hologram))
    
    plt.figure(figsize=(20,10))
    plt.imshow(np.squeeze(hologram), cmap='gray')
    
    
    
    bp = np.fft.ifft2(np.fft.fft2(hologram)*np.fft.fftshift(np.conj(phase)))
    plt.figure(figsize=(20,10))
    plt.imshow(np.abs(bp), cmap='gray')
    
    
    device = torch.device("cuda")
    
    
    criterion = RECLoss(z) #ONLY FOR GENERATOR
    criterion_2 = BCELosswithLogits() # FOR G AND
    # G = Generator().to(device)
    G = Generator_SR().to(device) # enable super resolution
    D = Discriminator().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=0.009)#9e-3
    optimizer_D = optim.Adam(D.parameters(), lr=0.005)#5e-3
    
    
    
    epoch = epoch
    period = 100
    period_train = 5 # train 5 times D and train G once
    eta = torch.Tensor(np.concatenate([np.abs(bp)[np.newaxis,:,:], np.zeros_like(np.abs(bp))[np.newaxis,:,:]], axis = 0))
    
    #back-progated holo
    holo = torch.Tensor(np.concatenate([np.real(hologram)[np.newaxis,:,:], np.imag(hologram)[np.newaxis,:,:]], axis = 0))
    
    eta = eta.to(device).unsqueeze(0)
    eta_500 = F.interpolate(eta, scale_factor=0.5, mode="bilinear")
    
    holo = holo.to(device).unsqueeze(0)
    holo_500 = F.interpolate(holo, scale_factor=0.5, mode="bilinear")
    
    
    # load the ground truth to compare
    # ground_truth = (np.array(Image.open('./gray.bmp')))
    ground_truth = np.abs(bp)
    ground_truth = (ground_truth-np.min(ground_truth))/(np.max(ground_truth)-np.min(ground_truth))
    
    
    t0 = 0 # 1e-2 # initial simulated annealing
    
    # temp_mask = mask # set mask as numpy and used to update the mask
    
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()
    
    # mask =  torch.tensor(mask).to(device)
    # mask = torch.ones(img.shape).to(device)
    mask = torch.ones(hologram.shape).to(device)
    
    
    D_loss = []
    G_loss = []
    A_loss = []
    PSNR_list = []
    SSIM_list = []
    Temp_amp = []
    Temp_phase = []
    Mask_list = []
    Mask_flag_1 = []
    Mask_flag_2 = [] 
    for i in range(epoch):
        batch_size =1
        real_labels = (0.2*torch.ones(batch_size, 1)+0.8).to(device)  
        fake_labels = torch.zeros(batch_size, 1).to(device)-real_labels # 

        for j in range(period_train):             
            ## Train D per k epoch
            
            # real loss: BCE_Loss(x, y): - y * log(D(x)) 
            outputs = D(holo_500[:,0,:,:].unsqueeze(1))
            d_loss_real = criterion_2(outputs, real_labels) #bce(pred_real,true_label)
            real_score = outputs
        
            # fake loss: - (1-y) * log(1 - D(x))
            fake_images = G(eta_500) # 1000x1000
            # fake_images_500 = F.interpolate(fake_images, scale_factor=0.5, mode="bilinear") # 500x500
            fp = forward_propogate(fake_images, z)
            G_holo_500 = F.interpolate(fp, scale_factor=0.5, mode="bilinear")   
            outputs = D(G_holo_500)
            d_loss_fake = criterion_2(outputs, fake_labels) #bce(pred_fake,true_fake)
            fake_score = outputs
            
            # Back propgate
            d_loss = d_loss_real + d_loss_fake # -10*criterion(fake_images, holo) 
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            d_loss.backward()
            optimizer_D.step()
    
        D_loss.append(d_loss.cpu().data.numpy())
        
        ## Train G : maximize log(D(G(z))
        fake_images = G(eta_500)
        # fake_images = F.interpolate(fake_images, scale_factor=0.5, mode="bilinear") # 500x500
        out = fake_images
        G_holo_500 = F.interpolate(forward_propogate(out,z ), scale_factor=0.5, mode="bilinear")   
        outputs = D(G_holo_500)  #  the generated holo from fake image

        if i >101:
            auto_loss = criterion(fake_images,holo,mask,mylamda) 
        else:
            auto_loss = criterion(fake_images,holo,mask,0) 
        g_loss = criterion_2(outputs, real_labels)+10*auto_loss  #bce_loss(pred_fake, true_labels)
        A_loss.append(auto_loss.cpu().data.numpy())
        G_loss.append(g_loss.cpu().data.numpy())
        
        # back propgate
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        

    #     mask_tensor = torch.zeros((eta.size())).to(device)
    #     mask_tensor[0,0,:,:] = mask.clone().detach()
    #     mask_tensor[0,1,:,:] = mask.clone().detach()
    #     eta = (torch.mul(eta,mask_tensor)*0.5 + torch.mul(eta,-(mask_tensor-1))).to(device)
    #     loss = criterion(out, holo) 
    #     loss.backward()
    #     optimizer.step()
    #     out = model(eta) 
        
        
        #print('epoch [{}/{}]     Loss: {}'.format(i+1, epoch, loss.cpu().data.numpy()))
        if ((i+1) % period) == 0:
            time_end = time.time()
            print('epoch [{}/{}]     Loss: {}   Time: {}'.format(i+1, epoch, auto_loss.cpu().data.numpy(), time_end-time_start))
            time_start = time.time()
            
            outtemp = out.cpu().data.squeeze(0)
            outtemp = outtemp
            plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
            plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
            
            Temp_amp.append(tensor2pil(plotout))
            
            PSNR_list.append(compare_psnr(ground_truth,np.array(tensor2pil(plotout))/255.))
            SSIM_list.append(compare_ssim(ground_truth,np.array(tensor2pil(plotout))/255.))
            
            
            plotout_p = outtemp.numpy()
    #         print('phase scale')
    #         print(plotout_p[0,100,:10])
    #         print(plotout_p[1,100,:10])
        
            plotout_p = np.arctan2(plotout_p[0,:,:], plotout_p[1,:,:])
    #         print(plotout_p[100,:10])
            plotout_p = Phase_unwrapping(plotout_p)
        
        
            plt.figure(figsize=(10,10))
            plt.imshow(tensor2pil(plotout), cmap='gray')
            plt.axis('off')
            plt.title('i: '+str(i),fontsize=20)
            plt.show()
            
            
            
            plt.figure(figsize=(10,10))
            plt.imshow((plotout_p), cmap='gray')
            plt.axis('off')
            plt.show()
            
    # #         print(plotout_p[100,:10])
            # #print(np.min(plotout_p))
            # plt.figure(figsize=(10,10))
            # plt.plot((plotout_p[100,:]))
            # plt.show()
            plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
            # print('epoch [{}/{}]     PSNR: {} |  SSIM: {} | Phase PSNR: {}'.format(i+1, epoch, compare_psnr(ground_truth,np.array(tensor2pil(plotout))/255.),compare_ssim(ground_truth,np.array(tensor2pil(plotout))/255.),compare_psnr(ground_truth,np.array(plotout_p)/255.)))
            
            
            Temp_phase.append(plotout_p)
            
            
            
            mask_tensor = torch.zeros((out.size())).to(device)
            
            mask_tensor[0,0,:,:] = mask
            mask_tensor[0,1,:,:] = mask
    
            x_mask = torch.mul(out,mask_tensor)*0 + torch.mul(out,(1-mask_tensor)) #make background to zero ->holo
            current_mask_loss = criterion(x_mask,holo,mask,0) 
            
            mask_new = mask
            # mask_new = seg(plotout)
            # plt.figure(figsize=(10,10))
            # plt.imshow(mask_new)
            # mask_new_2,prob = seg_gmm(plotout)
            # plt.figure(figsize=(10,10))
            # plt.imshow(mask_new_2)
            
            
            
            mask_new = torch.tensor(mask_new).to(device)
            
            mask_new_tensor = torch.zeros((out.size())).to(device)
            
            mask_new_tensor[0,0,:,:] = mask_new
            mask_new_tensor[0,0,:,:] = mask_new
            
            x_mask_new = torch.mul(out,mask_new_tensor)*0 + torch.mul(out,(1-mask_new_tensor))
            new_mask_loss = criterion(x_mask_new,holo,mask,0) 
            
            
            '''
            simulated annealing
            '''
            delta_t = new_mask_loss - current_mask_loss
            if delta_t<0:
                mask = mask_new
                Mask_list.append(mask.cpu().data)
                Mask_flag_1.append(1)
                Mask_flag_2.append(1)
            else:
                Mask_flag_1.append(0)
                p = torch.exp(-delta_t/t0)
                if torch.rand(1).to(device)<p:
                    mask = mask_new
                    Mask_list.append(mask.cpu().data)
                    Mask_flag_2.append(1)
                else:
                    Mask_flag_2.append(0)
            t0 = t0 / np.log(1 + i)
        
            
            plt.figure(figsize=(10,10))
            plt.imshow(mask.cpu().data)
            plt.axis('off')
    
            
            
            
    max_index = A_loss.index(min(A_loss))//100
    # max_index = len(PSNR_list)-1
    # max_index = 9
    
    index = max_index
    
    img_save = np.array((Temp_amp[index]))/255.
    imsave('./rec_gan_'      +str(z)+'_'+str(wavelength)+'_'+str(period_train)+'_'+str(mylamda)+'_'+format_time+'.bmp',np.uint8(np.squeeze(img_save)*255))
    imsave('./rec_phase_gan_'+str(z)+'_'+str(wavelength)+'_'+str(period_train)+'_'+str(mylamda)+'_'+format_time+'.bmp',np.uint8(np.squeeze(Temp_phase[index])*255))
    imsave('./holo_'+str(z)+'_'+str(wavelength)+'_'+format_time+'.bmp',np.squeeze(hologram))
    imsave('./bp_'  +str(z)+'_'+str(wavelength)+'_'+format_time+'.bmp',np.squeeze(np.abs(bp)))
    imsave('./bp_real_'+str(z)+'_'+str(wavelength)+'_'+format_time+'.bmp',np.squeeze(np.real(bp)))
    imsave('./bp_imag_'+str(z)+'_'+str(wavelength)+'_'+format_time+'.bmp',np.squeeze(np.imag(bp)))
    
    
    
    
    strcontent = str(mylamda)+'_'+str(wavelength)+ ' PSNR:'+str(PSNR_list[max_index])+'   SSIM:'+str(SSIM_list[max_index])+'_'+format_time
    f = open("gan_eval.txt",'a+')
    f.write(strcontent)
    f.write('\n')
    
    
    
    
    # np.save(format_time+'mask_flag_1.npy',np.array(Mask_flag_1))
    # np.save(format_time+'mask_flag_2.npy',np.array(Mask_flag_2))
    
    
    
    
    # #  background flatteness
    # img = (np.array(Image.open('./target_mask3.bmp')))
    # background = np.zeros(img.shape)
    # background[img==0]=1
    # phase = np.squeeze(Temp_phase[index])
    # h_x = phase.shape[0]
    # w_x =  phase.shape[1]
    # h_tv =np.linalg.norm (np.multiply(phase[1:,:]-phase[:h_x-1,:],background[1:,:])) #gradient in horizontal axis
    # w_tv =np.linalg.norm (np.multiply(phase[:,1:]-phase[:,:w_x-1],background[:,1:])) #gradient in vertical axis
    # flat = (h_tv+ w_tv)/(h_x*w_x)
    # print(flat)
    
    
    time_end_all = time.time()
    print('Done, total time cost: ', time_end_all-time_start_all)
    
    return Temp_amp, Temp_phase
    
    
if __name__ == '__main__':
        
    DIH_v32('./samples/TS-20220314115708421.tif', z=1700, mylamda=0, epoch=5000)    
    DIH_v32('./samples/TS-20220314115727220.tif', z=1600, mylamda=0, epoch=5000)    

    
    DIH_v32('./samples/TS-20220314120059941.tif', z=1700, mylamda=0, epoch=5000)    
    
    DIH_v32('./samples/TS-20220314120333012.tif', z=1700, mylamda=0, epoch=5000)    
    
    DIH_v32('./samples/TS-20220314120954414.tif', z=1650, mylamda=0, epoch=5000)    
    DIH_v32('./samples/TS-20220314122041352.tif', z=1650, mylamda=0, epoch=5000)    
    DIH_v32('./samples/TS-20220314121324493.tif', z=1650, mylamda=0, epoch=5000)    
    DIH_v32('./samples/TS-20220314121530284.tif', z=1650, mylamda=0, epoch=5000)    
    
    
    
    # DIH_v32('./samples/TS-20220314120536224.tif', z=1600, mylamda=0, epoch=5000) 
    # DIH_v32('./samples/TS-20220314120218002.tif', z=1600, mylamda=0, epoch=5000)    
    # DIH_v32('./samples/TS-20220314115904643.tif', z=1600, mylamda=0, epoch=5000)    
    # DIH_v32('./samples/TS-20220314120452319.tif', z=1600, mylamda=0, epoch=5000)    
    
    # DIH_v32('./samples/TS-20220314121315355.tif', z=1650, mylamda=0, epoch=5000)    

    

    # DIH_v32('./samples/TS-20220323160155098.tif', z=1800, mylamda=0, epoch=5000)    

    # DIH_v32('./samples/TS-20220323163844679.tif', z=1700, mylamda=0, epoch=5000)    


    # DIH_v32('./samples/TS-20220323164658438.tif', z=1500, mylamda=0, epoch=5000)    

    # DIH_v32('./samples/TS-20220402201706674.tif', z=9550, mylamda=0, epoch=5000)    
    
    # DIH_v32('./samples/TS-20220310163413358.tif', z=1500, mylamda=0, epoch=3000) 

    
    # [Temp_amp, Temp_phase] = DIH_v32('./samples/TS-20220402201632996.tif', z=9500, mylamda=0, epoch=5000)    





    # DIH_v32('./samples/TS-20220310163413358.tif', z=1500, mylamda=0, epoch=3000) 
    # DIH_v32('./samples/TS-20220314115727220.tif', z=1600, mylamda=0, epoch=3000)    
    # DIH_v32('./samples/TS-20220314120218002.tif', z=1600, mylamda=0, epoch=5000)    


    # DIH_v32('./samples/TS-20220323160155098.tif', z=1800, mylamda=0, epoch=5000)    


    # DIH_v32('./samples/TS-20220314121315355.tif', z=1650, mylamda=0, epoch=5000)    

























