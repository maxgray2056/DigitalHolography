# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:55:09 2022

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import math
import copy
import torch  
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from skimage.io import imsave
# from skimage.measure import compare_mse, compare_ssim, compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse

'''
Random seed
'''
SEED=5
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

def standardization(data):
    u = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data-u)/sigma
def norm(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))
def unwrap(x):
    y = x % (2 * np.pi)
    return torch.where(y > np.pi, 2*np.pi - y, y)
def fft2dc(x):
    return np.fft.fftshift(np.fft.fft2(x))
def ifft2dc(x):
    return np.fft.ifft2(np.fft.fftshift(x))
def Phase_unwrapping(in_, s=500):
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

image_size = 500

# um
Nx = 500
Ny = 500
z = 12000
wavelength = 0.532
deltaX = 4
deltaY = 4

'''
Load image
'''
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = rgb2gray(np.array(Image.open('./1951usaf_test_target.jpg')))
#img = np.sqrt(img)
img = (img-np.min(img))/(np.max(img)-np.min(img))
imsave('./gray.bmp',np.squeeze(img))
plt.figure(figsize=(20,10))
plt.imshow(np.squeeze(img), cmap='gray')

'''
Field mesurement/generate hologram
'''
phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
E = np.ones((Nx,Ny))  # illumination light
E = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(np.conj(phase)))
Es = img*E
S = np.fft.ifft2(np.fft.fft2(Es)*np.fft.fftshift(phase))
S1 = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(phase))
s=(S+1)*np.conj(S+1);
s1=(S1+1)*np.conj(S1+1);
g = s/s1

hologram = np.abs(g)

plt.figure(figsize=(20,10))
plt.imshow(hologram, cmap='gray')

'''
Back-propagate
'''
bp = np.fft.ifft2(np.fft.fft2(hologram)*np.fft.fftshift(np.conj(phase)))
plt.figure(figsize=(20,10))
plt.imshow(np.abs(bp), cmap='gray')




'''
Save 1st iter
'''
holo_1 = copy.deepcopy(hologram)
bp_1 = copy.deepcopy(bp)


'''
bp to bp
'''
img = np.abs(bp)
phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
E = np.ones((Nx,Ny))  # illumination light
E = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(np.conj(phase)))
Es = img*E
S = np.fft.ifft2(np.fft.fft2(Es)*np.fft.fftshift(phase))
S1 = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(phase))
s=(S+1)*np.conj(S+1);
s1=(S1+1)*np.conj(S1+1);
g = s/s1
hologram = np.abs(g)
plt.figure(figsize=(20,10))
plt.imshow(hologram, cmap='gray')
bp = np.fft.ifft2(np.fft.fft2(hologram)*np.fft.fftshift(np.conj(phase)))
plt.figure(figsize=(20,10))
plt.imshow(np.abs(bp), cmap='gray')

# 2nd iter
holo_2 = copy.deepcopy(hologram)
bp_2 = copy.deepcopy(bp)


'''
Unite scales
'''
plt.plot(np.unique(holo_1))
plt.plot(np.unique(np.abs(bp_1)))
holo_1_norm = (holo_1-np.mean(holo_1))*(np.max(holo_1)-np.min(holo_1))
bp_1_norm = np.abs(bp_1)
bp_1_norm = (bp_1_norm-np.mean(bp_1_norm))*(np.max(bp_1_norm)-np.min(bp_1_norm))

plt.plot(np.unique(holo_2))
plt.plot(np.unique(np.abs(bp_2)))
holo_2_norm = (holo_2-np.mean(holo_2))*(np.max(holo_2)-np.min(holo_2))
bp_2_norm = np.abs(bp_2)
bp_2_norm = (bp_2_norm-np.mean(bp_2_norm))*(np.max(bp_2_norm)-np.min(bp_2_norm))


scale_1 = (np.max(holo_1_norm)-np.min(holo_1_norm))
scale_2 = (np.max(holo_2_norm)-np.min(holo_2_norm))
holo_2_norm = holo_2_norm*(scale_1/scale_2)
plt.plot(np.unique(holo_1_norm))
plt.plot(np.unique(holo_2_norm))

scale_3 = (np.max(bp_1_norm)-np.min(bp_1_norm))
scale_4 = (np.max(bp_2_norm)-np.min(bp_2_norm))
bp_2_norm = bp_2_norm*(scale_3/scale_4)
plt.plot(np.unique(bp_1_norm))
plt.plot(np.unique(bp_2_norm))


plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(holo_1_norm+holo_2_norm, cmap='gray')

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(bp_1_norm+bp_2_norm, cmap='gray')




# img = plotout.numpy()
img = np.abs(bp)

img = rgb2gray(np.array(Image.open('./1951usaf_test_target.jpg')))
img = (img-np.min(img))/(np.max(img)-np.min(img))

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(img, cmap='gray')


holo_sum = []
bp_sum = []

dist = 12000

for z in range(0,5001,500):
    print(z)
        
    # for wavelength in range(485,650,5):    
        
    #     wavelength = wavelength/1000
    #     print(wavelength)
    
    # for deltaX in range(1,8):    
    #     deltaY = deltaX
    #     print(deltaY)
    
    '''
    Field mesurement/generate hologram
    '''
    phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
    E = np.ones((Nx,Ny))  # illumination light
    E = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(np.conj(phase)))
    
    Es = img*E
    S = np.fft.ifft2(np.fft.fft2(Es)*np.fft.fftshift(phase))
    
    S1 = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(phase))
    
    s=(S+1)*np.conj(S+1);
    s1=(S1+1)*np.conj(S1+1);
    g = s/s1
    
    hologram = np.abs(g)
    
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(hologram, cmap='gray')
    
    '''
    Back-propagate
    '''
    bp = np.fft.ifft2(np.fft.fft2(hologram)*np.fft.fftshift(np.conj(phase)))
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(np.abs(bp), cmap='gray')
    
    holo_sum.append(hologram)
    bp_sum.append(bp)
    
        
holo_total = 0
bp_total = 0    

    


for i in range(len(bp_sum)):
    # holo_sum[i] = standardization(holo_sum[i])
    holo_total = holo_total + holo_sum[i]
    
    # bp_sum[i] = standardization(bp_sum[i])
    bp_total = bp_total + bp_sum[i]
    
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(holo_total, cmap='gray')


# a = holo_sum[0]+holo_sum[1]+holo_sum[2]+holo_sum[3]+holo_sum[4]
# plt.imshow(a, cmap='gray')



plt.figure(figsize=(10,10))
plt.axis('off')
# plt.imshow(np.real(bp_total), cmap='gray')
# plt.imshow(np.imag(bp_total), cmap='gray')
plt.imshow(np.abs(bp_total), cmap='gray')





holo_sum_1 = copy.deepcopy(holo_total)
bp_sum_1 = copy.deepcopy(bp_total)



holo_sum_2 = copy.deepcopy(holo_total)
bp_sum_2 = copy.deepcopy(bp_total)

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(holo_sum_1+holo_sum_2, cmap='gray')
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(np.abs(bp_sum_1)+np.abs(bp_sum_2), cmap='gray')


a = holo_sum_1
b = bp_sum_1



# a = (holo_sum_1-np.mean(holo_sum_1))*(np.max(holo_sum_1)-np.min(holo_sum_1))
plt.plot(np.unique(a))

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(a*np.abs(bp_sum_1), cmap='gray')


plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(holo_sum_2*np.abs(bp_sum_2), cmap='gray')





a = norm(holo_sum_1) + norm(holo_sum_2)
b = norm(bp_sum_1) + norm(bp_sum_2)

plt.imshow(np.abs(b))












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




image_size = 1000

# um
Nx = image_size
Ny = image_size
z = 1800
wavelength = 0.520
deltaX = 2
deltaY = 2

# for z in range(9397500,9398500+1,20):

hologram = image_norm('./samples/TS-20220323160155098.tif', image_size)
# hologram = rgb2gray(np.array(Image.open('./samples/TS-20220310163406351.tif')))
hologram = (hologram-np.min(hologram))/(np.max(hologram)-np.min(hologram))
plt.figure(figsize=(20,10))
plt.imshow(np.squeeze(hologram), cmap='gray')



phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
# E = np.ones((Nx,Ny))  # illumination light
# E = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(np.conj(phase)))
bp = np.fft.ifft2(np.fft.fft2(hologram)*np.fft.fftshift(np.conj(phase)))
plt.figure(figsize=(20,10))
# plt.title('i: '+str(i),fontsize=20)
plt.imshow(np.abs(bp), cmap='gray')

# hologram_binary = hologram
# hologram_binary[hologram_binary>=np.mean(hologram_binary)]=255
# hologram_binary[hologram_binary<=np.mean(hologram_binary)]=0
# plt.imshow(hologram_binary,cmap='gray')
# cv2.imwrite('1951_USAF_1000x1000.png', hologram_binary)

# holo_sum = []
# bp_sum = []
# z_iter = 1000000
# num_iter = 50
# z_center = 10000
# z_left = z_center-num_iter
# z_right = z_center+num_iter+1
# num_image = 30


for i in range(1700,1900,20):
    # print(z)
    z = i
    # deltaX = i/10
    # deltaY = i/10
    
    # z = 941000
    phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
    # E = np.ones((Nx,Ny))  # illumination light
    # E = np.fft.ifft2(np.fft.fft2(E)*np.fft.fftshift(np.conj(phase)))
    bp = np.fft.ifft2(np.fft.fft2(hologram)*np.fft.fftshift(np.conj(phase)))
    plt.figure(figsize=(20,10))
    plt.title('i: '+str(i),fontsize=20)
    plt.imshow(np.abs(bp), cmap='gray')
        # holo_sum.append(hologram)
    # bp_sum.append(bp)
        
    
holo_total = 0
bp_total = 0    

for i in range(len(bp_sum)):
    # holo_sum[i] = standardization(holo_sum[i])
    # holo_total = holo_total + holo_sum[i]
    
    # bp_sum[i] = standardization(bp_sum[i])
    bp_total = bp_total + bp_sum[i]
    
# plt.figure()
# plt.imshow(holo_total, cmap='gray')

plt.figure(figsize=(20,10))
plt.imshow(np.real(bp_total), cmap='gray')
plt.figure(figsize=(20,10))
plt.imshow(np.imag(bp_total), cmap='gray')
plt.figure(figsize=(20,10))
plt.imshow(np.abs(bp_total), cmap='gray')




plt.figure(figsize=(20,10))
plt.imshow(np.abs(bp), cmap='gray')


bp_save = (np.abs(bp)-np.min(np.abs(bp)))/(np.max(np.abs(bp))-np.min(np.abs(bp))) * 255
cv2.imwrite('bp.jpg', bp_save)



plt.figure(figsize=(20,10))
plt.imshow(np.abs(phase), cmap='gray')
plt.figure(figsize=(20,10))
plt.imshow(np.real(phase), cmap='gray')
plt.figure(figsize=(20,10))
plt.imshow(Phase_unwrapping(phase,500), cmap='gray')


plt.figure(figsize=(20,10))
plt.imshow(np.abs(np.fft.ifft2(phase)), cmap='gray')
plt.figure(figsize=(20,10))
plt.imshow(np.real(np.fft.ifft2(phase)), cmap='gray')
plt.figure(figsize=(20,10))
plt.imshow(Phase_unwrapping(np.fft.ifft2(phase),500), cmap='gray')


def save_fig(name, image):
    x = image
    savefig = (x-np.min(x))/(np.max(x)-np.min(x)) * 255
    cv2.imwrite(name, savefig)
    return savefig




save_fig('phase_abs.png', np.abs(phase))
save_fig('phase_real.png', np.real(phase))
save_fig('phase_angle.png', Phase_unwrapping(phase,500))

save_fig('phase_fft_abs.png', np.abs(np.fft.ifft2(phase)))
save_fig('phase_fft_real.png', np.real(np.fft.ifft2(phase)))
save_fig('phase_fft_angle.png', Phase_unwrapping(np.fft.ifft2(phase),500))




save_fig('phase_9_p.png', plotout_p)




holo = cv2.imread('./samples/TS-20220402201632996.tif')
# holo = rgb2gray(holo)
holo = holo[250:750,250:750,:]
plt.figure(figsize=(20,10))
plt.imshow(holo)
cv2.imwrite('holo_new.png',holo)






# hologram = image_norm('./samples/TS-20220402201632996.tif', image_size)
import copy
import random


bp = cv2.imread('./bp_9500_0.52_2022-04-03 14_22_55.bmp')
bp = rgb2gray(-bp)
bp = bp[250:750,250:750]
# bp = cv2.cvtColor(-bp, cv2.COLOR_BGR2GRAY)
# bp = bp.astype(np.float64)
# # bp = (bp-np.min(bp))/(np.max(bp)-np.min(bp)) - 0.5
# bp = bp-128
plt.figure(figsize=(20,10))
plt.imshow(bp)


phase = cv2.imread('./rec_phase_gan_9500_0.52_5_0_2022-04-03 14_22_55.bmp')
phase = rgb2gray(-phase)
phase = phase[250:750,250:750]
plt.figure(figsize=(20,10))
plt.imshow(phase)

amp = cv2.imread('./rec_gan_9500_0.52_5_0_2022-04-03 14_22_55.bmp')
amp = rgb2gray(-amp)
amp = amp[250:750,250:750]
plt.figure(figsize=(20,10))
plt.imshow(amp)
cv2.imwrite('amp_new.png',amp)



plt.figure(figsize=(20,10))
plt.imshow(bp+amp)


th = 150
src = bp
bp_new = copy.deepcopy(src)
bp_new[np.where(src> th)] = 255
bp_new[np.where(src<=th)] = 0
plt.figure(figsize=(20,10))
plt.imshow(bp_new)


th = 130
src = amp
amp_new = copy.deepcopy(src)
amp_new[np.where(src> th)] = 255
amp_new[np.where(src<=th)] = 0
plt.figure(figsize=(20,10))
plt.imshow(amp_new)



# th2 = 130
# phase_new = copy.deepcopy(phase)
# phase_new[np.where(phase>th2)] = 255
# phase_new[np.where(phase<=th2)] = 0
# plt.figure(figsize=(20,10))
# plt.imshow(phase_new)



bp_new = cv2.medianBlur(bp_new.astype(np.uint8), 3)
amp_new = cv2.medianBlur(amp_new.astype(np.uint8), 3)

# bp_new = cv2.dilate(bp_new, np.ones((3,3),np.uint8) ,iterations = 1)
# amp_new = cv2.dilate(amp_new, np.ones((3,3),np.uint8) ,iterations = 1)

plt.figure(figsize=(20,10))
plt.imshow(bp_new)
plt.figure(figsize=(20,10))
plt.imshow(amp_new)

mask = amp_new & bp_new
plt.figure(figsize=(20,10))
plt.imshow(mask)



def findConnectedComponents(num_objects,labels):
    Components = {'labels':[], 'pixels':[]}
    for i in range(num_objects):
        Components['labels'].append(i)
        Components['pixels'].append(np.sum(labels == i))
    return Components

def findMaxComponent(ComponentList):
    pixels = ComponentList['pixels']
    ComponentList_list = sorted(pixels,reverse=True)
    index = pixels.index(ComponentList_list[1])
    return index

num_objects,labels = cv2.connectedComponents(mask)
# true_label = labels[X,Y]
ComponentList = findConnectedComponents(num_objects,labels)
 
for i in range(len(ComponentList['labels'])):
    if ComponentList['pixels'][i] < 5:
        labels[labels==ComponentList['labels'][i]] = 0

 # labels_dendrite = ComponentList['labels'][findMaxComponent(ComponentList)]
plt.figure(figsize=(20,10))
plt.imshow(labels)





mask = amp_new+bp_new
mask[labels> 0]=255
mask[labels==0]=0

phase_new = copy.deepcopy(phase)
phase_new[np.where(mask> 0)] = phase_new[np.where(mask> 0)]*2
# phase_new[np.where(mask<=0)] = 0
phase_new = (norm(phase_new)*255).astype(np.uint8)

plt.figure(figsize=(20,10))
plt.imshow(phase_new, cmap='gray')
cv2.imwrite('phase_new.png',phase_new)







mask_sum = []

for i in range(150,180):
    # print(z)
    th = i
    bp_new = copy.deepcopy(bp).astype(np.uint8)
    bp_new[np.where(bp> th)] = 255
    bp_new[np.where(bp<=th)] = 0
    
    amp_new = copy.deepcopy(amp).astype(np.uint8)
    amp_new[np.where(amp> th)] = 255
    amp_new[np.where(amp<=th)] = 0

    bp_new = cv2.medianBlur(bp_new.astype(np.uint8), 3)
    amp_new = cv2.medianBlur(amp_new.astype(np.uint8), 3)

    # plt.figure(figsize=(20,10))
    # plt.imshow(bp_new & amp_new)
    
    if random.random() > 0.5:
        mask = bp_new & amp_new
    if random.random() <= 0.5:
        mask = bp_new | amp_new
    
    mask_sum.append(mask)
   
    
mask_total = 0

for i in range(len(mask_sum)):
    # holo_sum[i] = standardization(holo_sum[i])
    # holo_total = holo_total + holo_sum[i]
    
    # bp_sum[i] = standardization(bp_sum[i])
    mask_total = mask_total + mask_sum[i]
    
# plt.figure()
# plt.imshow(holo_total, cmap='gray')
# bp_total[np.where(bp_total>0)] = 255
# bp_total[np.where(bp_total<=0)] = 0  

plt.figure(figsize=(20,10))
plt.imshow(mask_total)
    









K = 2

Z = bp_new.reshape((-1,1))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((bp.shape))
res3 = label.reshape((bp.shape))

#plt.imshow(res2[:,:,0])
plt.figure(figsize=(20,10))
plt.imshow(res2)

res3[res3==0] = 0
plt.figure(figsize=(20,10))
plt.imshow(res3)













from mpl_toolkits.mplot3d import Axes3D


def norm(data):
    return (data-np.std(data))/np.var(data)

test = cv2.imread('phase_new.png', cv2.COLOR_BGR2GRAY)

h,w = test.shape
image_size = h





z = norm(test)
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
xgrid,ygrid = np.meshgrid(np.arange(1,image_size+1), np.arange(1,image_size+1))
# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
#ax.plot_surface(x, y, z, color='b')
ax.view_init(elev=30, azim=60)
ax.plot_surface(xgrid, ygrid, z, cmap='rainbow')
ax.set_zlim(0,5)
plt.show()
# return grad, circle_list
































