import torch
import torch.nn as nn
import math
import numpy as np
from torchvision.utils import save_image as svim
import cv2

class DCT_2D(nn.Module):
    def __init__(self):
        super(DCT_2D, self).__init__()

    def dct_2d(self, x, norm=None):
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def _rfft(self, x, signal_ndim=1, onesided=True):
        odd_shape1 = (x.shape[1] % 2 != 0)
        x = torch.fft.rfft(x)
        x = torch.cat([x.real.unsqueeze(dim=2), x.imag.unsqueeze(dim=2)], dim=2)
        if onesided == False:
            _x = x[:, 1:, :].flip(dims=[1]).clone() if odd_shape1 else x[:, 1:-1, :].flip(dims=[1]).clone()
            _x[:,:,1] = -1 * _x[:,:,1]
            x = torch.cat([x, _x], dim=1)
        return x

    def dct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)
        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        Vc = self._rfft(v, 1, onesided=False)
        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def forward(self, x):
        x = self.dct_2d(x)

        return x

class IDCT_2D(nn.Module):
    def __init__(self):
        super(IDCT_2D, self).__init__()

    def idct_2d(self, X, norm=None):
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)
    
    def _irfft(self, x, signal_ndim=1, onesided=True):
        if onesided == False:
            res_shape1 = x.shape[1]
            x = x[:,:(x.shape[1] // 2 + 1),:]
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x, n=res_shape1)
        else:
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x)
        return x
    
    def idct(self, X, norm=None):
        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = self._irfft(V, 1, onesided=False)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)
    
    def forward(self, x):
        x = self.idct_2d(x)
        return x

#이미지 불러오고(imgs)
imgs = cv2.imread('./IR/010001.jpg')
imgs = np.array(imgs)
ddct = DCT_2D() 
didct = IDCT_2D()
# Forward
svim(imgs, './image_check1.jpeg') #원본
X = ddct(imgs) 


svim(X ,'./image_check2.jpeg') # DCT 이미지
# print(type(X))

_,_, h, w = X.shape
mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
diagonal = w-(int(w//20))
hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
hf_mask = hf_mask.unsqueeze(0).expand(X.size())
X = X * hf_mask
svim(X ,'./image_check33.jpeg') #DCT 강조이미지
new=torch.nn.ReLU(inplace=False)
Y = didct(X)

svim(Y ,'./image_check4.jpeg') 
dct_imgs=imgs+Y
svim(dct_imgs ,'./image_check5.jpeg') #최종 이미지