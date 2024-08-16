import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import logging
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import torch 
import logging 
import os
import shutil
import time
from skimage.metrics import structural_similarity
from thop import profile
from thop import clever_format
import shutil

class Metric():
    def __init__(self,GT,preHSI) -> None:
        self.eps = 2.2204e-16
        assert GT.shape == preHSI.shape
        self.GT = GT.detach().cpu().numpy()
        self.preHSI = preHSI.detach().cpu().numpy()
        self.PSNR,self.RMSE,self.SAM,self.ERGAS,self.SSIM = [],[],[],[],[]
        if len(GT.shape) == 4:
            for i in range(len(self.GT)):
                self.PSNR.append(self.calc_psnr(self.GT[i],self.preHSI[i]))
                self.RMSE.append(self.calc_rmse(self.GT[i],self.preHSI[i]))
                self.SAM.append(self.calc_sam(self.GT[i],self.preHSI[i]))
                self.ERGAS.append(self.calc_ergas(self.GT[i],self.preHSI[i]))
                self.SSIM.append(self.calc_ssim(self.GT[i],self.preHSI[i]))
            self.PSNR = np.array(self.PSNR).mean()
            self.RMSE = np.array(self.RMSE).mean()
            self.SAM = np.array(self.SAM).mean()
            self.ERGAS = np.array(self.ERGAS).mean()
            self.SSIM = np.array(self.SSIM).mean()

        if len(GT.shape) == 3:
            self.PSNR = self.calc_psnr(self.GT,self.preHSI)
            self.RMSE = self.calc_rmse(self.GT,self.preHSI)
            self.SAM = self.calc_sam(self.GT,self.preHSI)
            self.ERGAS = self.calc_ergas(self.GT,self.preHSI)
            self.SSIM = self.calc_ssim(self.GT,self.preHSI)

    @torch.no_grad()
    def calc_ergas(self, GT_image, fuse_image):
        GT_image = np.squeeze(GT_image)
        fuse_image = np.squeeze(fuse_image)
        GT_image = GT_image.reshape(GT_image.shape[0], -1)
        fuse_image = fuse_image.reshape(fuse_image.shape[0], -1)
        rmse = np.mean((GT_image-fuse_image)**2, axis=1)
        rmse = rmse**0.5
        mean = np.mean(GT_image, axis=1)
        ergas = np.mean((rmse/mean)**2)
        ergas = 100/4*ergas**0.5
        return ergas

    @torch.no_grad()
    def calc_psnr(self, GT_image, fuse_image):
        mse = np.mean((GT_image-fuse_image)**2)
        img_max = np.max(GT_image)
        psnr = 10*np.log10(img_max**2/mse)
        return psnr

    @torch.no_grad()
    def calc_rmse(self,GT_image, fuse_image):
        rmse = np.sqrt(np.mean((GT_image-fuse_image)**2))
        return rmse

    @torch.no_grad()
    def calc_sam(self, im1, im2):
        assert im1.shape == im2.shape
        H, W, C = im1.shape
        im1 = np.reshape(im1, (H * W, C))
        im2 = np.reshape(im2, (H * W, C))
        core = np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole + self.eps) / (deno + self.eps)).clip(-1, 1)))
        return np.mean(sam)
    
    @torch.no_grad()
    def calc_ssim(self, GT_image, fuse_image):
        ssim = structural_similarity(GT_image,fuse_image,data_range=1.)
        return ssim

def get_model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    return all_size

def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def beijing_time():
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    beijing_now = utc_now.astimezone(SHA_TZ)
    fmt = '%Y-%m-%d,%H:%M:%S'
    now_fmt=beijing_now.strftime(fmt)
    return  now_fmt

def set_seed(seed=9999):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_logger(model_name, logger_dir, log_out):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    if log_out == 1:
        ## make out.log
        log_file = f"{logger_dir}/out.log" 
        if not os.path.exists(log_file):
            os.mknod(log_file)  
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        ## copy model file 
        model_file_path = f'./models/{model_name}.py'
        shutil.copy(model_file_path,f"{logger_dir}/{model_name}.py" )
        
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO) 
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger

def test_speed(model,device,band_nums=31,scale=4):
    model.eval()
    if scale == 2:
        HSI = torch.randn((1,band_nums,32,32)).to(device)
        RGB = torch.randn((1,3,64,64)).to(device)
    if scale == 4:
        HSI = torch.randn((1,band_nums,16,16)).to(device)
        RGB = torch.randn((1,3,64,64)).to(device)
    if scale == 8:
        HSI = torch.randn((1,band_nums,16,16)).to(device)
        RGB = torch.randn((1,3,128,128)).to(device)
    # flops, params = profile(model, inputs=(HSI,RGB,))
    # flops, params = clever_format([flops, params], "%.6f")
    from fvcore.nn import FlopCountAnalysis, parameter_count,parameter_count_table
    flops = FlopCountAnalysis(model, (HSI,RGB)).total() /1e9
    params = parameter_count(model)[''] / 1e6
    # print(parameter_count_table(model))
    start_time = time.time()
    with torch.no_grad():
            model(HSI,RGB)
    end_time = time.time()
    inference_time = end_time - start_time
    return inference_time,flops,params


