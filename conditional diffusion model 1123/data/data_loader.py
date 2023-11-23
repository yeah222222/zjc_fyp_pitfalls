import os
import math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from decimal import Decimal
class data_loader(Dataset):
    
    def __init__(self,data_dir,batch_size,draw_every_class,image_size=(3,256,256)) :
        super(data_loader,self).__init__()
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.draw_every_class=draw_every_class
        self.c=image_size[0] 
        self.h=image_size[1]
        self.w=image_size[2]
        self.device='cuda:0'
        self.dataset=self.create_all_batches()

            
    def load_data(self):
        print('Load data...')
        img_dirs=os.listdir(self.data_dir)
        img_dirs=[os.path.join(self.data_dir,d) for d in img_dirs]
        return img_dirs
    
    def openimg_to_arr(self,fp):
        input_img=Image.open(fp)
        input_img_arr=np.array(input_img).transpose((2,0,1))  
        return input_img_arr
    
        
    def todecimal(self,s):
        return Decimal(s).quantize(Decimal('0.000'))
    
    def cal_deltac(self,fp1,fp2):#filepath1 filepath2    逆时针为sin为正 顺时针sin为负数
        #
        xyz1=np.array([self.todecimal(fp1.split("/")[-1].split(".jpg")[0].split("_")[1]), self.todecimal(fp1.split("/")[-1].split(".jpg")[0].split("_")[2])])
        xyz2=np.array([self.todecimal(fp2.split("/")[-1].split(".jpg")[0].split("_")[1]), self.todecimal(fp2.split("/")[-1].split(".jpg")[0].split("_")[2])])
        Norm = np.linalg.norm(xyz1) * np.linalg.norm(xyz2)
        rho = np.cross(xyz1, xyz2)
        az=math.acos(np.dot(xyz1,xyz2)/Norm)
        # print(az)
        if rho<0:
            sin_az=-math.sin(az)
        else:
            sin_az=math.sin(az)
        cos_az=math.cos(az)
        res=np.array([sin_az,cos_az])
        #print("res",res)
      #  print(np.ndim(res))
        return res
    
    def create_all_batches(self):
        datadirs=self.load_data()
        data_num_dict=dict()
        data_fps=dict()
        for d in datadirs:
            data_num_dict[d]=len(os.listdir(d))
            data_fps[d]=sorted([os.path.join(d,fp) for fp in os.listdir(d) if fp.endswith('.jpg')])
     
        all_input_imgs=np.empty(shape=(self.c,self.h,self.w))
        all_deltacs=np.empty(shape=(0,2))
        all_target_imgs=np.empty(shape=(self.c,self.h,self.w))
        
        #不重复随机抽取
         
        for d,files in data_fps.items():
            # print(len(data_fps[d]))
            # 一次性抽取所有的元素
           # images = np.random.choice(data_fps[d], size=2*self.draw_every_class, replace=False)
            # ffs=sorted(data_fps[d])
            # for i in range(len(ffs)-1):
            #     for j in range(1,len(ffs)):
            #         im1=ffs[i]
            #         im2=ffs[j]
            #         print("input",im1)
            #         print("output",im2)
            #         if im1==im2:
            #             continue
            #         else:
            #             print("input",im1)
            #             print("output",im2)
            #             deltac=self.cal_deltac(im1,im2)
            #             all_input_imgs=np.append(all_input_imgs,self.openimg_to_arr(im1),axis=0)
            #             all_deltacs=np.append(all_deltacs,[deltac],axis=0)
            #             all_target_imgs=np.append(all_target_imgs,self.openimg_to_arr(im2),axis=0)
        #============================================random选取=======================
            for i in range(self.draw_every_class):
                # 从预先抽取的元素中获取 im1 和 im2
                # im1 = images[2*i]
                # im2 = images[2*i + 1]
              #  print("im2",im2)
                #for overfitting
                # im1=data_fps[d][0]
                # im2=data_fps[d][1]
                im1,im2=np.random.choice(data_fps[d],2,replace=False)
                print("input",im1)
                print("output",im2)
                deltac=self.cal_deltac(im1,im2)
                all_input_imgs=np.append(all_input_imgs,self.openimg_to_arr(im1),axis=0)
                all_deltacs=np.append(all_deltacs,[deltac],axis=0)
                all_target_imgs=np.append(all_target_imgs,self.openimg_to_arr(im2),axis=0)
                
       # print("before reshape",all_input_imgs.shape)  #(6, 256, 256) 
       # print("deltac before",all_deltacs.shape)#(1, 2)
        #整理一下数据
        #多余的不要了
        total_num=(math.floor(self.draw_every_class*len(data_num_dict.keys())/self.batch_size))*self.batch_size
        # 生成一个打乱的索引
        indices = torch.randperm(total_num)
        
        all_input_imgs=all_input_imgs[3:3*(total_num+1)]
        all_deltacs=all_deltacs[:total_num]
        all_target_imgs=all_target_imgs[3:3*(total_num+1)]
        
      #  print("all input",all_input_imgs.shape) (3, 256, 256)
      #  print("deltac",all_deltacs.shape)(1, 2)

        all_input_imgs=all_input_imgs[np.newaxis,:,:,:]
        all_input_imgs=all_input_imgs.reshape(-1,self.c,self.h,self.w)
        all_deltacs=all_deltacs[np.newaxis,:]
       # print("alldeltacs",all_deltacs)
        all_deltacs=all_deltacs.reshape(-1,2)
     #   print("after reshape",all_deltacs)
        all_target_imgs=all_target_imgs[np.newaxis,:,:,:]
        all_target_imgs=all_target_imgs.reshape(-1,self.c,self.h,self.w)
        
        #打乱以上所有 防止每一个batch中都是同一物体
        all_input_imgs=all_input_imgs[indices]
        all_deltacs=all_deltacs[indices]
        all_target_imgs=all_target_imgs[indices]
        
        self.input_img_batches=np.expand_dims(all_input_imgs,axis=0).reshape(-1,self.batch_size,self.c,self.h,self.w)
        self.deltac_batches=np.expand_dims(all_deltacs,axis=0).reshape(-1,self.batch_size,2)       
        self.out_target_batches=np.expand_dims(all_target_imgs,axis=0).reshape(-1,self.batch_size,self.c,self.h,self.w)

        return self.input_img_batches,self.deltac_batches,self.out_target_batches
#如果想用pytorch.lightening就变成totalnum,c,h,w形式即可
#然后把下面的getitem的被miu掉的 重新整回来就行
            
    def __getitem__(self,idx):
        # batch=dict()
        # batch['image_target']=self.out_target_batches[idx]
        # batch['image_cond']=self.input_img_batches[idx]
        # batch['T']=self.deltac_batches[idx]
        return self.input_img_batches[idx],self.deltac_batches[idx],self.out_target_batches[idx]

    def __len__(self):
        return len(self.input_img_batches)
        
        
        
        
        