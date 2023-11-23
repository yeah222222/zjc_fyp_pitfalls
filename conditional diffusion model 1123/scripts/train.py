# import sys
# sys.path.append('/mnt/d/junch_data/test_junch')
import torch
import os,sys
sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/data', '../')))
from data.data_loader import data_loader
# sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/model', '../')))
# from model.whole_model import wholemodel
sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/model/ldm/modules/ddpm', '../')))
from model.ldm.modules.ddpm.diffusion_simplist import DDPM
from model.ldm.modules.ddpm.lr_scheduler import LambdaLinearScheduler
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import logging
from PIL import Image

from collections import OrderedDict


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def load_ckpt(ckpt,device):
    whole_model=DDPM().to(device)
 
# diffusion.load_state_dict(torch.load('/mnt/d/junch_data/diffusion_231019/DDPM_TEST/ddpm_zjc/logs/checkpoints_zjc256/yeah-run-iteration-1300-model.pth'))
    checkpoint=torch.load(ckpt)
    print("ckpt",checkpoint.keys())
    model_state_dict = OrderedDict((k.replace('model.', ''), v) for k, v in checkpoint.items() if k.startswith('model.'))
    
    # Modify the weights
    print("ckpt",model_state_dict.keys())
    for k in list(model_state_dict.keys()):
        # Check if the weight is three-dimensional
        if len(model_state_dict[k].shape) == 3:
            # Add a dimension to the end
            model_state_dict[k] = model_state_dict[k].unsqueeze(3)
        if len(model_state_dict[k].shape) == 4 and model_state_dict[k].shape[1] == 4:
            # Initialize a new weight with 8 input channels
            new_weight = torch.zeros(
                model_state_dict[k].shape[0],
                8,
                model_state_dict[k].shape[2],
                model_state_dict[k].shape[3],
                device=device,
            )
            # Copy the original weight to the first four channels
            new_weight[:, :4, :, :] = model_state_dict[k]
            # Replace the original weight with the new weight
            model_state_dict[k] = new_weight

    # whole_model.load_state_dict(checkpoint['state_dict'], strict=False)
    whole_model.model.load_state_dict(model_state_dict,strict=False)
    #print("ckpt:",whole_model)
    return whole_model

def train():
    #load logger
    logger = get_logger('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/logs/formal_finetune_diffusion1260_modifie_ccprojection_withmask2_withconcatenate_likezero123.log',verbosity=1, name="my_logger")
    assert(logger is not None)
    logger.info('start training!')

    #用自己的dataloader加载数据
    train_dataloader=data_loader(data_dir='/mnt/d/junch_data/diffusion_231019/MODEL_zjc/data/all_car_30_jpg_train/',batch_size=32,draw_every_class=64)
    val_dataloader=data_loader(data_dir='/mnt/d/junch_data/diffusion_231019/MODEL_zjc/data/all_car_30_jpg_val/',batch_size=32,draw_every_class=16)
    
    #其他参数
    device=torch.device("cuda:0")
    total_train_step = 0
    total_test_step=0
    acc_train_loss=0
    epoch = 1000
    #ckpt model loading
    ifckpt=True
    ckpt='/mnt/d/junch_data/diffusion_231019/DDPM_TEST/ddpm_zjc/logs/checkpoints_zjc256/yeah-run-iteration-1260-model.pth'
    if not ifckpt:
        model=DDPM().to(device)
    else:
        model=load_ckpt(ckpt,device)
        model.train()
       
    lr=1.0e-04
    #optimizer=torch.optim.AdamW(params=model.parameters(),lr=2e-4)   
    optimizer = torch.optim.AdamW([{"params": model.model.parameters(), "lr": lr},
                                   {"params":model.cond_stage_model.parameters(),"lr": lr},
                                {"params": model.cc_projection.parameters(), "lr": 10. * lr}], lr=lr)
    scheduler=LambdaLinearScheduler(warm_up_steps=[ 100 ],f_min=[ 1. ],f_max=[ 1. ],f_start=[ 1.e-6 ],cycle_lengths=[ 10000000000000 ])
    scheduler = LambdaLR(optimizer, lr_lambda=scheduler.schedule)
    
    #save path
    save_model_every_epoch=5
    save_path='/mnt/d/junch_data/diffusion_231019/MODEL_zjc/save_ckpt/formal_finetune_diffusion1260_modifie_ccprojection_withmask2_withconcatenate_likezero123/'
    
    for iteration in range(1, epoch + 1):
        logger.info("-----第 {} 轮训练开始-----".format(iteration))
        for x_cond, x_deltac,x_target in train_dataloader:

            x_cond=(torch.tensor(x_cond,dtype=torch.float32,device=device)/255.*2)-1 #dtype -float format 转成float类型
            deltac=torch.tensor(x_deltac,dtype=torch.float32,device=device)
            x_target=(torch.tensor(x_target,dtype=torch.float32,device=device)/255.*2)-1
 
            loss = model(x_target=x_target,x_cond=x_cond,deltac=deltac)
            acc_train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for i, param_group in enumerate(optimizer.param_groups):
                logger.info("Learning rate of group %d: %f" % (i, param_group['lr']))
           # logger.info("第%d个step的学习率:%f" % (iteration, optimizer.param_groups[0]['lr']))
            logger.info("训练次数: {}Loss: {}".format(total_train_step,loss.item()))
            scheduler.step()
            total_train_step = total_train_step + 1
          
            if model.use_ema:
                model.each_train_batch_end()
        #val
        if iteration%save_model_every_epoch==0:   
            with torch.no_grad():  # 没有梯度了
                model.eval()
                with model.ema_scope():
                    for x_test, x_deltac,y_test in val_dataloader:  
                        x_cond=(torch.tensor(x_test,dtype=torch.float32,device=device)/255.*2)-1 #dtype -float format 转成float类型
                        deltac=torch.tensor(x_deltac,dtype=torch.float32,device=device)
                        
                        x_target=(torch.tensor(y_test,dtype=torch.float32,device=device)/255.*2)-1

                        loss = model(x_target=x_target,x_cond=x_cond,deltac=deltac)
                                    
                        logger.info("Validation loss in the validation set is {}".format(loss.item()))
                        total_test_step+=1
                
        
        
        #save
        if iteration% save_model_every_epoch==0:
            state={'epoch':iteration,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(state,save_path+str(iteration)+'.pt')
            logger.info("save model epoch :{}".format(iteration))        
                              
if __name__=='__main__':
    train()

