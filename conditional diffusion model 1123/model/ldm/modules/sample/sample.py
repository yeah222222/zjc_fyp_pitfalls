import torch
import time
import sys,os
from PIL import Image
import numpy as np
import math

from einops import rearrange
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/data', '../')))
from data.data_loader import data_loader
sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/model/ldm/modules/ddpm', '../')))
from model.ldm.modules.ddpm.diffusion_simplist import DDPM
sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/model/ldm/modules/sample', '../')))
from model.ldm.modules.sample.util import load_and_preprocess,create_carvekit_interface
from model.ldm.modules.sample.ddim import DDIMSampler

def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        # input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    #print('new input_im:', lo(input_im))

    return input_im

@torch.no_grad()
def sample_model(input_im, model, sampler, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, T):

  #clip embedding + T
    with model.ema_scope():
        
        # T = torch.tensor([math.radians(x), math.sin(
        #     math.radians(y)), math.cos(math.radians(y)), z])
        print("T",T.shape)
        #T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
        T = T[:, None, :].repeat(n_samples, 1, 1).to('cuda:0')
        #==============================================原来=====================================
        c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
        c = torch.cat([c, T], dim=-1)
        c = model.cc_projection(c)
        #============================================现在========================================

        #c = model.cc_projection(T)
        cond = {}
        cond['c_crossattn'] = [c]
        cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                            .repeat(n_samples, 1, 1, 1)]
        if scale != 1.0:
            uc = {}
            uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
            uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
        else:
            uc = None

        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=None)
        print(samples_ddim.shape)
        # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def sample(models,input_im,device,T,h=256,w=256,scale=3.,n_samples=4, ddim_steps=50, ddim_eta=1.0):
    
    
    #input_im=preprocess_image(models=models,input_im=input_im,preprocess=False)
    
    # input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    # input_im=torch.tensor()
    input_im = input_im * 2 - 1
#    input_im = transforms.functional.resize(input_im, [h, w])
    print("input  immmmm",input_im.shape)
    sampler = DDIMSampler(models['turncam'])
    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    # used_x = x  # NOTE: Set this way for consistency.
    
    x_samples_ddim = sample_model(input_im, models['turncam'], sampler,h, w,
                                        ddim_steps, n_samples, scale, ddim_eta,T)

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * ((rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')+1)/2)
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    # description = None

    # if 'angles' in return_what:
    #     return (x, y, z, description, new_fig, show_in_im2, output_ims)
    # else:
    #     return (description, new_fig, show_in_im2, output_ims)
    return output_ims


def sample_images():
    
    
    model_path='/mnt/d/junch_data/diffusion_231019/MODEL_zjc/save_ckpt/formal_finetune_diffusion1260_modifie_ccprojection_withmask2_withconcatenate_likezero123/20.pt'
    device=torch.device('cuda:0')
    batchsize=1
    draweveryclass=1
    save_dir='/mnt/d/junch_data/diffusion_231019/MODEL_zjc/save_png'
    
    whole_model=DDPM().to(device)


    checkpoint=torch.load(model_path)

    whole_model.load_state_dict(checkpoint['state_dict'], strict=False)
    

    whole_model.eval()
    
    test_dataloader=data_loader(data_dir='/mnt/d/junch_data/diffusion_231019/MODEL_zjc/data/all_car_30_jpg_train/',batch_size=batchsize,draw_every_class=draweveryclass)
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = whole_model
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    bs=0
    for x_cond,deltac,x_target in test_dataloader:
        x_cond=torch.tensor(x_cond,dtype=torch.float32,device=device)/255. #dtype -float format 转成float类型
        deltac=torch.tensor(deltac,dtype=torch.float32,device=device)
        x_target=torch.tensor(x_target,dtype=torch.float32,device=device)
        print("x_target",x_target.shape)
        print("========================begin sampling================================")
        
        # deltac=deltac[
        # x_cond=x_cond[i]
       # print("xcond",x_cond[0].shape)
       #===========================================不用process 要加这句话 因为要把c,h,w变成b,c,h,w
   
        outputims=sample(models=models,input_im=x_cond[0].unsqueeze(0),device=device,T=deltac)
        for i in range(len(outputims)):
            output_path=os.path.join(save_dir,"bs"+str(bs)+"_"+str(i)+"_generate.png")
            outputims[i].save(output_path)
                
        output_path_target=os.path.join(save_dir,"bs"+str(bs)+"_"+str(i)+"_target.png")
     
        x_t=x_target[0].detach().cpu().numpy().squeeze() 
        x_t=np.transpose(x_t,(1,2,0))
     
        x_t = Image.fromarray(x_t.astype('uint8')).convert('RGB')
        x_t.save(output_path_target)
        bs+=1
if __name__=='__main__':
    sample_images()