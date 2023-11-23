import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy
import os,sys
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from ddpm.utils import extract,make_beta_schedule,default,instantiate_from_config
import pytorch_lightning as pl
# from ldm.modules.ema import LitEma
sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/model/ldm/modules/ddpm', '../')))
from model.ldm.modules.diffusionmodules.unet import UNetModel
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/model/ldm/modules/distributions', '../')))
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
#from torch.optim.lr_scheduler import LambdaLR
# sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/data', '../')))
# from data.data_loader import data_loader
# from torch.utils.data import DataLoader
# from model.ldm.modules.ddpm.lr_scheduler import LambdaLinearScheduler
sys.path.append(os.path.abspath(os.path.join('/mnt/d/junch_data/diffusion_231019/MODEL_zjc/model/ldm/modules/conditions', '../')))
from model.ldm.modules.conditions.imageEncoder import FrozenCLIPImageEmbedder
from model.ldm.modules.ddpm.ema import LitEma

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    t=t.to(a.device)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDPM(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    def __init__(
        self,
        model=UNetModel(in_channels=8, #之前concatenate 的时候是8
                        model_channels=128,
                        out_channels=4,
                        num_res_blocks=2,
                        attention_resolutions=[4,2,1],
                        use_spatial_transformer=True,
                        channel_mult=[1,2,4,4],
                        num_heads=8,
                        
                        context_dim=768),
        img_size=(256,256),
        img_channels=4,
        original_elbo_weight=1.,
        loss_type="l2",
        first_stage_config='/mnt/d/junch_data/diffusion_231019/MODEL_zjc/allconfigs/encoder_cfg/kl-f8.yaml',
        encoder_ckpt='/mnt/d/junch_data/diffusion_231019/MODEL_zjc/configs/first_stage/checkpoints/epoch=000029.ckpt',
        use_ema=True,
        l_simple_weight=1.,
        v_posterior=0.,
        parameterization="eps"
    ):
        super().__init__()

        self.model = model
        self.img_channels=img_channels
        self.original_elbo_weight=original_elbo_weight
        self.encoder_config_path=first_stage_config
        self.encoder_ckpt=encoder_ckpt
        self.device='cuda:0'
        self.l_simple_weight=l_simple_weight
        self.register_buffer('scale_factor', torch.tensor(1.))
        self.img_size = img_size
       
        self.loss_type = loss_type
        self.cond_stage_model=FrozenCLIPImageEmbedder()
        self.v_posterior=v_posterior
        self.use_ema = use_ema
        self.parameterization=parameterization
        if self.use_ema:  #使用ema 平滑diffusion model参数 防止训练出现过大的震荡
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.veryfirst=True
        self.register_schedule()
        
        self.instantiate_first_stage(first_stage_config)

            
        # self.cc_projection = nn.Linear(770, 768)
        # nn.init.eye_(list(self.cc_projection.parameters())[0][:768, :768])
        # nn.init.zeros_(list(self.cc_projection.parameters())[1])
        # self.cc_projection.requires_grad_(True)
        self.cc_projection = nn.Linear(770, 768)
        nn.init.eye_(list(self.cc_projection.parameters())[0][:768, :768])
        nn.init.zeros_(list(self.cc_projection.parameters())[1])
        self.cc_projection.requires_grad_(True)
        
    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2):

        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1]) #把alphas_cumprod_prev 向右移动了一个 并且在最前面加了个1

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
   

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        
        self.register_buffer('reciprocal_sqrt_alphas', to_torch(np.sqrt(1. / alphas)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))
       
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))          #后验分布 的方差 反向传播过程中可用
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))

        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
        
    #加载encode的代码
    def instantiate_first_stage(self, config):
        config = OmegaConf.load(self.encoder_config_path)
        pl_sd = torch.load(self.encoder_ckpt)
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
            
    @contextmanager #with ema_scope() 后面全是使用ema权重 如果出来了 在之后使用unet原始weights
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model) #copy权重到当前unet模型中 这样之后unet继续跑就可以用这个了
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")   
                     
    @torch.no_grad()
    def encode_first_stage(self, x):

        return self.first_stage_model.encode(x)    
    @torch.no_grad()
    def predict_noise(self,x,t,c):
        return self.model(x,t,c)
    @torch.no_grad()
    def remove_noise(self,x_target_with_noise,t, x_cond,  deltac):
        x_concatentate,cross_attn=self.getallconditions(x_cond=x_cond,deltac=deltac)
        # per_xt_conca=torch.cat([x_target_with_noise]+x_concatentate,dim=1)
        per_xt_conca=torch.cat([x_target_with_noise],dim=1)
        context=torch.cat(cross_attn, 1)
        return (
            (x_target_with_noise - extract(self.remove_noise_coeff, t, x_target_with_noise.shape) * self.model(per_xt_conca, t, context)) *
            extract(self.reciprocal_sqrt_alphas, t, x_target_with_noise.shape)
        )
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    # @torch.no_grad()
    # def sample(self, batch_size, device, x_cond,  deltac):
    #     if x_cond is not None and batch_size != len(x_cond):
    #         raise ValueError("sample batch size different from length of given y")

    #     x = torch.randn(batch_size, self.img_channels, *(32,32), device=device)
        
    #     #denoising
    #     for t in range(self.num_timesteps - 1, -1, -1):
    #         t_batch = torch.tensor([t], device=device).repeat(batch_size)
    #         x = self.remove_noise(x, t_batch, x_cond,  deltac)

    #         if t > 0:
    #             x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
    #     return x

    @torch.no_grad()
    def recon_x(self,batch_size,device,x_cond,  deltac):
        print("=================begin reconstruction=====================")
        x=self.sample(batch_size,device,x_cond,deltac)
        x=self.first_stage_model.decode(x)
        #变成
        return x,x.shape

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

    
        # if isinstance(self.first_stage_model, VQModelInterface):
        #     return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        # else:
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)
        
    # def perturb_x(self, x, t, noise):
    #     return (
    #         extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
    #         extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
    #     )   
        
    def get_first_stage_encoding(self, encoder_posterior):
       # print("type",type(encoder_posterior))
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
          #  print("jinru instance")
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    def get_learned_conditioning(self, c):
       
        c = self.cond_stage_model(c)
        #print("this is from clip embedding",c.shape)#[1,768]
        return c

            
    @torch.no_grad()
    def getallconditions(self,x_cond,deltac):
        #=========x_cond=========
        x_concatenate=[self.encode_first_stage(x_cond).mode().detach()]
        
        
       
        #=================================================================================================原来==========
        # clip_emb = self.get_learned_conditioning(x_cond).detach()
        # cross_attn = [self.cc_projection(torch.cat([clip_emb[:,None,:], deltac[:, None, :]], dim=-1))]
        #===============================================================================================后来=====================
        cross_attn = [self.cc_projection(deltac[:, None, :])]
        return x_concatenate,cross_attn
        
    def get_losses(self, x_target,x_cond, deltac, t,uncond=0.05,noise=None):#x target
        #----------xtarget==================
        
        encoder_posterior=self.encode_first_stage(x_target)
        x_target = self.get_first_stage_encoding(encoder_posterior).detach()
        print("x_target",x_target.shape)
        noise = default(noise, lambda: torch.randn_like(x_target))     
        perturbed_x_target = self.q_sample(x_target, t, noise)   
        #=========x_cond=========
        #x_concatenate=[self.encode_first_stage(x_cond).mode().detach()]
        
        
        #classifier-free guidance
        random = torch.rand(x_target.size(0), device=x_target.device) # random 张量，其大小与输入 x 的第一个维度（通常是批量大小）相同，并且元素值在0到1之间随机分布
        prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")  #whose size is the same as random tensor bool type 
        input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1") #input mask 中的每个值 必须在uncond和3uncond之间 为0.0 else 为1.0
        null_prompt = self.get_learned_conditioning([""])
        with torch.enable_grad():
            print("jinrujinrujinru")
            clip_emb = self.get_learned_conditioning(x_cond).detach()[:,None,:]
            null_prompt = self.get_learned_conditioning([""]).detach()
            print("clip",clip_emb.shape)
            cross_attn = [self.cc_projection(torch.cat([torch.where(prompt_mask, null_prompt, clip_emb), deltac[:, None, :]], dim=-1))] 
            #================================================================================================================================原来设计==============
            # cross_attn = [self.cc_projection(torch.cat([clip_emb[:,None,:], deltac[:, None, :]], dim=-1))] 
                #================================================================================================================================现在设计==============
            #cross_attn = [self.cc_projection(deltac[:, None, :])] 
            print("crossattn============================",cross_attn[0].shape)
            #cross_attn = [self.cc_projection(torch.cat([clip_emb[:,None,:], deltac[:, None, :]], dim=-1))]
        x_concatenate=[input_mask * self.encode_first_stage((x_cond.to(x_cond.device))).mode().detach()]
        #if no concatenate========================zjc========================================
        #x_concatenate=[self.encode_first_stage((x_cond.to(x_cond.device))).mode().detach()]
        #================================开始封装============================================
        per_xt_conca=torch.cat([perturbed_x_target] + x_concatenate, dim=1)
       # per_xt_conca=torch.cat([perturbed_x_target], dim=1)
        context=torch.cat(cross_attn, 1)
        
        estimated_noise = self.model(per_xt_conca, t, context)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            # loss = F.mse_loss(estimated_noise, noise,, reduction='none')
            loss = torch.nn.functional.mse_loss(estimated_noise, noise, reduction='none')
            loss = loss.mean(dim=[1, 2, 3])

            # log_prefix = 'train' if self.training else 'val'

            # loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
            loss_simple = loss.mean() *self.l_simple_weight#self.l_simple_weight

            loss_vlb = (self.lvlb_weights[t] * loss).mean()
            #loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

            loss = loss_simple + self.original_elbo_weight * loss_vlb #loss_vlb：这是 VAE 中的变分下界（Variational Lower Bound）损失部分 

           # loss_dict.update({f'{log_prefix}/loss': loss}) #所以ddpm里面的loss分为三个部分 loss_simple(l2) , loss_simple加权后的简单损失, loss_vlb变分下界的损失
        return loss

    def forward(self, x_target,x_cond, deltac):
        b, c, h, w = x_target.shape
        device = x_target.device
        # print(x_target.shape)
        print("h:",h)
        # if h != self.img_size[0]:
        #     raise ValueError("image height does not match diffusion parameters")
        # if w != self.img_size[0]:
        #     raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        with torch.no_grad():
            if self.veryfirst==True:
                print("=============================this is first very very first==========================")
                encoder_posterior = self.encode_first_stage(x_target)
                z = self.get_first_stage_encoding(encoder_posterior).detach()
                # del self.scale_factor
                # self.register_buffer('scale_factor', 1. / z.flatten().std())
                self.scale_factor.data = 1. / z.flatten().std()
                print("after changiing",self.scale_factor)
                self.veryfirst=False
        return self.get_losses(x_target,x_cond, deltac, t)
    
    def each_train_batch_end(self):
        print("jinru ema")
        if self.use_ema:
            self.model_ema(self.model)
            
    # @torch.no_grad()
    # def predict_noise(self,x,t):
    #   #  print("in the model t",t)
    #     e_t=self.model(x=x,timesteps=t)
    #     return e_t