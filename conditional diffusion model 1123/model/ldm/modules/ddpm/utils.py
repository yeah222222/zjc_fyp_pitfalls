import torch
import importlib

from inspect import isfunction
def extract(a, t, x_shape):
    b, *_ = t.shape
    a=a.to(t.device)    
    out = a.gather(-1, t)
    finalout=out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return finalout



def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    return betas.numpy()

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    # if not "target" in config:
    #     if config == '__is_first_stage__':
    #         return None
    #     elif config == "__is_unconditional__":
    #         return None
    #     raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))