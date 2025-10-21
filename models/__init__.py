import os
import torch
import torch.nn as nn
from functools import partial

def get_models(model_type, model_scale, model_root, epoch, device):
    if model_type == 'mae':
        from .mae.models_mae import mae_vit_small_patch16 as vit_small
        model = vit_small().to(device)

        checkpoint = torch.load(os.path.join(model_root, f'checkpoint-{epoch}.pth'), map_location='cpu')
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
    elif model_type == 'dino' or model_type == 'ibot' or model_type == 'mugs':
        from .dino.vision_transformer import vit_small, load_pretrained_weights
        if model_scale == 'vit_small':

            model = vit_small().to(device)
            load_pretrained_weights(
                model, 
                os.path.join(model_root, f'checkpoint{str(epoch).zfill(4)}.pth'), 
                'teacher', 'small', 16
            )
            msg = ''

        else:
            raise NotImplementedError
    
    elif model_type == 'ijepa':
        from .ijepa.deit import deit_base

        if model_scale == 'vit_base':
            model = deit_base().to(device)
            checkpoint = torch.load(os.path.join(model_root, f'ijepa-ep{epoch}.pth.tar'), map_location=torch.device('cpu'))
            pretrained_dict = checkpoint['encoder']
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
            msg = model.load_state_dict(pretrained_dict)

        else:
            raise NotImplementedError
    
    elif model_type == 'mocov3':
        from .mocov3.vit_moco import vit_small

        if model_scale == 'vit_small':
            model = vit_small().to(device)
            state_dict = torch.load(os.path.join(model_root, f'checkpoint_{str(epoch).zfill(4)}.pth.tar'), map_location="cpu")["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.momentum_encoder.')}
            state_dict = {k.replace("module.momentum_encoder.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError

    elif model_type == 'mec':
        if model_scale == 'resnet50':
            from .mec.resnet50 import resnet50
            model = resnet50().to(device)
            state_dict = torch.load(os.path.join(model_root, f'checkpoint_{str(epoch).zfill(4)}.pth.tar'), map_location="cpu")["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.encoder.')}
            state_dict = {k.replace("module.encoder.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError

    elif model_type == 'resa':
        if model_scale == 'resnet50':
            from .mec.resnet50 import resnet50
            model = resnet50().to(device)
            state_dict = torch.load(os.path.join(model_root, f'checkpoint_{str(epoch).zfill(4)}.pth.tar'), map_location="cpu")["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.encoder.')}
            state_dict = {k.replace("module.encoder.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError

    elif model_type == 'simsiam':
        if model_scale == 'resnet50':
            from .mec.resnet50 import resnet50
            model = resnet50().to(device)
            state_dict = torch.load(os.path.join(model_root, f'checkpoint_{str(epoch).zfill(4)}.pth.tar'), map_location="cpu")["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.encoder.')}
            state_dict = {k.replace("module.encoder.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError

    elif model_type == 'densecl':
        if model_scale == 'resnet50':
            from .densecl.resnet50 import resnet50
            model = resnet50().to(device)
            state_dict = torch.load(os.path.join(model_root, f'epoch_{epoch}.pth'), map_location="cpu")["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if k.startswith('backbone.')}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError

    elif model_type == 'swav':
        if model_scale == 'resnet50':
            from .swav.resnet50 import resnet50
            model = resnet50().to(device)
            state_dict = torch.load(os.path.join(model_root, f'ckp-{epoch}.pth'), map_location="cpu")
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError
    
    elif model_type == 'vicreg' or model_type == 'vicregl':
        if model_scale == 'resnet50':
            from .swav.resnet50 import resnet50
            model = resnet50().to(device)
            state_dict = torch.load(os.path.join(model_root, f'model_{epoch}.pth'), map_location="cpu")['model']
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
            state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError

    elif model_type == 'byol':
        from .byol.byol import EncoderwithProjection
        model = EncoderwithProjection().to(device)
        state_dict = torch.load(os.path.join(model_root, f'resnet50_{str(epoch)}.pth.tar'), map_location="cpu")['model']

        state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.target_network.')}
        state_dict = {k.replace("module.target_network.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
    
    elif model_type == 'barlowtwins':
        if model_scale == 'resnet50':
            from .swav.resnet50 import resnet50
            model = resnet50().to(device)
            state_dict = torch.load(os.path.join(model_root, f'checkpoint_{epoch}.pth'), map_location="cpu")['model']
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
            state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            raise NotImplementedError

    elif model_type == 'esvit':
        from .esvit.swin_transformer import SwinTransformer
        if model_scale == 'swint_tiny':
            model = SwinTransformer(
                img_size=[ 224, 224 ],
                in_chans=3,
                num_classes=1000,
                patch_size=4,
                embed_dim=96,
                depths=[ 2, 2, 6, 2 ],
                num_heads=[ 3, 6, 12, 24 ],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0,
                attn_drop_rate=0 ,
                drop_path_rate= 0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                ape=False,
                patch_norm=True,
                use_dense_prediction=False,
            ).to(device)

            state_dict = torch.load(os.path.join(model_root, f'checkpoint{str(epoch).zfill(4)}.pth'), map_location="cpu")['teacher']
           
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    print(msg)
    return model
    
