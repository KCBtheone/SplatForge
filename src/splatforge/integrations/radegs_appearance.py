"""Independent train-only appearance module from VastGaussian Eq.3/Fig.9.

Only rendered RGB and the original training-view index enter the predictor.
The paper leaves initialization unspecified; use a declared unit initial map
without consuming the Gaussian model's RNG stream. No author code or weights.
"""
import torch
from torch import nn
import torch.nn.functional as F

from ..studio.losses import photometric_loss,ssim_loss


class DecoupledAppearance(nn.Module):
    def __init__(self, views):
        super().__init__()
        self.embedding=nn.Embedding(views,64)
        nn.init.normal_(self.embedding.weight,mean=0.,std=.01)
        layers=[nn.Conv2d(67,256,3,padding=1)]
        for channels in (256,128,64,32):
            layers.extend((nn.PixelShuffle(2),nn.Conv2d(channels//4,channels//2,3,padding=1),nn.ReLU()))
        self.upsample=nn.Sequential(*layers)
        self.output=nn.Sequential(nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,3,3,padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.ones_(self.output[-1].bias)
        self.counts=None

    def bind_counters(self,counts):
        if self.counts is not None:
            raise ValueError('Appearance counters already bound')
        self.counts=counts
        def hook(role):
            def backward(gradient):
                counts[role]+=1
                return gradient
            return backward
        self.embedding.weight.register_hook(hook('paper_appearance_embedding_backward'))
        for parameter in list(self.upsample.parameters())+list(self.output.parameters()):
            parameter.register_hook(hook('paper_appearance_network_backward'))

    def forward(self,image,view_index):
        if image.dtype!=torch.float32 or image.ndim!=3 or image.shape[-1]!=3:
            raise ValueError('Appearance requires the unchanged FP32 HWC render')
        h,w=image.shape[:2]
        if min(h,w)<32 or not 0<=int(view_index)<self.embedding.num_embeddings:
            raise ValueError('Appearance requires a valid original view and image >=32 pixels')
        if self.counts is not None:
            self.counts['paper_appearance_forward']+=1
        # Keep CNN arithmetic FP32, including on CUDA with TF32-capable cuDNN.
        with torch.backends.cudnn.flags(allow_tf32=False):
            low=F.interpolate(image.permute(2,0,1)[None],size=(h//32,w//32),mode='bilinear',align_corners=False)
            code=self.embedding.weight[int(view_index)].reshape(1,64,1,1).expand(1,64,*low.shape[-2:])
            value=self.upsample(torch.cat((low,code),dim=1))
            value=F.interpolate(value,size=(h,w),mode='bilinear',align_corners=False)
            mapping=self.output(value)[0].permute(1,2,0)
        return image*mapping

    def photometric_loss(self,image,target,view_index,ssim_weight):
        if target.dtype!=torch.float32 or target.shape!=image.shape:
            raise ValueError('Appearance supervision must be the unchanged FP32 training RGB')
        adjusted=self(image,view_index)
        if self.counts is not None:
            self.counts['paper_appearance_raw_ssim']+=1
            if image.requires_grad:
                def backward(gradient):
                    self.counts['paper_appearance_render_backward']+=1
                    return gradient
                image.register_hook(backward)
        return (1-ssim_weight)*(adjusted-target).abs().mean()+ssim_weight*ssim_loss(image,target)


def unit_guard(module,image,target,view_index,weight):
    """Actual initial full-image F/true-VJP; reference never updates a model."""
    a=image.detach().clone().requires_grad_()
    b=image.detach().clone().requires_grad_()
    actual=module.photometric_loss(a,target,view_index,weight)
    expected=photometric_loss(b,target,weight)
    # A nontrivial VJP cotangent removes mean reduction from the RGB gradient;
    # this is scoring only, not a scale applied to the actual training loss.
    cotangent=float(image.numel())
    ga=torch.autograd.grad(actual*cotangent,a)[0]
    ge=torch.autograd.grad(expected*cotangent,b)[0]
    errors={}
    for name,x,y in [('loss',actual,expected),('render_true_vjp',ga,ge)]:
        difference=(x-y).abs()
        finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        bad=int((difference>3e-5+3e-4*y.abs()).sum())
        errors[name]=dict(passed=finite and bad==0,max_abs=float(difference.detach().max()),bad=bad)
    return dict(passed=all(v['passed'] for v in errors.values()),errors=errors,
        image_shape=list(image.shape),cotangent=cotangent,model_io_fp32=True,
        scope='Declared unit initial full CNN/map F and RGB true VJP versus original photometric loss. Read-only scoring;trained weights and actual Adam/resume checked separately.')


def recovery_state(model,optimizer):
    """Small CPU-only state for the actual same-process checkpoint boundary."""
    state={'model':{k:v.detach().cpu().clone() for k,v in model.decoupled_appearance.state_dict().items()},'groups':[]}
    for group in optimizer.param_groups:
        if group['name'] not in ('appearance_embedding','appearance_network'):
            continue
        state['groups'].append(dict(name=group['name'],lr=group['lr'],states=[
            {k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v
             for k,v in optimizer.state.get(parameter,{}).items()} for parameter in group['params']]))
    return state


def checkpoint_recovery_state(payload):
    prefix='decoupled_appearance.'
    values=payload['pipeline']['model'];optimizer=payload['optimizer']
    result=dict(model={k[len(prefix):]:v.detach().cpu().clone() for k,v in values.items()
                      if k.startswith(prefix)},groups=[])
    if not result['model'] or not payload['config']['model'].get('paper_decoupled_appearance'):
        raise ValueError('The original checkpoint must contain actual paper appearance state')
    for group in optimizer['param_groups']:
        if group['name'] in ('appearance_embedding','appearance_network'):
            result['groups'].append(dict(name=group['name'],lr=group['lr'],states=[
                {k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v
                 for k,v in optimizer['state'][index].items()} for index in group['params']]))
    if len(result['groups'])!=2:raise ValueError('Missing actual appearance optimizer groups')
    return result


def restore_guard(model,optimizer,expected):
    actual=recovery_state(model,optimizer);bad=0;maximum=0.
    def same(a,b):
        nonlocal bad,maximum
        if isinstance(a,dict) and isinstance(b,dict):
            if a.keys()!=b.keys():bad+=1;return
            for key in a:same(a[key],b[key])
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):bad+=1;return
            for x,y in zip(a,b):same(x,y)
        elif isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            if a.shape!=b.shape or a.dtype!=b.dtype:bad+=1;return
            error=(a-b).abs();maximum=max(maximum,float(error.max()))
            bad+=int((error>3e-5+3e-4*b.abs()).sum())
            if not bool(torch.isfinite(a).all()):bad+=1
        elif a!=b:bad+=1
    same(actual,expected)
    return dict(passed=bad==0,bad=bad,max_abs=maximum,
        groups=[g['name'] for g in actual['groups']],scope='Actual CNN/embedding and original Adam state readback after Trainer checkpoint/setup;original tolerance,no byte/hash comparison')
