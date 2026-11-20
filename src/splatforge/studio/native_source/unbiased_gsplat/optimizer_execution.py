"""Optional standard Adam execution across the original independent groups.

SPDX-License-Identifier: Apache-2.0
Group names, learning rates, state and population-replacement semantics stay owned
by the existing torch.optim.Adam instance. No primitive or parameter-count cap.
"""
import math
import torch

GROUP_KERNEL_CALLS=0
ORIGINAL_INITIALIZATION_CALLS=0
GROUP_NATIVE_OVERRIDE=None


@torch.no_grad()
def adam_step(optimizer, *, fused_groups=False):
    global GROUP_KERNEL_CALLS, ORIGINAL_INITIALIZATION_CALLS
    if not fused_groups:
        return optimizer.step()
    if getattr(optimizer,'_optimizer_step_pre_hooks',{}) or getattr(optimizer,'_optimizer_step_post_hooks',{}):
        raise ValueError('Joint-group execution does not bypass optimizer step hooks')
    entries=[];policy=None;initialization=False
    for group in optimizer.param_groups:
        if any(group.get(key,False) for key in ('amsgrad','maximize','capturable','differentiable','fused')) or group.get('weight_decay',0)!=0:
            raise ValueError('Joint-group execution requires the original real Adam policy')
        current=(group['betas'],group['eps'])
        if policy is not None and current!=policy:
            raise ValueError('Joint-group execution requires common original beta/epsilon policy')
        policy=current
        rate=group['lr']
        if type(rate) not in (int,float) or not math.isfinite(rate) or rate<0:
            raise ValueError('Original Adam learning rates must be finite scalar values')
        for parameter in group['params']:
            gradient=parameter.grad
            if gradient is None:continue
            if parameter.dtype!=torch.float32 or gradient.dtype!=parameter.dtype or gradient.is_sparse:
                raise ValueError('Joint-group execution requires dense original float32 derivatives')
            state=optimizer.state.get(parameter,{})
            initialization=initialization or not state
            if state:
                if any(key not in state for key in ('step','exp_avg','exp_avg_sq')):
                    raise ValueError('Original Adam state is incomplete')
                if state['step'].device.type!='cpu' or state['step'].numel()!=1:
                    raise ValueError('Joint-group execution requires the original noncapturable Adam counter')
                for key in ('exp_avg','exp_avg_sq'):
                    if state[key].shape!=parameter.shape or state[key].dtype!=parameter.dtype or state[key].device!=parameter.device:
                        raise ValueError('Original Adam moments do not match their parameter')
            entries.append((parameter,gradient,state,rate))
    if not entries:return None
    if initialization:
        # Use the unchanged optimizer once to initialize its public state schema.
        ORIGINAL_INITIALIZATION_CALLS+=1
        return optimizer.step()
    import joint_group_adam_cuda as native
    if GROUP_NATIVE_OVERRIDE is not None:native=GROUP_NATIVE_OVERRIDE
    if not getattr(native,'all_original_groups_single_float32_launch',False):
        raise RuntimeError('Explicit joint-group float32 Adam backend is unavailable')
    beta1,beta2=policy[0];epsilon=policy[1]
    step_sizes=[];correction2=[]
    for parameter,gradient,state,rate in entries:
        state['step'].add_(1);step=float(state['step'])
        step_sizes.append(rate/(1-beta1**step))
        correction2.append(math.sqrt(1-beta2**step))
    GROUP_KERNEL_CALLS+=native.step([x[0] for x in entries],[x[1] for x in entries],
        [x[2]['exp_avg'] for x in entries],[x[2]['exp_avg_sq'] for x in entries],
        step_sizes,correction2,beta1,beta2,epsilon)
    torch.autograd.graph.increment_version([x[0] for x in entries]+
        [x[2]['exp_avg'] for x in entries]+[x[2]['exp_avg_sq'] for x in entries])
    return None
