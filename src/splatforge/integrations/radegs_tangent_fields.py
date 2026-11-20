"""FP32 camera-factor preparation for the constrained RaDe geometry candidate.

Normalize the homogeneous quaternion matrix after composition with the camera
rotation, then apply tangent scales. This preserves the real-arithmetic map
R_camera R_quaternion diag(scales), avoiding the normalized/scaled world-factor
intermediate. No precision, regularizer, camera or loss semantics change.
"""
import torch


def prepare_camera_factor(means, quats, scales, view):
    w,x,y,z=quats.unbind(-1)
    squared_norm=quats.square().sum(-1).clamp_min(1e-24)
    numerator=torch.stack((squared_norm-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),squared_norm-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),squared_norm-2*(x*x+y*y)),-1).reshape(-1,3,3)
    factor=(view[:3,:3]@numerator)/squared_norm[:,None,None]
    factor=factor*scales[:,None,:]
    camera=means@view[:3,:3].T+view[:3,3]
    return factor,camera
