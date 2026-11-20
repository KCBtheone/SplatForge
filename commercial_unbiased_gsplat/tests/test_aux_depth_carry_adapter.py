"""CPU dispatch contract: preserve old ABI, bind choice to autograd context, reject missing native feature."""
import sys,unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from unbiased_gsplat import _cuda

class AuxDepthCarryAdapterTests(unittest.TestCase):
 def invoke(self,enabled,available=True):
  calls=[]
  def forward(means,ray,opacity,*args):
   return torch.ones((1,1,1)),torch.ones((1,1,1),dtype=torch.bool),torch.ones((1,1,1)),torch.zeros((1,1,1),dtype=torch.int32)
  def backward(*args):
   calls.append(args)
   return torch.zeros_like(args[0]),torch.zeros_like(args[1]),torch.zeros_like(args[2])
  extension=SimpleNamespace(rasterize_unbiased_aux=forward,rasterize_unbiased_aux_backward=backward,optional_aux_depth_vjp_carry=available)
  with patch.object(_cuda,'unbiased_gsplat_cuda',extension),patch.object(_cuda,'AUX_DEPTH_VJP_CARRY',enabled),patch.object(_cuda,'AUX_DEPTH_VJP_CARRY_CALLS',0):
   ray=torch.ones((1,9),requires_grad=True)
   depth,_,conv,_=_cuda.rasterize_unbiased_aux(torch.ones((1,2),requires_grad=True),ray,torch.ones(1,requires_grad=True),torch.zeros((1,1,1),dtype=torch.int32),torch.zeros(1,dtype=torch.int32),1,1,16,.1,.6,1.)
   # A later mode change must not rewrite a previously created graph's choice.
   _cuda.AUX_DEPTH_VJP_CARRY=not enabled
   (depth.sum()+conv.sum()).backward()
   return calls,_cuda.AUX_DEPTH_VJP_CARRY_CALLS
 def test_default_keeps_old_native_abi(self):
  calls,count=self.invoke(False,available=False)
  self.assertEqual(len(calls),1);self.assertEqual(len(calls[0]),14);self.assertEqual(count,0)
 def test_opt_in_is_bound_to_forward_context(self):
  calls,count=self.invoke(True)
  self.assertEqual(len(calls),1);self.assertEqual(len(calls[0]),15);self.assertIs(calls[0][-1],True);self.assertEqual(count,1)
 def test_missing_native_feature_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'unavailable'):
   self.invoke(True,available=False)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA device required')
class NativeCarryCudaContractTests(unittest.TestCase):
 def arguments(self,depths,opacity,threshold,gc):
  n=len(depths);means=torch.full((n,2),.5,device='cuda');ray=torch.zeros((n,3,3),device='cuda');ray[:,0,0]=1;ray[:,1,1]=1;ray[:,2,2]=torch.tensor(depths,device='cuda')
  return (means,ray.reshape(n,9),torch.full((n,),opacity,device='cuda'),torch.zeros((1,1,1),device='cuda',dtype=torch.int32),torch.arange(n,device='cuda',dtype=torch.int32),torch.full((1,1,1),.4,device='cuda'),torch.full((1,1,1),gc,device='cuda'),1,1,16,.1,threshold,1.,1.25)
 def test_native_selected_hits_and_termination(self):
  self.assertTrue(_cuda.unbiased_gsplat_cuda.optional_aux_depth_vjp_carry)
  for label,depths,opacity,threshold,gc,expected in [
   ('middle',[1.,1.2,1.3],.2,.55,2.,[-.8,1.,.5]),
   ('last',[1.,2.,3.],.2,.85,2.,[0.,0.,.4]),
   ('opacity_stop',[1.,1.2,1.3],.999,.6,2.,[.4,0.,0.]),
   ('single',[1.],.7,.6,2.,[.4])]:
   with self.subTest(case=label):
    args=self.arguments(depths,opacity,threshold,gc);target=torch.zeros((len(depths),9),device='cuda');target[:,8]=torch.tensor(expected,device='cuda')
    for carry in (False,True):
     result=_cuda.unbiased_gsplat_cuda.rasterize_unbiased_aux_backward(*args,carry)
     torch.testing.assert_close(result[1],target,atol=1e-4,rtol=1e-4)
 def test_actual_autograd_opt_in_and_call_counter(self):
  args=self.arguments([1.,1.2,1.3],.2,.55,2.);means,ray,opacity,offsets,ids=args[:5];ray.requires_grad_()
  with patch.object(_cuda,'AUX_DEPTH_VJP_CARRY',True),patch.object(_cuda,'AUX_DEPTH_VJP_CARRY_CALLS',0):
   depth,valid,conv,_=_cuda.rasterize_unbiased_aux(means,ray,opacity,offsets,ids,1,1,16,.1,.55,1.,1.25)
   self.assertTrue(bool(valid.any()));torch.testing.assert_close(depth,torch.full_like(depth,1.2),atol=1e-4,rtol=1e-4)
   (.4*depth.sum()+2*conv.sum()).backward()
   torch.testing.assert_close(ray.grad[:,8],torch.tensor([-.8,1.,.5],device='cuda'),atol=1e-4,rtol=1e-4)
   self.assertEqual(_cuda.AUX_DEPTH_VJP_CARRY_CALLS,1)

if __name__=='__main__':unittest.main()
