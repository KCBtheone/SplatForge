import unittest
import torch
from unbiased_gsplat.planar_gaussian import quaternion_rotation, primitive_plane_features, plane_depth
from unbiased_gsplat.regional_commercial import compose_two_families, semantic_and_leakage


class CommercialPlanarTests(unittest.TestCase):
    def test_plane_ratio_is_opacity_invariant_and_not_center_depth(self):
        k=torch.tensor([[20.,0,2],[0,20,2],[0,0,1]],dtype=torch.double)
        normal=torch.tensor([.3,0,-1.],dtype=torch.double)
        for opacity in [.02,.2,.9]:
            n=normal.expand(4,4,3)*opacity
            distance=torch.full((4,4),-2*opacity,dtype=torch.double)
            z,valid=plane_depth(n,distance,torch.full((4,4),opacity),k)
            x=(torch.arange(4,dtype=torch.double)+.5-2)/20
            expected=2/(1-.3*x)
            torch.testing.assert_close(z,expected.expand(4,4));self.assertTrue(valid.all())

    def test_minimum_axis_and_signed_orientation(self):
        means=torch.tensor([[0.,0,3.]])
        q=torch.tensor([[1.,0,0,0.]])
        n,d=primitive_plane_features(means,q,torch.tensor([[.2,.3,.01]]),torch.eye(4))
        torch.testing.assert_close(n,torch.tensor([[0.,0,-1.]]));self.assertEqual(d.item(),-3)
        torch.testing.assert_close(quaternion_rotation(q),torch.eye(3)[None])

    def test_ratio_gradcheck_and_grazing_rejection(self):
        n=torch.tensor([[[.2,.1,-1.]]],dtype=torch.double,requires_grad=True)
        d=torch.tensor([[-2.]],dtype=torch.double,requires_grad=True)
        k=torch.eye(3,dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:plane_depth(a,b,torch.ones(1,1),k)[0],(n,d)))
        z,v=plane_depth(torch.zeros_like(n),d,torch.ones(1,1),k)
        self.assertFalse(v.any());self.assertTrue(torch.isfinite(z).all())

    def test_unguarded_semantic_gradient_survives_trusted_body(self):
        ah=torch.tensor([[.8]],requires_grad=True);ab=torch.tensor([[.5]],requires_grad=True)
        def layer(a,z):return dict(alpha=a,depth=torch.full_like(a,z),valid=torch.ones_like(a,dtype=torch.bool),rgb=a[...,None].expand(-1,-1,3))
        body_domain=torch.ones_like(ah,dtype=torch.bool);hair_domain=~body_domain
        c=compose_two_families(layer(ah,2),layer(ab,3),trusted_body=body_domain)
        semantic,leak=semantic_and_leakage(c,hair_domain,body_domain)
        gradient=torch.autograd.grad(semantic,ah,retain_graph=True)[0]
        self.assertGreater(gradient.item(),0);self.assertGreater(leak.item(),0)
        torch.testing.assert_close(c['rgb'],layer(ab,3)['rgb'])

    def test_whole_layer_visibility_is_explicit(self):
        def l(a,z):return dict(alpha=torch.tensor([[a]]),depth=torch.tensor([[z]]),valid=torch.tensor([[True]]),rgb=torch.tensor([[[a,0.,0.]]]))
        c=compose_two_families(l(.8,2),l(.5,3))
        self.assertTrue(c['hair_selected'].item());self.assertAlmostEqual(c['native_alpha'].item(),.9,places=6)


if __name__=='__main__':unittest.main()
