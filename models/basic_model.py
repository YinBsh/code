import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        n_classes = 6

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self, audio, visual,w=None):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        if w is None:
            out = self.fusion_module(a, v)
        else:
            out = self.fusion_module(a*w[1], v*w[0])


        return out
