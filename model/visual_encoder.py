import torch
from torch import nn, Tensor
from .external_tools.pointnet2.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
import numpy as np
from .common.utils import *

class PointNetPP(nn.Module):
    """
    Pointnet++ encoder.
    For the hyper parameters please advise the paper (https://arxiv.org/abs/1706.02413)
    """

    def __init__(self, sa_n_points: list,
                 sa_n_samples: list,
                 sa_radii: list,
                 sa_mlps: list,
                 bn=True,
                 use_xyz=True):
        super().__init__()
        n_sa = len(sa_n_points)
        if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
            raise ValueError('Lens of given hyper-params are not compatible')

        self.encoder = nn.ModuleList()

        for i in range(n_sa):
            self.encoder.append(PointnetSAModuleMSG(
                npoint=sa_n_points[i],
                nsamples=sa_n_samples[i],
                radii=sa_radii[i],
                mlps=sa_mlps[i],
                bn=bn,
                use_xyz=use_xyz,
            ))

        out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
        self.fc = nn.Linear(out_n_points * sa_mlps[-1][-1][-1], sa_mlps[-1][-1][-1])

    def forward(self, features):
        """
        @param features: B x N_objects x N_Points x 3 + C
        """
        xyz, features = break_up_pc(features)

        for i in range(len(self.encoder)):
            xyz, features = self.encoder[i](xyz, features)
        return self.fc(features.view(features.size(0), -1))

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotate_number = 4
        self.device = torch.cuda.current_device()
        self.obj_feature_mapping = create_mapping(768, 768, 0.1)
        self.box_feature_mapping = create_mapping(7, 768, 0.1)
        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                        sa_n_samples=[[32], [32], [None]],
                                        sa_radii=[[0.2], [0.4], [None]],
                                        sa_mlps=[[[3, 64, 64, 128]],
                                                [[128, 128, 128, 256]],
                                                [[256, 256, 768, 768]]])
        self.post_obj_enc = nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=768, 
                                                nhead=8, dim_feedforward=2048, activation="gelu"), num_layers=2)
    
    @torch.no_grad()
    def aug_input(self, input_points, contrast_range=(0.5, 1.5), noise_std_dev=0.02):
        input_points = input_points.float().to(self.device)
        xyz = input_points[:, :, :, :3]  # Get x,y,z coordinates (B, N, P, 3)
        B, N, P = xyz.shape[:3]  # Get dimensions
        input_points_multiview = []
        rgb = input_points[..., 3:6].clone() # (B, N, P, 3)
        # Randomly rotate/color_aug if training
        if self.training:
            rotate_matrix = get_random_rotation_matrix(self.rotate_number, self.device)
            xyz = torch.matmul(xyz.reshape(B*N*P, 3), rotate_matrix).reshape(B, N, P, 3)
            rgb = get_augmented_color(rgb, contrast_range, noise_std_dev, self.device) 
        # multi-view
        for theta in torch.Tensor([i*2.0*np.pi/self.rotate_number for i in range(self.rotate_number)]).to(self.device):  
            rotate_matrix = get_rotation_matrix(theta, self.device)
            rotated_xyz = torch.matmul(xyz.reshape(B*N*P, 3), rotate_matrix).reshape(B, N, P, 3)
            rotated_input_points = torch.clone(input_points)
            rotated_input_points[..., :3] = rotated_xyz
            rotated_input_points[..., 3:6] = rgb
            input_points_multiview.append(rotated_input_points)
        # Stack list of tensors into a single tensor
        input_points_multiview = torch.stack(input_points_multiview, dim=1)
        return input_points_multiview
    
    @torch.no_grad()
    def aug_box(self, box_infos):
        box_infos = box_infos.float().to(self.device)
        bxyz = box_infos[...,:3] # B,N,3
        B,N = bxyz.shape[:2]
        bxyz[..., 0] = scale_to_unit_range(bxyz[..., 0]) # normed x
        bxyz[..., 1] = scale_to_unit_range(bxyz[..., 1]) # normed y
        bxyz[..., 2] = scale_to_unit_range(bxyz[..., 2]) # normed z
        # Randomly rotate if training
        if self.training:
            rotate_matrix = get_random_rotation_matrix(self.rotate_number, self.device)
            bxyz = torch.matmul(bxyz.reshape(B*N, 3), rotate_matrix).reshape(B,N,3)        
        # multi-view
        bsize = box_infos[...,3:]
        boxs=[]
        for theta in torch.Tensor([i*2.0*np.pi/self.rotate_number for i in range(self.rotate_number)]).to(self.device):
            rotate_matrix = get_rotation_matrix(theta, self.device)
            rxyz = torch.matmul(bxyz.reshape(B*N, 3),rotate_matrix).reshape(B,N,3)
            boxs.append(torch.cat([rxyz,bsize],dim=-1))
        boxs=torch.stack(boxs,dim=1)
        return boxs
    
    def forward(self, obj_points, boxes):
        first_layer_with_params = next(l for l in self.box_feature_mapping if hasattr(l, 'weight'))
        obj_points = self.aug_input(obj_points) # B, R, N, P, 6
        B,R,N,P = obj_points.shape[:4]
        obj_points = obj_points.reshape(B*R,N,P,6)
        boxes = self.aug_box(boxes)
        box_feature = self.box_feature_mapping(boxes).reshape(B*R, N, 768)
        
        ## obj_encoding
        objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack) # (B*R, N, D)        
        ## obj_encoding
        obj_feats = self.obj_feature_mapping(objects_features) # (B*R, N, D)  
        obj_feats += box_feature
        obj_feats = self.post_obj_enc(obj_feats.transpose(0,1)).transpose(0,1)
        obj_feats = rotation_aggregate(obj_feats.reshape(B,R,N,768))
        
        return obj_feats