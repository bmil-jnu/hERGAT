# hergat/model.py
# hERGAT model architecture (PyTorch) + attention outputs
# - Compatible with checkpoints saved as state_dict
# - Device is configurable (no hardcoded cuda index)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HergatHParams:
    radius: int
    T: int
    input_feature_dim: int
    input_bond_dim: int
    fingerprint_dim: int
    output_units_num: int
    p_dropout: float

    physicochemical_feature_dim: int
    physicochemical_feature_dim_1: int
    physicochemical_feature_dim_2: int
    final1_fc1: int
    final1_fc2: int


class Fingerprint(nn.Module):
    """
    Graph-attention + GRU message passing used in the original notebook.
    Returns intermediate tensors useful for attention visualization.
    """
    def __init__(
        self,
        radius: int,
        T: int,
        input_feature_dim: int,
        input_bond_dim: int,
        fingerprint_dim: int,
        p_dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.radius = int(radius)
        self.T = int(T)
        self.device = device

        # Graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim + input_bond_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for _ in range(self.radius)])
        self.align = nn.ModuleList([nn.Linear(2 * fingerprint_dim, 1) for _ in range(self.radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for _ in range(self.radius)])

        # Graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2 * fingerprint_dim, 1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)

        self.dropout = nn.Dropout(p=float(p_dropout))

    def forward(
        self,
        atom_list: torch.Tensor,
        bond_list: torch.Tensor,
        atom_degree_list: torch.Tensor,
        bond_degree_list: torch.Tensor,
        atom_mask: torch.Tensor,
    ):
        """
        Shapes (batch=1 is OK):
          atom_list:        (B, L, A)
          bond_list:        (B, M, Bdim)
          atom_degree_list: (B, L, K)  int64 indices
          bond_degree_list: (B, L, K)  int64 indices
          atom_mask:        (B, L)     0/1
        """
        device = self.device
        atom_list = atom_list.to(device)
        bond_list = bond_list.to(device)
        atom_degree_list = atom_degree_list.to(device)
        bond_degree_list = bond_degree_list.to(device)
        atom_mask = atom_mask.to(device)

        atom_mask_3d = atom_mask.unsqueeze(2)  # (B, L, 1)

        batch_size, mol_length, _ = atom_list.size()
        atom_feature = F.relu(self.atom_fc(atom_list))

        atom_feature_viz: List[torch.Tensor] = []
        atom_feature_viz.append(self.atom_fc(atom_list))

        # gather neighbor atom/bond features
        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)

        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.relu(self.neighbor_fc(neighbor_feature))

        # masks (padding token index is mol_length - 1)
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length - 1] = 1
        attend_mask[attend_mask == mol_length - 1] = 0
        attend_mask = attend_mask.float().unsqueeze(-1).to(device)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length - 1] = 0
        softmax_mask[softmax_mask == mol_length - 1] = -9
        softmax_mask = softmax_mask.float().unsqueeze(-1).to(device)

        _, _, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

        align_score = F.relu(self.align[0](self.dropout(feature_align)))
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, -2)
        attention_weight = attention_weight * attend_mask

        atom_attention_weight_viz: List[torch.Tensor] = []
        atom_attention_weight_viz.append(attention_weight)

        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        context = torch.sum(attention_weight * neighbor_feature_transform, -2)
        context = F.relu(context)

        context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

        activated_features = F.relu(atom_feature)
        atom_feature_viz.append(activated_features)

        for d in range(1, self.radius):
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            neighbor_feature = torch.stack(neighbor_feature, dim=0)

            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
            feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

            align_score = F.relu(self.align[d](self.dropout(feature_align)))
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score, -2)
            attention_weight = attention_weight * attend_mask
            atom_attention_weight_viz.append(attention_weight)

            neighbor_feature_transform = self.attend[d](self.dropout(neighbor_feature))
            context = torch.sum(attention_weight * neighbor_feature_transform, -2)
            context = F.relu(context)

            context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

            activated_features = F.relu(atom_feature)
            atom_feature_viz.append(activated_features)

        mol_feature_unbounded_viz: List[torch.Tensor] = []
        mol_feature_unbounded_viz.append(torch.sum(atom_feature * atom_mask_3d, dim=-2))

        mol_feature = torch.sum(activated_features * atom_mask_3d, dim=-2)
        activated_features_mol = F.relu(mol_feature)

        mol_feature_viz: List[torch.Tensor] = []
        mol_feature_viz.append(mol_feature)

        mol_attention_weight_viz: List[torch.Tensor] = []
        mol_softmax_mask = atom_mask_3d.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.float().to(device)

        for _ in range(self.T):
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * atom_mask_3d
            mol_attention_weight_viz.append(mol_attention_weight)

            activated_features_transform = self.mol_attend(self.dropout(activated_features))
            mol_context = torch.sum(mol_attention_weight * activated_features_transform, -2)
            mol_context = F.relu(mol_context)

            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
            mol_feature_unbounded_viz.append(mol_feature)

            activated_features_mol = F.relu(mol_feature)
            mol_feature_viz.append(activated_features_mol)

        mol_prediction = mol_feature  # (B, fingerprint_dim)

        return (
            atom_feature,
            atom_feature_viz,
            atom_attention_weight_viz,
            mol_feature_viz,
            mol_feature_unbounded_viz,
            mol_attention_weight_viz,
            mol_prediction,
        )


class ExtendedFingerprint(nn.Module):
    """
    hERGAT final classifier that concatenates graph embedding + physicochemical vector.
    This matches the notebook's ExtendedFingerprint_viz layer layout and naming.
    """
    def __init__(self, hp: HergatHParams, device: torch.device):
        super().__init__()
        self.hp = hp
        self.device = device

        self.base = Fingerprint(
            radius=hp.radius,
            T=hp.T,
            input_feature_dim=hp.input_feature_dim,
            input_bond_dim=hp.input_bond_dim,
            fingerprint_dim=hp.fingerprint_dim,
            p_dropout=hp.p_dropout,
            device=device,
        )

        self.physicochemical_fc = nn.Linear(hp.physicochemical_feature_dim, hp.physicochemical_feature_dim_1)
        self.physicochemical_bn = nn.BatchNorm1d(hp.physicochemical_feature_dim_1)

        self.physicochemical_fc2 = nn.Linear(hp.physicochemical_feature_dim_1, hp.physicochemical_feature_dim_2)
        self.physicochemical_bn2 = nn.BatchNorm1d(hp.physicochemical_feature_dim_2)

        self.physicochemical_fc3 = nn.Linear(hp.physicochemical_feature_dim_2, hp.fingerprint_dim)
        self.physicochemical_bn3 = nn.BatchNorm1d(hp.fingerprint_dim)

        self.final1_fc = nn.Linear(hp.fingerprint_dim * 2, hp.final1_fc1)
        self.bn_final1 = nn.BatchNorm1d(hp.final1_fc1)

        self.final2_fc = nn.Linear(hp.final1_fc1, hp.final1_fc2)
        self.bn_final2 = nn.BatchNorm1d(hp.final1_fc2)

        self.final3_fc = nn.Linear(hp.final1_fc2, hp.output_units_num)

        self.dropout = nn.Dropout(p=float(hp.p_dropout))

    def forward(
        self,
        atom_list: torch.Tensor,
        bond_list: torch.Tensor,
        atom_degree_list: torch.Tensor,
        bond_degree_list: torch.Tensor,
        atom_mask: torch.Tensor,
        physicochemical_features: torch.Tensor,
    ):
        physicochemical_features = physicochemical_features.to(self.device)

        (
            atom_feature,
            atom_feature_viz,
            atom_attention_weight_viz,
            mol_feature_viz,
            mol_feature_unbounded_viz,
            mol_attention_weight_viz,
            mol_prediction,
        ) = self.base(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)

        x = F.relu(self.physicochemical_bn(self.physicochemical_fc(physicochemical_features)))
        x = self.dropout(x)
        x = F.relu(self.physicochemical_bn2(self.physicochemical_fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.physicochemical_bn3(self.physicochemical_fc3(x)))
        x = self.dropout(x)

        combined = torch.cat([mol_prediction, x], dim=-1)

        x1 = F.relu(self.bn_final1(self.final1_fc(combined)))
        x1 = self.dropout(x1)
        x2 = F.relu(self.bn_final2(self.final2_fc(x1)))
        x2 = self.dropout(x2)
        logits = self.final3_fc(x2)
        probs = F.softmax(logits, dim=1)

        return (
            atom_feature,
            atom_feature_viz,
            atom_attention_weight_viz,
            mol_feature_viz,
            mol_feature_unbounded_viz,
            mol_attention_weight_viz,
            probs,
        )
