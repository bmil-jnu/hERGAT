import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:1')

class Fingerprint_viz(nn.Module):
    def __init__(self, radius, T, input_feature_dim, input_bond_dim,
                 fingerprint_dim, output_units_num, p_dropout):
        super(Fingerprint_viz, self).__init__()

        # Graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim+input_bond_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim, 1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])

        # Graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2*fingerprint_dim, 1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)

        self.dropout = nn.Dropout(p=p_dropout)
        # self.output = nn.Linear(fingerprint_dim, output_units_num)

        self.radius = radius
        self.T = T

    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        atom_list = atom_list.to(device)
        bond_list = bond_list.to(device)
        atom_degree_list = atom_degree_list.to(device)
        bond_degree_list = bond_degree_list.to(device)
        atom_mask = atom_mask.to(device)
        
        atom_mask = atom_mask.unsqueeze(2)
        batch_size, mol_length, num_atom_feat = atom_list.size()
        atom_feature = F.relu(self.atom_fc(atom_list))
        
        atom_feature_viz = []
        atom_feature_viz.append(self.atom_fc(atom_list))
                                
        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)

        # Concatenate atom and bond features
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.relu(self.neighbor_fc(neighbor_feature))

        # Generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.type(torch.FloatTensor).unsqueeze(-1).to(device)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9
        softmax_mask = softmax_mask.type(torch.FloatTensor).unsqueeze(-1).to(device)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

        align_score = F.relu(self.align[0](self.dropout(feature_align)))
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, -2)
        attention_weight = attention_weight * attend_mask
        
        atom_attention_weight_viz = []
        atom_attention_weight_viz.append(attention_weight)
                                
                                
                                
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
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
            context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
            context = F.relu(context)
            context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

            activated_features = F.relu(atom_feature)
            atom_feature_viz.append(activated_features)                    
            
        mol_feature_unbounded_viz = []
        mol_feature_unbounded_viz.append(torch.sum(atom_feature * atom_mask, dim = -2))
                                
        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        activated_features_mol = F.relu(mol_feature)
        
        mol_feature_viz = []
        mol_feature_viz.append(mol_feature)
        
                                
        mol_attention_weight_viz = []
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.FloatTensor).to(device)

        for t in range(self.T):
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * atom_mask
            mol_attention_weight_viz.append(mol_attention_weight)
                                
            activated_features_transform = self.mol_attend(self.dropout(activated_features))
            mol_context = torch.sum(torch.mul(mol_attention_weight, activated_features_transform), -2)
            mol_context = F.relu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
            mol_feature_unbounded_viz.append(mol_feature)

            activated_features_mol = F.relu(mol_feature)
            mol_feature_viz.append(activated_features_mol)

        mol_prediction = mol_feature

        return atom_feature, atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz, mol_attention_weight_viz, mol_prediction

        
        
class ExtendedFingerprint_viz(Fingerprint_viz):
    def __init__(self, radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, output_units_num, p_dropout, physicochemical_feature_dim):
        super().__init__(radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, output_units_num, p_dropout)
        self.physicochemical_fc = nn.Linear(physicochemical_feature_dim, 512)
        self.physicochemical_bn = nn.BatchNorm1d(512)
        self.physicochemical_fc2 = nn.Linear(512, fingerprint_dim)
        self.physicochemical_bn2 = nn.BatchNorm1d(fingerprint_dim)
        self.final1_fc = nn.Linear(fingerprint_dim * 2, 256)
        self.bn_final1 = nn.BatchNorm1d(256) # Added BatchNorm Layer

        self.final2_fc = nn.Linear(256, 64)
        self.bn_final2 = nn.BatchNorm1d(64) # Added BatchNorm Layer
        self.final3_fc = nn.Linear(64, output_units_num)

    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, physicochemical_features):

        physicochemical_features = physicochemical_features.to(device)

        atom_feature, atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz, mol_attention_weight_viz, mol_prediction = super().forward(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
        
        processed_physicochemical_features = F.relu(self.physicochemical_fc(physicochemical_features))
        processed_physicochemical_features2 = self.dropout(processed_physicochemical_features)
        processed_physicochemical_features3 = F.relu(self.physicochemical_fc2(processed_physicochemical_features2))
        processed_physicochemical_features4 = self.dropout(processed_physicochemical_features3)
        
        combined_feature_vector = torch.cat([mol_prediction, processed_physicochemical_features4], dim=-1)
        
        #fingerprint_dim => 1024
        final_prediction = F.relu(self.bn_final1(self.final1_fc(combined_feature_vector))) # Applied BatchNorm Layer
        final2_prediction = self.dropout(final_prediction)
        final3_prediction = F.relu(self.bn_final2(self.final2_fc(final2_prediction))) # Applied BatchNorm Layer
        final4_prediction = self.dropout(final3_prediction)
        final5_prediction = self.final3_fc(final4_prediction)
        final5_prediction = F.softmax(final5_prediction, dim = 1)

       
      
        return atom_feature, atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz, mol_attention_weight_viz, final5_prediction

