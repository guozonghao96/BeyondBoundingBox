import torch
import torch.nn as nn
import numpy as np
from ..registry import LOSSES
from .utils import weighted_loss
# from mmdet.ops.iou import convex_iou_element

def contrastive_func(source_map,
                     target_map, 
                     match_pos_index=None,
                     t=1.0,
                     p=2.0,
                     reduction='mean'):
    N, C = source_map.size()
    source_vector = source_map.reshape(N, -1)
    target_vector = target_map.reshape(N, -1)
    all_vector = torch.cat([source_vector, target_vector], 0)
    multiply_all_vector = torch.mm(all_vector, all_vector.t())    
    norm_vector = torch.norm(all_vector, p=p, keepdim=True, dim=-1)
    multiply_norm_vector = torch.mm(norm_vector, norm_vector.t())
    multiply_norm_vector = torch.clamp(multiply_norm_vector, min=1e-8)
    cos_similarity = (multiply_all_vector / multiply_norm_vector) / t
    exp_cos_similarity = torch.exp(cos_similarity)
    loss = 0
    half = exp_cos_similarity.size(0) // 2
    
    if match_pos_index is None:
        for i in range(exp_cos_similarity.size(0)):
            if i < exp_cos_similarity.size(0) // 2:
                loss += -torch.log(exp_cos_similarity[i, i + half] /
                                    (exp_cos_similarity[i].sum() - exp_cos_similarity[i, i]))
            else:
                loss += -torch.log(exp_cos_similarity[i, i - half] /
                                    (exp_cos_similarity[i].sum() - exp_cos_similarity[i, i]))
        if reduction == 'mean':
            return loss / exp_cos_similarity.size(0)
        else:
            return loss
    else:
        assert half == len(match_pos_index), 'neg_index must be the same long as source map'
            ## 只有++例或者+-例的,将直接跳过，返回loss=0
#         have_neg_pos = True
        all_pos = False
        only_one_pos = False
        
        one_pos = 0
        only_pos = 0
        for i in range(half):
            if len(match_pos_index[i]) == 1:
                one_pos += 1
            if len(match_pos_index[i]) == half:
                all_pos = True
                loss = (source_map.sum() + target_map.sum()) * 0
#                 print('no neg_pos')
                return loss
        if one_pos == half:
            only_one_pos = True
            
        if not all_pos:
            for i in range(half - 2 if not only_one_pos else half): # 无反例的正例，no -+ only ++          
                pos_indy = match_pos_index[i]
                num_pos_ = len(pos_indy)
                pos_indy = match_pos_index[i].repeat(2)
                pos_indy[-num_pos_:] = pos_indy[-num_pos_:] + half
                pos_indx = torch.ones_like(pos_indy) * i
                loss += -torch.log(exp_cos_similarity[i, i + half] /
                                (exp_cos_similarity[i].sum() - exp_cos_similarity[pos_indx, pos_indy].sum())
                            )
                if torch.isinf(loss) or torch.isnan(loss):
                    print(pos_indy, pos_indy.size())
                    print(exp_cos_similarity, exp_cos_similarity.size())
                    print(exp_cos_similarity[i])
                    print(exp_cos_similarity[pos_indx, pos_indy])
                    exit()
                
        if reduction == 'mean':
            return loss / half
        else:
            return loss
        
def sim_loss(cls_feats, match_ori_pos_index, have_neg, cls_feat, t, p, length, gap): # var_points [B, N, 256]
#     B, num_pos, channels = cls_feats.size()
#     loss_element = cls_feats.var(0)
#     loss = loss_element.sum() / num_pos
######################dist6#################    
#     B, num_pos, channels = cls_feats.size()
#     assert B == 4, 'must rotate 4 direction'    
#     cls_feats = cls_feats.permute(1, 0, 2)
#     sim_loss = 0
#     for feat_every_direction in cls_feats:
#         source_vector = torch.cat([feat_every_direction[0].repeat(3, 1), 
#                                    feat_every_direction[1].repeat(2, 1), 
#                                    feat_every_direction[2].repeat(1, 1)], dim=0)    
#         target_vector = torch.cat([feat_every_direction[1:], 
#                                   feat_every_direction[2:],
#                                   feat_every_direction[3:]], dim=0)

#         multiply_all_vector = torch.bmm(source_vector.unsqueeze(1), 
#                                         target_vector.unsqueeze(-1)).reshape(-1, 1)

#         target_norm_vector = torch.norm(target_vector, p=2, keepdim=True, dim=-1)
#         source_norm_vector = torch.norm(source_vector, p=2, keepdim=True, dim=-1)
#         multiply_norm_vector = torch.clamp(source_norm_vector * target_norm_vector, min=1e-8)
#         cos_similarity = (multiply_all_vector / multiply_norm_vector) / t   
#         sim_loss += (1 - cos_similarity).mean()       
#     loss = sim_loss / num_pos

#############局部统计方差 ############
    
#     thr_ori = torch.linspace(0, channels , channels // (length - gap)).int()
#     thr_new = torch.zeros_like(thr_ori)
#     thr_new[:-1] = thr_ori[1:]    
#     index_thr = torch.cat([thr_ori.reshape(-1, 1)[:-1], thr_new.reshape(-1, 1)[:-1]], dim=1)
#     split_section = index_thr[:, 1] - index_thr[:, 0]
    
#     cls_feats = cls_feats.permute(1, 0, 2)
#     for k, feat_every_direction in enumerate(cls_feats):
#         thr_feat = torch.split(feat_every_direction, split_section.numpy().tolist(), dim=1)
#         stack_thr_feat = torch.stack(thr_feat, dim=0)
#         thr_var = stack_thr_feat.var(-1).var(-1)
#     loss = thr_var / num_pos    

###########contrasitve loss ############
#     assert len(cls_feats) == 3, 'must rotate 3 match'
#     loss = 0
#     match_num = 0
#     for match_cls_feat, match_pos_index, neg in zip(cls_feats, match_ori_pos_index, have_neg):
#         if len(match_cls_feat) == 0:
#             loss += cls_feat.sum() * 0
#             continue
#         match_num += 1
#         match_cls_feat_cat = torch.cat(match_cls_feat)
#         source_feat = match_cls_feat_cat[0::2]
#         target_feat = match_cls_feat_cat[1::2]
#         loss += contrastive_func(source_feat, target_feat, match_pos_index, t=t, p=p)
#     loss = loss / match_num if match_num != 0 else loss 
##########kl loss ################
#     B, num_pos, channels = cls_feats.size()
#     assert B == 4, 'must rotate 4 direction'
#     cls_feats = cls_feats.permute(1, 0, 2)
#     kl_loss = 0
#     for feat_every_direction in cls_feats:
#         source_vector = torch.cat([feat_every_direction[0].repeat(3, 1), 
#                                    feat_every_direction[1].repeat(2, 1), 
#                                    feat_every_direction[2].repeat(1, 1)], dim=0)    
#         target_vector = torch.cat([feat_every_direction[1:], 
#                                   feat_every_direction[2:],
#                                   feat_every_direction[3:]], dim=0)
#         mean_prob = (source_vector + target_vector) / 2
        
#         prob_divide1 = (source_vector / mean_prob)
#         prob_divide2 = (target_vector / mean_prob)
#         kl_ = source_vector * torch.log(prob_divide1) / 2 + target_vector * torch.log(prob_divide2) / 2
#         kl_loss += kl_.sum(-1).mean(-1)
#     loss = kl_loss / num_pos  

##########sigmoid kl loss ################
#     B, H, W, C = cls_feats.size()
#     assert B == 4, 'must rotate 4 direction'
# #     cls_feats = cls_feats.permute(1, 0, 2)
#     cls_vector = cls_feats.reshape(B, 1, -1)
#     source_vector = torch.cat([cls_vector[0].repeat(3, 1), 
#                                cls_vector[1].repeat(2, 1), 
#                                cls_vector[2].repeat(1, 1)], dim=0)  
    
#     target_vector = torch.cat([cls_vector[1:].squeeze(1), 
#                                cls_vector[2:].squeeze(1), 
#                                cls_vector[3:].squeeze(1)], dim=0)   
    
#     mean_prob = (source_vector + target_vector) / 2
    
#     prob_divide1 = (source_vector / mean_prob)
#     prob_divide2 = (target_vector / mean_prob)
#     kl_ = source_vector * torch.log(prob_divide1) / 2 + target_vector * torch.log(prob_divide2) / 2
#     loss = kl_.sum(-1).mean(-1)

# ########## cam var ################
#     B, C, H, W = cls_feats.size()
#     assert B == 4, 'must rotate 4 direction'
#     cls_feats = cls_feats.permute(1, 0, 2, 3)
#     loss_element = cls_feats.var(1).reshape(C, -1)
#     loss = loss_element.mean(-1).sum()

# ########## cam contrastive ################
#     B, C, H, W = cls_feats.size()
#     assert B == 4, 'must rotate 4 direction'
#     cls_vector = cls_feats.reshape(B, C, -1)
#     source_vector = cls_vector[0]
#     loss = 0
#     for i in range(3):
#         target_vector = cls_vector[i]
#         loss += contrastive_func(source_vector, target_vector)
#     loss /= 3

# ########## cam ||~||1 ################
    B, C, H, W = cls_feats.size()
    assert B == 4, 'must rotate 4 direction'
    ori_cls_feat = cls_feats[0]
    loss = 0
    for cls_feat in cls_feats[1:]:
        loss += torch.mean(torch.abs(ori_cls_feat - cls_feat))
#     loss /= 3
    return loss


@LOSSES.register_module
class SimLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, p=2.0, t=1.0, length=32, gap=4):
        super(SimLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.t = t
        self.p = p
        self.length = length
        self.gap = gap
    
    def forward(self,
                cls_feats,
                match_ori_pos_index=None,
                have_neg=None,
                cls_feat=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_sim = self.loss_weight * sim_loss(
            cls_feats,
            match_ori_pos_index,
            have_neg,
            cls_feat,
            self.t,
            self.p,
            self.length,
            self.gap
        )
        return loss_sim
