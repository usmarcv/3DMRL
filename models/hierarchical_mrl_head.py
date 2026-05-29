# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Matryoskha_Constrastive_Loss(nn.Module):
#     def __init__(self, config):
#         super(Matryoskha_Constrastive_Loss, self).__init__()
        
#         self.config = config


#     def forward(self, feat1, feat2, mlr_dim, logit_scale=1.0, mask=None):
#         """
#         feat1, feat2:
#             - OU tensor [B, D]  (modo slicing)
#             - OU lista de tensores progressivos (modo residual head)

#         mlr_dim: lista de dimensões progressivas
#         """

#         total_loss = 0.0
#         total_acc = 0.0

#         loss_per_dim = {}
#         acc_per_dim = {}

#         # Detecta se é progressive head (lista) ou slicing
#         hierarchical_mrl_head = isinstance(feat1, list)

#         for i, dim_slice in enumerate(mlr_dim):

#             if hierarchical_mrl_head:
#                 # já vem separado por dimensão
#                 features1 = F.normalize(feat1[i], dim=1)
#                 features2 = F.normalize(feat2[i], dim=1)
#             else:
#                 # slicing tradicional
#                 features1 = F.normalize(feat1[:, :dim_slice], dim=1)
#                 features2 = F.normalize(feat2[:, :dim_slice], dim=1)

#             # -------- DDP --------
#             if self.config.ngpu > 1:
#                 all_features1 = torch.cat(
#                     torch.distributed.nn.all_gather(features1), dim=0
#                 )
#                 all_features2 = torch.cat(
#                     torch.distributed.nn.all_gather(features2), dim=0
#                 )

#                 logits = logit_scale * all_features1 @ all_features2.T
#             else:
#                 logits = logit_scale * features1 @ features2.T

        
#             if mask is not None:
#                 if mask.dtype == torch.bool:
#                     logits = logits.masked_fill(~mask, -1e9)
#                 else:
#                     logits = logits + (1.0 - mask) * -1e9

#             labels = torch.arange(logits.shape[0], device=self.config.device)

#             loss_dim = (
#                 F.cross_entropy(logits, labels) +
#                 F.cross_entropy(logits.T, labels)
#             ) / 2

#             acc_dim = (logits.argmax(dim=1) == labels).float().mean()

#             loss_per_dim[dim_slice] = loss_dim.detach().item()
#             acc_per_dim[dim_slice] = acc_dim.detach().item()

#             #set the relative importance of each dimension slice to 1.0 for now, but can be changed later
#             #the same used in MRL: https://github.com/RAIVNLab/MRL/blob/main/MRL.py#L26
#             relative_importance = torch.ones_like(loss_dim)

#             total_loss += relative_importance * loss_dim
#             total_acc += relative_importance * acc_dim

#         return total_loss, total_acc, loss_per_dim, acc_per_dim



# class Hierarchical_Matryoskha_Head(nn.Module):
#     def __init__(self, base_dim, nesting_list: List[int]):
#         super(Hierarchical_Matryoskha_Head, self).__init__()

#         self.base_dim = base_dim # the dimension of the output feature in your case is 1280 as well as the OpenShape
#         self.nesting_list = nesting_list
#         self.projection = nn.Linear(base_dim, nesting_list[0])
#         self.nested_projections = nn.ModuleList()

#         for i in range(1, len(nesting_list)):
#             in_dim = nesting_list[i-1]
#             out_dim = nesting_list[i] - nesting_list[i-1]
#             self.nested_projections.append(nn.Linear(in_dim, out_dim))

#     def forward(self, x):
#         # x is the output feature from the backbone, with shape (batch_size, base_dim)
#         # take the first dimension here
#         z = self.projection(x)  # shape: (batch_size, nesting_list[0])
#         outputs = [] #list for storing the outputs (embeddings) of each level of nesting 
#         # current = z
#         outputs.append(F.normalize(z, dim=-1)) #normalize

#         # hierarchically project the output to the next level of nesting
#         for i in range(1, len(self.nesting_list)):
#             # nested_input = outputs[-1]  # get the last output as input for the next projection
#             # nested_output = self.nested_projections[i-1](nested_input)  # shape: (batch_size, nesting_list[i] - nesting_list[i-1])
#             # outputs.append(nested_output)
#             x = self.nested_projections[i-1](z)  # shape: (batch_size, nesting_list[i] - nesting_list[i-1])
#             z = torch.cat([z, x], dim=-1)  # concatenate the previous output with the new x
#             outputs.append(F.normalize(z, dim=-1)) #normalize

#         return outputs  # list of tensors with shapes defined by nesting_list
