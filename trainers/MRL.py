import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import torch.distributed as dist



'''
Loss function for Matryoshka Representation Learning 
'''



class GatherLayer(torch.autograd.Function):
    """
    Realiza o all_gather mantendo o fluxo de gradientes (backward pass).
    """
    @staticmethod
    def forward(ctx, x):
        # Cria tensores vazios para receber os dados de todas as GPUs
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        # No backward, empilhamos os gradientes que vieram de todas as operações
        all_gradients = torch.stack(grads)
        # Somamos os gradientes de todas as GPUs (all_reduce)
        dist.all_reduce(all_gradients)
        # Cada GPU pega de volta apenas o gradiente correspondente ao seu 'rank'
        return all_gradients[dist.get_rank()]

def gather_features(features):
    """Função auxiliar para aplicar o GatherLayer e concatenar"""
    if dist.is_available() and dist.is_initialized():
        gathered = GatherLayer.apply(features)
        return torch.cat(gathered, dim=0)
    return features



class MRL_Contrastive_Loss:
    def __init__(self, config, nesting_list):
        self.config = config
        self.nesting_list = nesting_list

    def __call__(self, feat1, feat2, logit_scale=1, mask=None):
        total_loss = 0.0
        
        # Dicionários separados para bater com o que seu código espera
        loss_dims = {}
        acc_dims = {}
        
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        local_batch_size = feat1.shape[0]

        for dim in self.nesting_list:
            # 1. Fatiamento e Normalização
            norm_feat1 = F.normalize(feat1[:, :dim], dim=1)
            norm_feat2 = F.normalize(feat2[:, :dim], dim=1)

            # 2. Gather Diferenciável
            global_feat1 = gather_features(norm_feat1)
            global_feat2 = gather_features(norm_feat2)

            # 3. Multiplicação
            logits_1_to_2 = logit_scale * norm_feat1 @ global_feat2.T
            logits_2_to_1 = logit_scale * norm_feat2 @ global_feat1.T

            if mask is not None:
                logits_1_to_2 = logits_1_to_2 * mask
                logits_2_to_1 = logits_2_to_1 * mask

            # 4. Labels
            labels = torch.arange(local_batch_size, device=feat1.device) + (rank * local_batch_size)

            # 5. Loss
            loss_1 = F.cross_entropy(logits_1_to_2, labels)
            loss_2 = F.cross_entropy(logits_2_to_1, labels)
            dim_loss = (loss_1 + loss_2) / 2
            total_loss += dim_loss

            # 6. Acurácia
            acc_1 = (logits_1_to_2.argmax(dim=1) == labels).float().mean()
            acc_2 = (logits_2_to_1.argmax(dim=1) == labels).float().mean()
            dim_acc = (acc_1 + acc_2) / 2

            # 7. Salvando nos dicionários
            loss_dims[f"loss_{dim}d"] = dim_loss.item()
            acc_dims[f"acc_{dim}d"] = dim_acc.item()

        # Usando a acurácia da dimensão máxima (última da lista) como a acurácia global da época
        max_dim = self.nesting_list[-1]
        total_acc = acc_dims[f"acc_{max_dim}d"]

        # Retorna exatamente as 4 variáveis que o seu mrl_trainer.py linha 416 espera
        return total_loss, total_acc, loss_dims, acc_dims













# class Matryoshka_CE_Loss(nn.Module):
# 	def __init__(self, relative_importance: List[float]=None, **kwargs):
# 		super(Matryoshka_CE_Loss, self).__init__()

# 		self.criterion = nn.CrossEntropyLoss(**kwargs)
# 		# relative importance shape: [G]
# 		self.relative_importance = relative_importance

# 	def forward(self, output, target):
# 		# output shape: [G granularities, N batch size, C number of classes]
# 		# target shape: [N batch size]

# 		# Calculate losses for each output and stack them. This is still O(N)
# 		losses = torch.stack([self.criterion(output_i, target) for output_i in output])
		
# 		# Set relative_importance to 1 if not specified
# 		rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)
		
# 		# Apply relative importance weights
# 		weighted_losses = rel_importance * losses
# 		return weighted_losses.sum()


# class MRL_Linear_Layer(nn.Module):
# 	def __init__(self, nesting_list: List, num_classes=0, efficient=False, **kwargs):
# 		super(MRL_Linear_Layer, self).__init__()
		
# 		self.nesting_list = nesting_list
# 		self.num_classes = num_classes # Number of classes for classification
# 		self.efficient = efficient

# 		#Fixed features according to the dim
# 		if self.efficient:
# 			setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))		
# 		else:	
# 			for i, num_feat in enumerate(self.nesting_list):
# 				setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))	

# 	def reset_parameters(self):
# 		if self.efficient:
# 			self.nesting_classifier_0.reset_parameters()
# 		else:
# 			for i in range(len(self.nesting_list)):
# 				getattr(self, f"nesting_classifier_{i}").reset_parameters()


# 	def forward(self, x):
# 		nesting_logits = ()
# 		for i, num_feat in enumerate(self.nesting_list):
# 			if self.efficient:
# 				if self.nesting_classifier_0.bias is None:
# 					nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
# 				else:
# 					nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
# 			else:
# 				nesting_logits +=  (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)

# 		return nesting_logits


# class FixedFeatureLayer(nn.Linear):
#     '''
#     For our fixed feature baseline, we just replace the classification layer with the following. 
#     It effectively just look at the first "in_features" for the classification. 
#     '''

#     def __init__(self, in_features, out_features, **kwargs):
#         super(FixedFeatureLayer, self).__init__(in_features, out_features, **kwargs)

#     def forward(self, x):
#         if not (self.bias is None):
#             out = torch.matmul(x[:, :self.in_features], self.weight.t()) + self.bias
#         else:
#             out = torch.matmul(x[:, :self.in_features], self.weight.t())
#         return out
        