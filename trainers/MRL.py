import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import torch.distributed as dist


'''
Loss function for 3D Matryoshka Representation Learning 
'''
class MRL_Projection_Layer(nn.Module):
    def __init__(self, nesting_list: List, out_dim=1280, efficient=False, **kwargs):
        super(MRL_Projection_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.out_dim = out_dim # your dim clip expects
        self.efficient = efficient # in this parameter is the total dim
        
        if self.efficient:
            # Cria apenas UMA matriz grande projetando a maior fatia para o CLIP
            self.proj_0 = nn.Linear(nesting_list[-1], self.out_dim, **kwargs)      
        else:   
            # Cria uma camada de projeção independente para cada granularidade
            for i, dim in enumerate(self.nesting_list):
                setattr(self, f"proj_{i}", nn.Linear(dim, self.out_dim, **kwargs))    

    def forward(self, x):
        projected_logits = ()
        for i, dim in enumerate(self.nesting_list):
            if self.efficient:
                # slicing from projection weight and bias to project only the relevant dimensions
                weight_slice = self.proj_0.weight[:, :dim]
                out = torch.matmul(x[:, :dim], weight_slice.t())
                if self.proj_0.bias is not None:
                    out += self.proj_0.bias
                projected_logits += (out, )
            else:
                # Projeção tradicional por camadas separadas
                projected_logits += (getattr(self, f"proj_{i}")(x[:, :dim]),)

        return projected_logits


class MRL_Linear_Heads(nn.Module):
    def __init__(self, mrl_dims, out_dim):
        super().__init__()

        self.heads = nn.ModuleDict({
            str(dim): nn.Linear(dim, out_dim) for dim in mrl_dims
        })

    def forward(self, features):
        outputs = {}
        for dim_str, head in self.heads.items():
            dim = int(dim_str)
            outputs[dim_str] = head(features[:, :dim])
        return outputs


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

            acc_1 = (logits_1_to_2.argmax(dim=1) == labels).float().mean()
            acc_2 = (logits_2_to_1.argmax(dim=1) == labels).float().mean()
            dim_acc = (acc_1 + acc_2) / 2

            loss_dims[f"loss_{dim}d"] = dim_loss.item()
            acc_dims[f"acc_{dim}d"] = dim_acc.item()

        max_dim = self.nesting_list[-1]
        total_acc = acc_dims[f"acc_{max_dim}d"]

        return total_loss, total_acc, loss_dims, acc_dims


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

