class MRL_Contrastive_Layer(nn.Module):
    """
    Adaptação da MRL_Linear_Layer para contrastive learning.
    
    Em vez de classificadores lineares por escala, usa um único
    projection head Linear(out_channel, clip_dim, bias=False)
    e fatia as colunas para cada nesting dim — igual ao efficient mode
    do paper MRL, mas para contrastive loss em vez de classificação.
    
    nesting_list : [8, 16, 32, 64, 128, 256, 512]
    out_channel  : 512  — saída do PointBERT
    clip_dim     : 1280 — espaço CLIP
    """
    def __init__(self, nesting_list: List[int], out_channel: int, clip_dim: int):
        super().__init__()
        self.nesting_list = nesting_list
        self.out_channel  = out_channel
        self.clip_dim     = clip_dim

        # Único projection head — equivalente ao efficient mode do paper
        # weight shape: (clip_dim, out_channel) = (1280, 512)
        # fatia colunas [:, :dim] para cada nesting dim
        self.nested_proj = nn.Linear(out_channel, clip_dim, bias=False)

    def reset_parameters(self):
        self.nested_proj.reset_parameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        x : (B, out_channel=512) — saída raw do PointBERT

        Retorna tuple de projeções, uma por nesting dim:
            ((B, clip_dim=1280), (B, clip_dim=1280), ...)
        
        Cada projeção usa só as primeiras `dim` colunas do PointBERT
        e as mapeia para o espaço CLIP completo (1280).
        """
        nesting_projs = ()
        W = self.nested_proj.weight  # (1280, 512)

        for dim in self.nesting_list:
            # fatia input e weight — igual ao efficient mode do paper
            x_sliced      = x[:, :dim]           # (B, dim)
            W_sliced      = W[:, :dim]            # (1280, dim)
            proj          = x_sliced @ W_sliced.T # (B, 1280)
            nesting_projs += (proj,)

        return nesting_projs  # tuple de (B, 1280) — um por dim


def mrl_loss(self, feat_pc, feat_clip, logit_scale=1, mask=None):
    """
    feat_pc   : (B, 512)  — saída raw do PointBERT
    feat_clip : (B, 1280) — CLIP frozen

    1. mrl_layer.forward(feat_pc) → tuple de projeções (B, 1280) por dim
    2. Para cada projeção: contrastive_loss vs CLIP completo
    """
    total_loss   = 0.0
    total_acc    = 0.0
    loss_per_dim = {}
    acc_per_dim  = {}

    weights = torch.ones(len(self.mrl_layer.nesting_list), device=self.config.device)

    # projeções aninhadas — uma por dim
    nesting_projs = self.mrl_layer(feat_pc)  # tuple de (B, 1280)

    for i, (dim, proj) in enumerate(zip(self.mrl_layer.nesting_list, nesting_projs)):

        # contrastive loss usando seu calc_contrastive_loss
        loss_dim, acc_dim = self.calc_contrastive_loss(
            proj,       # (B, 1280) — shape projetado para espaço CLIP
            feat_clip,  # (B, 1280) — CLIP completo, alvo fixo
            logit_scale,
            mask=mask,
        )

        total_loss += weights[i] * loss_dim
        total_acc  += acc_dim

        loss_per_dim[dim] = loss_dim.detach().item()
        acc_per_dim[dim]  = acc_dim.detach().item()

    return total_loss, total_acc / len(self.mrl_layer.nesting_list), loss_per_dim, acc_per_dim