import torch
from torch import nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet1DModel
#from src.custom_unet import UNet1DModel  # 假设你将 UNet1DModel 保存为 cc

from modules.transformer import TransformerEncoder

# def _align_prompt(prompt: torch.Tensor, target_len: int) -> torch.Tensor:
#     """
#     将 prompt (shape [dim, old_len]) 填充或截断到长度 [dim, target_len]
#     """
#     dim, old_len = prompt.shape
#     if old_len == target_len:
#         return prompt
#     elif old_len < target_len:
#         pad_amount = target_len - old_len
#         return F.pad(prompt, (0, pad_amount), "constant", 0.0)
#     else:
#         return prompt[:, :target_len]
def _align_prompt(prompt: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    将 prompt (shape [batch, dim, old_len]) 或 (shape [dim, old_len]) 填充或截断到长度 [batch, dim, target_len] 或 [dim, target_len]
    """
    if len(prompt.shape) == 3:  # 如果是三维张量 [batch, dim, old_len]
        batch_size, dim, old_len = prompt.shape
        if old_len == target_len:
            return prompt
        elif old_len < target_len:
            pad_amount = target_len - old_len
            return F.pad(prompt, (0, pad_amount), "constant", 0.0)  # 填充
        else:
            return prompt[:, :, :target_len]  # 截断
    elif len(prompt.shape) == 2:  # 如果是二维张量 [dim, old_len]
        dim, old_len = prompt.shape
        if old_len == target_len:
            return prompt
        elif old_len < target_len:
            pad_amount = target_len - old_len
            return F.pad(prompt, (0, pad_amount), "constant", 0.0)  # 填充
        else:
            return prompt[:, :target_len]  # 截断
    else:
        raise ValueError("Prompt tensor must be either 2D or 3D.")

    

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = (
            hyp_params.orig_d_l,
            hyp_params.orig_d_a,
            hyp_params.orig_d_v,
        )
        self.d_l, self.d_a, self.d_v = (
            hyp_params.proj_dim,
            hyp_params.proj_dim,
            hyp_params.proj_dim,
        )
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        self.llen, self.alen, self.vlen = hyp_params.seq_len
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = hyp_params.output_dim

        # 1D Convolutional projections for each modality
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, bias=False)

        # Cross-modal attention sub-networks (six directions)
        self.trans_l_with_a = self.get_network(self_type="la")
        self.trans_l_with_v = self.get_network(self_type="lv")

        self.trans_a_with_l = self.get_network(self_type="al")
        self.trans_a_with_v = self.get_network(self_type="av")

        self.trans_v_with_l = self.get_network(self_type="vl")
        self.trans_v_with_a = self.get_network(self_type="va")

        # Self-attention memory sub-networks (one per modality)
        self.trans_l_mem = self.get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)

        # Fully connected layers for fusion + residual
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        self.device = torch.device("cuda" if hyp_params.use_cuda else "cpu")
        self.to(self.device)

    def get_network(self, self_type, layers=-1):
        """
        Returns a TransformerEncoder configured according to self_type:
        - For cross-attention, embed_dim = d_*
        - For self-attention memory, embed_dim = 2*d_* (since inputs concatenated)
        """
        if self_type == "la":
            embed_dim = self.d_l
            attn_dropout = self.attn_dropout
        elif self_type == "lv":
            embed_dim = self.d_l
            attn_dropout = self.attn_dropout
        elif self_type == "al":
            embed_dim = self.d_a
            attn_dropout = self.attn_dropout_a
        elif self_type == "av":
            embed_dim = self.d_a
            attn_dropout = self.attn_dropout_a
        elif self_type == "vl":
            embed_dim = self.d_v
            attn_dropout = self.attn_dropout_v
        elif self_type == "va":
            embed_dim = self.d_v
            attn_dropout = self.attn_dropout_v
        elif self_type == "l_mem":
            embed_dim = 2 * self.d_l
            attn_dropout = self.attn_dropout
        elif self_type == "a_mem":
            embed_dim = 2 * self.d_a
            attn_dropout = self.attn_dropout
        elif self_type == "v_mem":
            embed_dim = 2 * self.d_v
            attn_dropout = self.attn_dropout
        else:
            raise ValueError(f"Unknown network type: {self_type}")

        num_layers = layers if layers > 0 else self.layers
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=num_layers,
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def forward(self, x_l, x_a, x_v, missing_mod=None):
        """
        x_l: [batch, seq_len_l, orig_d_l]
        x_a: [batch, seq_len_a, orig_d_a]
        x_v: [batch, seq_len_v, orig_d_v]
        missing_mod: [batch]  (not used in MULTModel)
        """

        # Embedding dropout and shape adjustment
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        print(f"Shape of x_l after dropout & transpose: {x_l.shape}")  # [batch, orig_d_l, seq_len_l]
        x_a = x_a.transpose(1, 2)
        print(f"Shape of x_a after transpose: {x_a.shape}")  # [batch, orig_d_a, seq_len_a]
        x_v = x_v.transpose(1, 2)
        print(f"Shape of x_v after transpose: {x_v.shape}")  # [batch, orig_d_v, seq_len_v]

        max_len = max(x_l.size(2), x_a.size(2), x_v.size(2))

        

        # 对每个模态进行填充，确保它们的序列长度一致
        x_l = F.pad(x_l, (0, max_len - x_l.size(2)), "constant", 0.0)
        x_a = F.pad(x_a, (0, max_len - x_a.size(2)), "constant", 0.0)
        x_v = F.pad(x_v, (0, max_len - x_v.size(2)), "constant", 0.0)

        # Convolutional projections
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        print(f"Shape of proj_x_l after conv proj: {proj_x_l.shape}")  # [batch, d_l, seq_len_l]
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        print(f"Shape of proj_x_a after conv proj: {proj_x_a.shape}")  # [batch, d_a, seq_len_a]
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        print(f"Shape of proj_x_v after conv proj: {proj_x_v.shape}")  # [batch, d_v, seq_len_v]

        # Change to [seq_len, batch, dim] for Transformer
        proj_x_l = proj_x_l.permute(2, 0, 1)
        print(f"Shape of proj_x_l after permute: {proj_x_l.shape}")  # [seq_len_l, batch, d_l]
        proj_x_a = proj_x_a.permute(2, 0, 1)
        print(f"Shape of proj_x_a after permute: {proj_x_a.shape}")  # [seq_len_a, batch, d_a]
        proj_x_v = proj_x_v.permute(2, 0, 1)
        print(f"Shape of proj_x_v after permute: {proj_x_v.shape}")  # [seq_len_v, batch, d_v]

        # Text branch: cross-attention with audio & visual, then self-attention
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        print(f"Shape of h_l_with_as: {h_l_with_as.shape}")  # [seq_len_l, batch, d_l]
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        print(f"Shape of h_l_with_vs: {h_l_with_vs.shape}")  # [seq_len_l, batch, d_l]
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)  # [seq_len_l, batch, 2*d_l]
        print(f"Shape of h_ls after cat (text): {h_ls.shape}")
        h_ls = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]  # [batch, 2*d_l]
        print(f"Shape of last_h_l: {last_h_l.shape}")

        # Audio branch
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        print(f"Shape of h_a_with_ls: {h_a_with_ls.shape}")
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        print(f"Shape of h_a_with_vs: {h_a_with_vs.shape}")
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)  # [seq_len_a, batch, 2*d_a]
        print(f"Shape of h_as after cat (audio): {h_as.shape}")
        h_as = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        last_h_a = h_as[-1]  # [batch, 2*d_a]
        print(f"Shape of last_h_a: {last_h_a.shape}")

        # Visual branch
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        print(f"Shape of h_v_with_ls: {h_v_with_ls.shape}")
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        print(f"Shape of h_v_with_as: {h_v_with_as.shape}")
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)  # [seq_len_v, batch, 2*d_v]
        print(f"Shape of h_vs after cat (visual): {h_vs.shape}")
        h_vs = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]  # [batch, 2*d_v]
        print(f"Shape of last_h_v: {last_h_v.shape}")

        # Concatenate all modalities
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)  # [batch, combined_dim]
        print(f"Shape of last_hs after concatenation: {last_hs.shape}")

        # Residual block
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training)
        )
        print(f"Shape of last_hs_proj before residual add: {last_hs_proj.shape}")
        last_hs_proj = last_hs_proj + last_hs  # [batch, combined_dim]
        print(f"Shape of last_hs_proj after residual add: {last_hs_proj.shape}")

        output = self.out_layer(last_hs_proj)  # [batch, output_dim]
        print(f"Shape of output: {output.shape}")
        return output


class PromptModel(nn.Module):
    def __init__(self, hyp_params):
        super(PromptModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = (
            hyp_params.orig_d_l,
            hyp_params.orig_d_a,
            hyp_params.orig_d_v,
        )
        self.d_l, self.d_a, self.d_v = (
            hyp_params.proj_dim,
            hyp_params.proj_dim,
            hyp_params.proj_dim,
        )
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        self.prompt_length = hyp_params.prompt_length
        self.prompt_dim = hyp_params.prompt_dim
        self.llen, self.alen, self.vlen = hyp_params.seq_len
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = hyp_params.output_dim

        # Generative prompts for single-modality missing cases (3 possible single missing)
        generative_prompt = torch.zeros(3, self.prompt_dim, self.prompt_length)
        self.generative_prompt = nn.Parameter(generative_prompt)

        # Cross-modal diffusion layers (first-stage coarse estimates)
        self.l2a = DiffusionConditionalLayer(self.orig_d_l, self.prompt_dim, cond_dim=256)
        self.l2v = DiffusionConditionalLayer(self.orig_d_l, self.prompt_dim, cond_dim=256)
        self.v2a = DiffusionConditionalLayer(self.orig_d_v, self.prompt_dim, cond_dim=256)
        self.v2l = DiffusionConditionalLayer(self.orig_d_v, self.prompt_dim, cond_dim=256)
        self.a2v = DiffusionConditionalLayer(self.orig_d_a, self.prompt_dim, cond_dim=256)
        self.a2l = DiffusionConditionalLayer(self.orig_d_a, self.prompt_dim, cond_dim=256)

        # Second-stage diffusion layers for fine-grained completion.
        # Now src_dim = prompt_dim, tgt_dim = prompt_dim
        self.l_ap = DiffusionConditionalLayer(self.prompt_dim, self.prompt_dim, cond_dim=256)
        self.l_vp = DiffusionConditionalLayer(self.prompt_dim, self.prompt_dim, cond_dim=256)
        self.l_avp = DiffusionConditionalLayer(self.prompt_dim, self.prompt_dim, cond_dim=256)

        self.a_lp = DiffusionConditionalLayer(self.prompt_dim, self.prompt_dim, cond_dim=256)
        self.a_vp = DiffusionConditionalLayer(self.prompt_dim, self.prompt_dim, cond_dim=256)
        self.a_lvp = DiffusionConditionalLayer(self.prompt_dim, self.prompt_dim, cond_dim=256)

        self.v_ap = DiffusionConditionalLayer(self.prompt_dim, self.prompt_dim, cond_dim=256)
        self.v_lp = DiffusionConditionalLayer(self.prompt_dim, self.prompt_dim, cond_dim=256)
        self.v_alp = DiffusionConditionalLayer(self.prompt_dim, self.prompt_dim, cond_dim=256)

        # 1×1 Conv projections into unified d_l, d_a, d_v
        self.proj_l = nn.Conv1d(300, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(5, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(20, self.d_v, kernel_size=1, padding=0, bias=False)

        # Modality-specific prompts (signal when modality is present or missing)
        self.promptl_m = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_m = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_m = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))
        self.promptl_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))

        # Missing-type prompts (3 single-missing types)
        self.missing_type_prompt = nn.Parameter(
            torch.zeros(3, 50, 60)
        )
        self.m_a = nn.Parameter(torch.zeros(self.prompt_dim, 2 * self.prompt_dim))
        self.m_v = nn.Parameter(torch.zeros(self.prompt_dim, 2 * self.prompt_dim))
        self.m_l = nn.Parameter(torch.zeros(self.prompt_dim, 2 * self.prompt_dim))

        # Crossmodal Attention layers
        self.trans_l_with_a = self.get_network(self_type="la")
        self.trans_l_with_v = self.get_network(self_type="lv")
        self.trans_a_with_l = self.get_network(self_type="al")
        self.trans_a_with_v = self.get_network(self_type="av")
        self.trans_v_with_l = self.get_network(self_type="vl")
        self.trans_v_with_a = self.get_network(self_type="va")

        # Self-Attention memory networks
        self.trans_l_mem = self.get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)

        # Final fusion projection + residual
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        self.device = torch.device("cuda" if hyp_params.use_cuda else "cpu")
        self.to(self.device)

    def get_network(self, self_type, layers=-1):
        """
        Return a TransformerEncoder with appropriate dimensions.
        """
        if self_type == "la":
            embed_dim = self.d_l
            attn_dropout = self.attn_dropout
        elif self_type == "lv":
            embed_dim = self.d_l
            attn_dropout = self.attn_dropout
        elif self_type == "al":
            embed_dim = self.d_a
            attn_dropout = self.attn_dropout_a
        elif self_type == "av":
            embed_dim = self.d_a
            attn_dropout = self.attn_dropout_a
        elif self_type == "vl":
            embed_dim = self.d_v
            attn_dropout = self.attn_dropout_v
        elif self_type == "va":
            embed_dim = self.d_v
            attn_dropout = self.attn_dropout_v
        elif self_type == "l_mem":
            embed_dim = 2 * self.d_l
            attn_dropout = self.attn_dropout
        elif self_type == "a_mem":
            embed_dim = 2 * self.d_a
            attn_dropout = self.attn_dropout
        elif self_type == "v_mem":
            embed_dim = 2 * self.d_v
            attn_dropout = self.attn_dropout
        else:
            raise ValueError(f"Unknown network type: {self_type}")

        num_layers = layers if layers > 0 else self.layers
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, num_layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def get_complete_data(self, x_l, x_a, x_v, missing_mode):
        """
        x_l: [orig_d_l, seq_len_l]
        x_a: [orig_d_a, seq_len_a]
        x_v: [orig_d_v, seq_len_v]
        missing_mode: int in {0..6}

        返回:
          x_l: [1, d_l, seq_len_l]
          x_a: [1, d_a, seq_len_a]
          x_v: [1, d_v, seq_len_v]
        """
        # 扩展 batch 维度
        x_l = x_l.unsqueeze(0)  # [1, orig_d_l, seq_len_l]
        x_a = x_a.unsqueeze(0)  # [1, orig_d_a, seq_len_a]
        x_v = x_v.unsqueeze(0)  # [1, orig_d_v, seq_len_v]

        # 检查是否为空，如果为空则返回有效的张量
        if x_l is None or x_a is None or x_v is None:
            raise ValueError("One of the input modalities is None.")

        orig_l = x_l.shape[-1]
        orig_a = x_a.shape[-1]
        orig_v = x_v.shape[-1]

        max_len = max(orig_l, orig_a, orig_v)
        x_a = _align_prompt(x_a, max_len)  # 填充或截断音频模态
  # Fill or truncate audio modality
        # 0: 缺失文本 (使用音频+视觉生成文本)
        if missing_mode == 0:
            x_tgt_l = x_l if self.training else None

            v2l_output = self.v2l(x_v, x_tgt=x_tgt_l)  # [1, prompt_dim, T_pad_l]
            a2l_output = self.a2l(x_a, x_tgt=x_tgt_l)  # [1, prompt_dim, T_pad_l]

            cat_l = torch.cat(
                [self.generative_prompt[0], a2l_output[0], v2l_output[0]], dim=1
            ).unsqueeze(0)  # [1, prompt_dim, prompt_length + 2*T_pad_l]

            x_l_gen = self.l_avp(cat_l, x_tgt=x_tgt_l)  # [1, prompt_dim, dynamic_len]
            tgt_len = x_l_gen.shape[2]
            print(f"Shape of prompt before alignment: {self.promptl_m.shape}")
            aligned_prompt_l = _align_prompt(self.promptl_m, tgt_len).unsqueeze(0)
            print(f"Shape of aligned_prompt_l: {aligned_prompt_l.shape}")

            x_l_gen = x_l_gen + aligned_prompt_l  # [1, prompt_dim, dynamic_len]

            x_a_proj = self.proj_a(x_a)  # [1, d_a, orig_a]
            aligned_a_nm = _align_prompt(self.prompta_nm, orig_a).unsqueeze(0)
            x_a_proj = x_a_proj + aligned_a_nm  # [1, d_a, orig_a]

            x_v_proj = self.proj_v(x_v)  # [1, d_v, orig_v]
            aligned_v_nm = _align_prompt(self.promptv_nm, orig_v).unsqueeze(0)
            x_v_proj = x_v_proj + aligned_v_nm  # [1, d_v, orig_v]

            return x_l_gen, x_a_proj, x_v_proj

        # 1: 缺失音频 (使用文本+视觉生成音频)
        elif missing_mode == 1:
            x_tgt_a = x_a if self.training else None

            l2a_output = self.l2a(x_l, x_tgt=x_tgt_a)  # [1, prompt_dim, T_pad_a]
            v2a_output = self.v2a(x_v, x_tgt=x_tgt_a)  # [1, prompt_dim, T_pad_a]

            cat_a = torch.cat(
                [self.generative_prompt[1], l2a_output[0], v2a_output[0]], dim=1
            ).unsqueeze(0)
            x_a_gen = self.a_lvp(cat_a, x_tgt=x_tgt_a)  # [1, prompt_dim, dynamic_len]
            tgt_len = x_a_gen.shape[2]
            aligned_prompt_a = _align_prompt(self.prompta_m, tgt_len).unsqueeze(0)
            x_a_gen = x_a_gen + aligned_prompt_a  # [1, prompt_dim, dynamic_len]

            x_l_proj = self.proj_l(x_l)  # [1, d_l, orig_l]
            aligned_l_nm = _align_prompt(self.promptl_nm, orig_l).unsqueeze(0)
            x_l_proj = x_l_proj + aligned_l_nm  # [1, d_l, orig_l]

            x_v_proj = self.proj_v(x_v)  # [1, d_v, orig_v]
            aligned_v_nm = _align_prompt(self.promptv_nm, orig_v).unsqueeze(0)
            x_v_proj = x_v_proj + aligned_v_nm  # [1, d_v, orig_v]

            return x_l_proj, x_a_gen, x_v_proj

        # 2: 缺失视觉 (使用文本+音频生成视觉)
        elif missing_mode == 2:
            x_tgt_v = x_v if self.training else None

            l2v_output = self.l2v(x_l, x_tgt=x_tgt_v)  # [1, prompt_dim, T_pad_v]
            a2v_output = self.a2v(x_a, x_tgt=x_tgt_v)  # [1, prompt_dim, T_pad_v]

            cat_v = torch.cat(
                [self.generative_prompt[2], l2v_output[0], a2v_output[0]], dim=1
            ).unsqueeze(0)
            x_v_gen = self.v_alp(cat_v, x_tgt=x_tgt_v)  # [1, prompt_dim, dynamic_len]
            tgt_len = x_v_gen.shape[2]
            aligned_prompt_v = _align_prompt(self.promptv_m, tgt_len).unsqueeze(0)
            x_v_gen = x_v_gen + aligned_prompt_v  # [1, prompt_dim, dynamic_len]

            x_l_proj = self.proj_l(x_l)  # [1, d_l, orig_l]
            aligned_l_nm = _align_prompt(self.promptl_nm, orig_l).unsqueeze(0)
            x_l_proj = x_l_proj + aligned_l_nm  # [1, d_l, orig_l]

            x_a_proj = self.proj_a(x_a)  # [1, d_a, orig_a]
            aligned_a_nm = _align_prompt(self.prompta_nm, orig_a).unsqueeze(0)
            x_a_proj = x_a_proj + aligned_a_nm  # [1, d_a, orig_a]

            return x_l_proj, x_a_proj, x_v_gen

        # 3: 缺失文本+音频 (仅视觉存在)
        elif missing_mode == 3:
            x_tgt_l = x_l if self.training else None
            x_tgt_a = x_a if self.training else None

            v2l_output = self.v2l(x_v, x_tgt=x_tgt_l)  # [1, prompt_dim, T_pad_l]
            v2a_output = self.v2a(x_v, x_tgt=x_tgt_a)  # [1, prompt_dim, T_pad_a]

            cat_l = torch.cat([self.generative_prompt[0], v2l_output[0]], dim=1).unsqueeze(0)
            x_l_gen = self.l_vp(cat_l, x_tgt=x_tgt_l)  # [1, prompt_dim, dynamic_len]
            tgt_len = x_l_gen.shape[2]
            print(f"Shape of prompt before alignment: {self.promptl_m.shape}")
            aligned_prompt_l = _align_prompt(self.promptl_m, tgt_len).unsqueeze(0)
            print(f"Shape of aligned_prompt_l: {aligned_prompt_l.shape}")

            x_l_gen = x_l_gen + aligned_prompt_l  # [1, prompt_dim, dynamic_len]

            cat_a = torch.cat([self.generative_prompt[1], v2a_output[0]], dim=1).unsqueeze(0)
            x_a_gen = self.a_vp(cat_a, x_tgt=x_tgt_a)  # [1, prompt_dim, dynamic_len]
            tgt_len = x_a_gen.shape[2]
            aligned_prompt_a = _align_prompt(self.prompta_m, tgt_len).unsqueeze(0)
            x_a_gen = x_a_gen + aligned_prompt_a  # [1, prompt_dim, dynamic_len]

            x_v_proj = self.proj_v(x_v)  # [1, d_v, orig_v]
            aligned_v_nm = _align_prompt(self.promptv_nm, orig_v).unsqueeze(0)
            x_v_proj = x_v_proj + aligned_v_nm  # [1, d_v, orig_v]

            return x_l_gen, x_a_gen, x_v_proj

        # 4: 缺失文本+视觉 (仅音频存在)
        elif missing_mode == 4:
            x_tgt_l = x_l if self.training else None
            x_tgt_v = x_v if self.training else None

            a2l_output = self.a2l(x_a, x_tgt=x_tgt_l)  # [1, prompt_dim, T_pad_l]
            a2v_output = self.a2v(x_a, x_tgt=x_tgt_v)  # [1, prompt_dim, T_pad_v]

            cat_l = torch.cat([self.generative_prompt[0], a2l_output[0]], dim=1).unsqueeze(0)
            x_l_gen = self.l_avp(cat_l, x_tgt=x_tgt_l)  # [1, prompt_dim, dynamic_len]
            tgt_len = x_l_gen.shape[2]
            print(f"Shape of prompt before alignment: {self.promptl_m.shape}")
            aligned_prompt_l = _align_prompt(self.promptl_m, tgt_len).unsqueeze(0)
            print(f"Shape of aligned_prompt_l: {aligned_prompt_l.shape}")

            x_l_gen = x_l_gen + aligned_prompt_l  # [1, prompt_dim, dynamic_len]

            cat_v = torch.cat([self.generative_prompt[2], a2v_output[0]], dim=1).unsqueeze(0)
            x_v_gen = self.v_ap(cat_v, x_tgt=x_tgt_v)  # [1, prompt_dim, dynamic_len]
            tgt_len = x_v_gen.shape[2]
            aligned_prompt_v = _align_prompt(self.promptv_m, tgt_len).unsqueeze(0)
            x_v_gen = x_v_gen + aligned_prompt_v  # [1, prompt_dim, dynamic_len]

            x_a_proj = self.proj_a(x_a)  # [1, d_a, orig_a]
            aligned_a_nm = _align_prompt(self.prompta_nm, orig_a).unsqueeze(0)
            x_a_proj = x_a_proj + aligned_a_nm  # [1, d_a, orig_a]

            return x_l_gen, x_a_proj, x_v_gen

        # 5: 缺失音频+视觉 (仅文本存在)
        elif missing_mode == 5:
            x_tgt_a = x_a if self.training else None
            x_tgt_v = x_v if self.training else None

            l2a_output = self.l2a(x_l, x_tgt=x_tgt_a)  # [1, prompt_dim, T_pad_a]
            l2v_output = self.l2v(x_l, x_tgt=x_tgt_v)  # [1, prompt_dim, T_pad_v]

            cat_a = torch.cat([self.generative_prompt[1], l2a_output[0]], dim=1).unsqueeze(0)
            x_a_gen = self.a_lvp(cat_a, x_tgt=x_tgt_a)  # [1, prompt_dim, dynamic_len]
            tgt_len = x_a_gen.shape[2]
            aligned_prompt_a = _align_prompt(self.prompta_m, tgt_len).unsqueeze(0)
            x_a_gen = x_a_gen + aligned_prompt_a  # [1, prompt_dim, dynamic_len]

            cat_v = torch.cat([self.generative_prompt[2], l2v_output[0]], dim=1).unsqueeze(0)
            x_v_gen = self.v_lp(cat_v, x_tgt=x_tgt_v)  # [1, prompt_dim, dynamic_len]
            tgt_len = x_v_gen.shape[2]
            aligned_prompt_v = _align_prompt(self.promptv_m, tgt_len).unsqueeze(0)
            x_v_gen = x_v_gen + aligned_prompt_v  # [1, prompt_dim, dynamic_len]

            x_l_proj = self.proj_l(x_l)  # [1, d_l, orig_l]
            aligned_l_nm = _align_prompt(self.promptl_nm, orig_l).unsqueeze(0)
            x_l_proj = x_l_proj + aligned_l_nm  # [1, d_l, orig_l]

            return x_l_proj, x_a_gen, x_v_gen

        # 6: 全部存在，无缺失
        else:
            x_l_proj = self.proj_l(x_l)  # [1, d_l, orig_l]
            aligned_l_nm = _align_prompt(self.promptl_nm, orig_l).unsqueeze(0)
            x_l_proj = x_l_proj + aligned_l_nm  # [1, d_l, orig_l]

            x_a_proj = self.proj_a(x_a)  # [1, d_a, orig_a]
            aligned_a_nm = _align_prompt(self.prompta_nm, orig_a).unsqueeze(0)
            x_a_proj = x_a_proj + aligned_a_nm  # [1, d_a, orig_a]

            x_v_proj = self.proj_v(x_v)  # [1, d_v, orig_v]
            aligned_v_nm = _align_prompt(self.promptv_nm, orig_v).unsqueeze(0)
            x_v_proj = x_v_proj + aligned_v_nm  # [1, d_v, orig_v]

            return x_l_proj, x_a_proj, x_v_proj

    def forward(self, x_l, x_a, x_v, missing_mod):
        """
        x_l: [batch, seq_len_l, orig_d_l]
        x_a: [batch, seq_len_a, orig_d_a]
        x_v: [batch, seq_len_v, orig_d_v]
        missing_mod: [batch] integers in {0..6}
        """
        # Keep a copy of raw inputs for ground-truth
        raw_l, raw_a, raw_v = x_l.clone(), x_a.clone(), x_v.clone()

        # 1. Embedding dropout + transpose (to channels-first)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        print(f"Shape of x_l after dropout & transpose: {x_l.shape}")
        x_a = x_a.transpose(1, 2)
        print(f"Shape of x_a after transpose: {x_a.shape}")
        x_v = x_v.transpose(1, 2)
        print(f"Shape of x_v after transpose: {x_v.shape}")

        max_len = max(x_l.size(2), x_a.size(2), x_v.size(2))

        # 对每个模态进行填充，确保它们的序列长度一致
        x_l = F.pad(x_l, (0, max_len - x_l.size(2)), "constant", 0.0)
        x_a = F.pad(x_a, (0, max_len - x_a.size(2)), "constant", 0.0)
        x_v = F.pad(x_v, (0, max_len - x_v.size(2)), "constant", 0.0)
        print(f"Shape of x_l after padding: {x_l.shape}")
        print(f"Shape of x_a after padding: {x_a.shape}")
        print(f"Shape of x_v after padding: {x_v.shape}")

        xx_l, xx_a, xx_v = None, None, None
        # Loop over each sample in the batch for missing-data completion
        for idx in range(x_l.size(0)):
            # Build raw_*_s in [orig_dim, seq_len] from original ground truth
            raw_l_s = raw_l[idx].transpose(0, 1)  # [orig_d_l, seq_len_l]
            raw_a_s = raw_a[idx].transpose(0, 1)  # [orig_d_a, seq_len_a]
            raw_v_s = raw_v[idx].transpose(0, 1)  # [orig_d_v, seq_len_v]
            print(f"Shape of raw_l_s: {raw_l_s.shape}")
            print(f"Shape of raw_a_s: {raw_a_s.shape}")
            print(f"Shape of raw_v_s: {raw_v_s.shape}")
            missing_mod = missing_mod.view(-1)
            print(f"missing_mod 的形状是: {missing_mod.shape}")
            for idx in range(min(x_l.size(0), missing_mod.size(0))):  # 确保 idx 不会超出大小
                mode = missing_mod[idx].item()

                if xx_l is None:
                        xx_l = torch.zeros(1, self.d_l, raw_l_s.size(1))  # 初始化为零张量
                        xx_a = torch.zeros(1, self.d_a, raw_a_s.size(1))  # 初始化为零张量
                        xx_v = torch.zeros(1, self.d_v, raw_v_s.size(1))  # 初始化为零张量

                x_l_temp, x_a_temp, x_v_temp = self.get_complete_data(
                    raw_l_s, raw_a_s, raw_v_s, mode
                )
                print(f"Shape of x_l_temp: {x_l_temp.shape}")
                print(f"Shape of x_a_temp: {x_a_temp.shape}")
                print(f"Shape of x_v_temp: {x_v_temp.shape}")
                # Each is [1, d_*, seq_len_*]
            
                # 获取最大序列长度
                max_lenl = max(x_l_temp.size(2), xx_l.size(2))

                # 对x_l_temp和xx_l进行填充，使它们的序列长度一致
                x_l_temp = F.pad(x_l_temp, (0, max_lenl - x_l_temp.size(2)), "constant", 0.0)
                xx_l = F.pad(xx_l, (0, max_lenl - xx_l.size(2)), "constant", 0.0)

                # 对audio模态的填充
                max_lena = max(x_a_temp.size(2), xx_a.size(2))
                x_a_temp = F.pad(x_a_temp, (0, max_lena - x_a_temp.size(2)), "constant", 0.0)
                xx_a = F.pad(xx_a, (0, max_lena - xx_a.size(2)), "constant", 0.0)

                # 对visual模态的填充
                max_lenv = max(x_v_temp.size(2), xx_v.size(2))
                x_v_temp = F.pad(x_v_temp, (0, max_lenv - x_v_temp.size(2)), "constant", 0.0)
                xx_v = F.pad(xx_v, (0, max_lenv - xx_v.size(2)), "constant", 0.0)



                if xx_l is None:
                    xx_l, xx_a, xx_v = x_l_temp, x_a_temp, x_v_temp
                else:
                    xx_l = torch.cat([xx_l, x_l_temp], dim=0)
                    xx_a = torch.cat([xx_a, x_a_temp], dim=0)
                    xx_v = torch.cat([xx_v, x_v_temp], dim=0)

        # xx_l: [batch, d_l, seq_len_l]; xx_a: [batch, d_a, seq_len_a]; xx_v: [batch, d_v, seq_len_v]
        print(f"Shape of xx_l: {xx_l.shape}")
        print(f"Shape of xx_a: {xx_a.shape}")
        print(f"Shape of xx_v: {xx_v.shape}")

        # 2. Compute missing-type projection matrix mp (7 types)
        self.get_proj_matrix()
        batch_prompt = None
        for idx in range(xx_l.size(0)):
            mode = missing_mod[idx].item()
            single_proj = torch.matmul(
                self.missing_type_prompt,
                self.mp[mode]
            )  # [3, prompt_length, 2*prompt_dim]
            print(f"Shape of single_proj for mode {mode}: {single_proj.shape}")
            if batch_prompt is None:
                batch_prompt = single_proj.unsqueeze(0)  # [1, 3, prompt_length, 2*prompt_dim]
            else:
                batch_prompt = torch.cat([batch_prompt, single_proj.unsqueeze(0)], dim=0)
        # batch_prompt: [batch, 3, prompt_length, 2*prompt_dim]
        print(f"Shape of batch_prompt before transpose: {batch_prompt.shape}")
        batch_prompt = batch_prompt.transpose(0, 1)
        print(f"Shape of batch_prompt after transpose: {batch_prompt.shape}")
        # Now: batch_prompt[0] = text prompts [batch, prompt_length, 2*prompt_dim], etc.

        # 3. Prepare for Transformer: [seq_len, batch, d_*]
        proj_x_l = xx_l.permute(2, 0, 1)  # [seq_len_l, batch, d_l]
        print(f"Shape of proj_x_l: {proj_x_l.shape}")
        proj_x_a = xx_a.permute(2, 0, 1)  # [seq_len_a, batch, d_a]
        print(f"Shape of proj_x_a: {proj_x_a.shape}")
        proj_x_v = xx_v.permute(2, 0, 1)  # [seq_len_v, batch, d_v]
        print(f"Shape of proj_x_v: {proj_x_v.shape}")

        # Text branch
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        print(f"Shape of h_l_with_as: {h_l_with_as.shape}")
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        print(f"Shape of h_l_with_vs: {h_l_with_vs.shape}")
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)  # [seq_len_l, batch, 2*d_l]
        print(f"Shape of h_ls after cat (text): {h_ls.shape}")
        # Append text-specific prompts (prompt_length steps)
        text_prompts = batch_prompt[0].transpose(0, 1)  # [prompt_length, batch, 2*prompt_dim]
        print(f"Shape of text_prompts: {text_prompts.shape}")
        h_ls = torch.cat([h_ls, text_prompts], dim=0)
        print(f"Shape of h_ls after appending prompts: {h_ls.shape}")
        h_ls = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]  # [batch, 2*d_l]
        print(f"Shape of last_h_l: {last_h_l.shape}")

        # Audio branch
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        print(f"Shape of h_a_with_ls: {h_a_with_ls.shape}")
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        print(f"Shape of h_a_with_vs: {h_a_with_vs.shape}")
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)  # [seq_len_a, batch, 2*d_a]
        print(f"Shape of h_as after cat (audio): {h_as.shape}")
        audio_prompts = batch_prompt[1].transpose(0, 1)  # [prompt_length, batch, 2*prompt_dim]
        print(f"Shape of audio_prompts: {audio_prompts.shape}")
        h_as = torch.cat([h_as, audio_prompts], dim=0)
        print(f"Shape of h_as after appending prompts: {h_as.shape}")
        h_as = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        last_h_a = h_as[-1]  # [batch, 2*d_a]
        print(f"Shape of last_h_a: {last_h_a.shape}")

        # Visual branch
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        print(f"Shape of h_v_with_ls: {h_v_with_ls.shape}")
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        print(f"Shape of h_v_with_as: {h_v_with_as.shape}")
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)  # [seq_len_v, batch, 2*d_v]
        print(f"Shape of h_vs after cat (visual): {h_vs.shape}")
        visual_prompts = batch_prompt[2].transpose(0, 1)  # [prompt_length, batch, 2*prompt_dim]
        print(f"Shape of visual_prompts: {visual_prompts.shape}")
        h_vs = torch.cat([h_vs, visual_prompts], dim=0)
        print(f"Shape of h_vs after appending prompts: {h_vs.shape}")
        h_vs = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]  # [batch, 2*d_v]
        print(f"Shape of last_h_v: {last_h_v.shape}")

        # Concatenate all three modalities
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)  # [batch, combined_dim]
        print(f"Shape of last_hs after concatenation: {last_hs.shape}")

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training)
        )
        print(f"Shape of last_hs_proj before residual add: {last_hs_proj.shape}")
        last_hs_proj = last_hs_proj + last_hs
        print(f"Shape of last_hs_proj after residual add: {last_hs_proj.shape}")
        output = self.out_layer(last_hs_proj)  # [batch, output_dim]
        print(f"Shape of output: {output.shape}")
        return output

    def get_proj_matrix(self):
        """
        构建 self.mp，形状 [7, 2*prompt_dim, prompt_length]
        按照缺失模式 0..6 分别对应下面的组合：
        0: 文本缺失 (a, v 存在)       → a_v_lm
        1: 音频缺失 (a 缺失, v, l 存在) → am_v_l
        2: 视觉缺失 (v 缺失, a, l 存在) → a_vm_l
        3: 文本+音频缺失 (只剩 v)       → am_v_lm
        4: 文本+视觉缺失 (只剩 a)       → a_vm_lm
        5: 音频+视觉缺失 (只剩 l)       → am_vm_l
        6: 无缺失 (a, v, l 都存在)     → a_v_l
        """
        # Mode 0: 文本缺失 → a, v 存在，l 缺失
        a_v_lm = (
            self.prompta_nm @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_m  @ self.m_l
        ).unsqueeze(0)

        # Mode 1: 音频缺失 → a 缺失，v, l 存在
        am_v_l = (
            self.prompta_m  @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(0)

        # Mode 2: 视觉缺失 → v 缺失，a, l 存在
        a_vm_l = (
            self.prompta_nm @ self.m_a
            + self.promptv_m  @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(0)

        # Mode 3: 文本+音频缺失 → 只剩 v
        am_v_lm = (
            self.prompta_m  @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_m  @ self.m_l
        ).unsqueeze(0)

        # Mode 4: 文本+视觉缺失 → 只剩 a
        a_vm_lm = (
            self.prompta_nm @ self.m_a
            + self.promptv_m  @ self.m_v
            + self.promptl_m  @ self.m_l
        ).unsqueeze(0)

        # Mode 5: 音频+视觉缺失 → 只剩 l
        am_vm_l = (
            self.prompta_m  @ self.m_a
            + self.promptv_m  @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(0)

        # Mode 6: 无缺失 → a, v, l 都存在
        a_v_l = (
            self.prompta_nm @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(0)

        # 按照模式 0..6 的顺序拼接
        self.mp = torch.cat(
            [a_v_lm, am_v_l, a_vm_l, am_v_lm, a_vm_lm, am_vm_l, a_v_l],
            dim=0
        )  # 结果 [7, 2*prompt_dim, prompt_length]


        # mp = []
        # # Mode 0: text missing → only audio+visual present
        # m0 = torch.matmul(self.promptl_nm.t(), self.m_l).t()
        # print(f"Shape of m0: {m0.shape}")
        # mp.append(m0)

        # # Mode 1: audio missing
        # m1 = torch.matmul(self.prompta_nm.t(), self.m_a).t()
        # print(f"Shape of m1: {m1.shape}")
        # mp.append(m1)

        # # Mode 2: visual missing
        # m2 = torch.matmul(self.promptv_nm.t(), self.m_v).t()
        # print(f"Shape of m2: {m2.shape}")
        # mp.append(m2)

        # # Mode 3: text+audio missing → only visual present
        # m3 = torch.matmul(self.promptl_m.t(), self.m_l).t()
        # print(f"Shape of m3 step1: {m3.shape}")
        # m3 = torch.matmul(self.prompta_m.t(), self.m_a).t()
        # print(f"Shape of m3 after step2: {m3.shape}")
        # mp.append(m3)

        # # Mode 4: text+visual missing → only audio present
        # m4 = torch.matmul(self.promptl_m.t(), self.m_l).t()
        # print(f"Shape of m4 step1: {m4.shape}")
        # m4 = torch.matmul(self.promptv_m.t(), self.m_v).t()
        # print(f"Shape of m4 after step2: {m4.shape}")
        # mp.append(m4)

        # # Mode 5: audio+visual missing → only text present
        # m5 = torch.matmul(self.prompta_m.t(), self.m_a).t()
        # print(f"Shape of m5 step1: {m5.shape}")
        # m5 = torch.matmul(self.promptv_m.t(), self.m_v).t()
        # print(f"Shape of m5 after step2: {m5.shape}")
        # mp.append(m5)

        # # Mode 6: no missing (all present)
        # # 修改 m6 的形状，使其与 m0 到 m5 保持一致
        # # 假设 m6 的目标形状应该为 [60, 50]
        # m6 = torch.zeros(60, 50, device=self.promptl_m.device)  # 使用零初始化，形状为 [60, 50]c
        # print(f"Shape of m6: {m6.shape}")
        # mp.append(m6)

        # self.mp = torch.stack(mp, dim=0)  # [7, prompt_length, 2*prompt_dim]
        # print(f"Shape of self.mp: {self.mp.shape}")


class DiffusionConditionalLayer(nn.Module):
    def __init__(self, src_dim, tgt_dim, cond_dim=256, num_steps=50, is_train=True):
        super(DiffusionConditionalLayer, self).__init__()
        self.is_train = is_train
        self.cond_dim = cond_dim
        self.tgt_dim = tgt_dim
        self.num_steps = num_steps

        # Condition encoder: Conv1d(src_dim → cond_dim)
        self.cond_encoder = nn.Sequential(
            nn.Conv1d(src_dim, cond_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cond_dim, cond_dim, kernel_size=1),
        )

        # Noise scheduler (DDPM)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

        # Placeholder UNet; will rebuild dynamically based on input size
        self.denoiser = UNet1DModel(
            sample_size=64,
            in_channels=1,
            out_channels=tgt_dim,
            layers_per_block=2,
            block_out_channels=(128, 256, 512),
            down_block_types=("DownBlock1D",) * 3,
            up_block_types=("UpBlock1D",) * 3,
        )

    def forward(self, x_cond, x_tgt=None):
        """
        x_cond: [B, src_dim, T_cond]
        x_tgt: [B, tgt_dim, T_tgt] or None
        Returns:
          If training & x_tgt given: returns pred_noise [B, tgt_dim, T_pad]
          If inference (x_tgt None): returns generated [B, tgt_dim, T_pad] via DDIM sample
        """
        B, C_cond, T_cond = x_cond.shape
        print(f"Shape of x_cond: {x_cond.shape}")
        T_tgt = x_tgt.shape[-1] if x_tgt is not None else 0
        print(f"Shape of x_tgt: {x_tgt.shape if x_tgt is not None else None}")

        # We want both x_cond and x_tgt (if provided) to be padded to the same length
        min_len = 64
        T_target = max(min_len, T_cond, T_tgt)
        print(f"T_target: {T_target}")

        # Pad x_cond to length T_target
        pad_cond = T_target - T_cond
        if pad_cond > 0:
            x_cond = F.pad(x_cond, (0, pad_cond), "constant", 0.0)
        print(f"Shape of x_cond after padding: {x_cond.shape}")

        # Pad x_tgt to length T_target (if provided)
        if x_tgt is not None:
            pad_tgt = T_target - T_tgt
            if pad_tgt > 0:
                x_tgt = F.pad(x_tgt, (0, pad_tgt), "constant", 0.0)
            print(f"Shape of x_tgt after padding: {x_tgt.shape}")

        # Now both have length T_target
        cond = self.cond_encoder(x_cond)  # [B, cond_dim, T_target]
        print(f"Shape of cond after cond_encoder: {cond.shape}")

        if self.training and self.is_train:
            if x_tgt is None:
                return self.ddim_sample(cond)
            noise = torch.randn_like(x_tgt)  # [B, tgt_dim, T_target]
            print(f"Shape of noise: {noise.shape}")
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=x_cond.device,
            ).long()
            print(f"Timesteps: {timesteps}")
            noisy = self.noise_scheduler.add_noise(x_tgt, noise, timesteps)  # [B, tgt_dim, T_target]
            print(f"Shape of noisy: {noisy.shape}")

            expected_in = noisy.shape[1] + cond.shape[1]  # noisy_ch + cond_dim
            if (
                self.denoiser.config.in_channels != expected_in
                or self.denoiser.config.sample_size != cond.shape[-1]
            ):
                # Rebuild UNet1DModel with correct in_channels & sample_size
                self.denoiser = UNet1DModel(
                    sample_size=cond.shape[-1],
                    in_channels=expected_in,
                    out_channels=self.tgt_dim,
                    layers_per_block=2,
                    block_out_channels=(128, 256, 512),
                    down_block_types=("DownBlock1D",) * 3,
                    up_block_types=("UpBlock1D",) * 3,
                ).to(x_cond.device)
                print(f"Rebuilt denoiser with in_channels={expected_in} and sample_size={cond.shape[-1]}")

            noisy_cond = torch.cat([noisy, cond], dim=1)  # [B, noisy_ch+cond_dim, T_target]
            print(f"Shape of noisy_cond: {noisy_cond.shape}")
            pred = self.denoiser(noisy_cond, timesteps).sample  # [B, tgt_dim, T_target]
            print(f"Shape of pred: {pred.shape}")
            return pred

        else:
            # Inference: DDIM sampling
            return self.ddim_sample(cond)

    def ddim_sample(self, cond):
        """
        Performs DDIM reverse diffusion from pure noise, conditioned on 'cond'.
        cond: [B, cond_dim, T]
        Returns: [B, tgt_dim, T]
        """
        B, _, T = cond.shape
        print(f"Shape of cond in ddim_sample: {cond.shape}")
        x_t = torch.randn((B, self.tgt_dim, T), device=cond.device)
        print(f"Initial x_t: {x_t.shape}")

        timesteps = self.noise_scheduler.timesteps
        for t in reversed(timesteps):
            expected_in = x_t.shape[1] + cond.shape[1]
            if (
                self.denoiser.config.in_channels != expected_in
                or self.denoiser.config.sample_size != cond.shape[-1]
            ):
                self.denoiser = UNet1DModel(
                    sample_size=cond.shape[-1],
                    in_channels=expected_in,
                    out_channels=self.tgt_dim,
                    layers_per_block=2,
                    block_out_channels=(128, 256, 512),
                    down_block_types=("DownBlock1D",) * 3,
                    up_block_types=("UpBlock1D",) * 3,
                ).to(cond.device)
                print(f"Rebuilt denoiser in ddim_sample with in_channels={expected_in} and sample_size={cond.shape[-1]}")

            noisy_cond = torch.cat([x_t, cond], dim=1)  # [B, tgt_dim+cond_dim, T]
            print(f"Shape of noisy_cond in ddim_sample: {noisy_cond.shape}")
            noise_pred = self.denoiser(noisy_cond, t).sample  # [B, tgt_dim, T]
            print(f"Shape of noise_pred: {noise_pred.shape}")
            step = self.noise_scheduler.step(noise_pred, t, x_t)
            x_t = step.prev_sample  # update to previous timestep
            print(f"Updated x_t in ddim_sample: {x_t.shape}")

        return x_t  # [B, tgt_dim, T]
