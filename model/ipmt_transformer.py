import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ops.modules import MSDeformAttn
from model.positional_encoding import SinePositionalEncoding
from einops import rearrange

class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

class MyCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert num_heads==1, "currently only implement num_heads==1"
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)
        self.proj_drop  = nn.Dropout(proj_drop)

        self.drop_prob = 0.1


    def forward(self, q, k, v, supp_valid_mask=None, supp_mask=None):
        # B, N, C = q.shape

        q = self.q_fc(q)
    
        k = self.k_fc(k)
        v = self.v_fc(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale # [bs, n, n]

        if supp_mask is not None:
            supp_mask = (~supp_mask).unsqueeze(1).float()
            supp_mask = supp_mask * -10000.0
            attn = attn + supp_mask        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class QSCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert num_heads==1, "currently only implement num_heads==1"
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q1_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v1_fc = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop1  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)

        self.proj_drop  = nn.Dropout(proj_drop)

        self.drop_prob = 0.1


    def forward(self, prototype, q_x, qry_attn_mask ):

        q1 = self.q1_fc(prototype)
        k1 = self.k1_fc(q_x)
        v1 = self.v1_fc(q_x)
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale # [bs, n, n]

        if qry_attn_mask is not None:
            qry_attn_mask = (~qry_attn_mask).unsqueeze(1).float()
            qry_attn_mask = qry_attn_mask * -10000.0
            attn1 = attn1 + qry_attn_mask        
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop1(attn1)

        x1 = (attn1 @ v1)
        x = self.proj(x1)
        x = self.proj_drop(x) 
        return x


class CyCTransformer(nn.Module):
    def __init__(self,
                 embed_dims=384, 
                 num_heads=1,
                 su_num_layers = 3, 
                 num_layers=5,
                 num_levels=1,
                 num_points=9,
                 use_ffn=True,
                 dropout=0.1,
                 shot=1,

                 ):
        super(CyCTransformer, self).__init__()
        self.embed_dims             = embed_dims
        self.num_heads              = num_heads
        self.su_num_layers          = su_num_layers
        self.num_layers             = num_layers
        self.num_levels             = num_levels
        self.use_ffn                = use_ffn
        self.feedforward_channels   = embed_dims*3
        self.dropout                = dropout
        self.shot                   = shot
        self.use_self               = True
        

        self.cross_layers = []
        self.qry_self_layers  = []
        self.layer_norms = []
        self.ffns = []
        self.decoder_cross_attention_layers = []
        self.merge_conv = [] 
        self.update_q = []
        self.res_q = []
        for c_id in range(self.num_layers):
            self.cross_layers.append(
                        MyCrossAttention(embed_dims, attn_drop=self.dropout, proj_drop=self.dropout),
                    )
            self.layer_norms.append(nn.LayerNorm(embed_dims))
            self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
            self.layer_norms.append(nn.LayerNorm(embed_dims))

        for l_id in range(self.num_layers):

            self.layer_norms.append(nn.LayerNorm(embed_dims))
            if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))
            
            if self.use_self:
                self.qry_self_layers.append(
                    MSDeformAttn(embed_dims, num_levels, num_heads, num_points)
                )
                self.layer_norms.append(nn.LayerNorm(embed_dims))

                if self.use_ffn:
                    self.ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
                    self.layer_norms.append(nn.LayerNorm(embed_dims))

            self.decoder_cross_attention_layers.append(
                        QSCrossAttention(embed_dims, attn_drop=self.dropout, proj_drop=self.dropout),
                    )
        for l_id in range(self.num_layers-1):
            self.update_q.append(nn.Sequential(
                    nn.Conv2d(embed_dims*2, embed_dims, kernel_size=1, padding=0, bias=False),
                    nn.ReLU(inplace=True)
                    ))
            self.res_q.append(nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True)
                    ))


        if self.use_self:
            self.qry_self_layers  = nn.ModuleList(self.qry_self_layers)
        self.ffns         = nn.ModuleList(self.ffns)
        self.layer_norms  = nn.ModuleList(self.layer_norms)
        self.decoder_cross_attention_layers = nn.ModuleList(self.decoder_cross_attention_layers)
        self.cross_layers = nn.ModuleList(self.cross_layers)

        self.positional_encoding = SinePositionalEncoding(embed_dims//2, normalize=True) 
        self.level_embed = nn.Parameter(torch.rand(num_levels, embed_dims))
        nn.init.xavier_uniform_(self.level_embed)

        self.proj_drop  = nn.Dropout(dropout)
            
        self.mask_embed = FFN(embed_dims, embed_dims, dropout=self.dropout)
        self.decoder_norm = nn.LayerNorm(embed_dims)
        self.prototype = nn.Embedding(1, embed_dims)

        self.update_q         = nn.ModuleList(self.update_q)
        self.res_q         = nn.ModuleList(self.res_q)

    def init_weights(self, distribution='uniform'):
        """Initialize the transformer weights."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
        return reference_points

    def get_qry_flatten_input(self, x, qry_masks):
        src_flatten = [] 
        qry_valid_masks_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []        
        for lvl in range(self.num_levels):   
            src = x[lvl]
            bs, c, h, w = src.shape
            
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).permute(0, 2, 1) # [bs, c, h*w] -> [bs, h*w, c]
            src_flatten.append(src)

            if qry_masks is not None:
                qry_mask = qry_masks[lvl]
                qry_valid_mask = []
                qry_mask = F.interpolate(
                    qry_mask.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                for img_id in range(bs):
                    qry_valid_mask.append(qry_mask[img_id]==255)
                qry_valid_mask = torch.stack(qry_valid_mask, dim=0)
            else:
                qry_valid_mask = torch.zeros((bs, h, w))

            pos_embed = self.positional_encoding(qry_valid_mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            pos_embed_flatten.append(pos_embed)

            qry_valid_masks_flatten.append(qry_valid_mask.flatten(1))

        src_flatten = torch.cat(src_flatten, 1) # [bs, num_elem, c]
        qry_valid_masks_flatten = torch.cat(qry_valid_masks_flatten, dim=1) # [bs, num_elem]
        pos_embed_flatten = torch.cat(pos_embed_flatten, dim=1) # [bs, num_elem, c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # [num_lvl, 2]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [num_lvl]
        
        return src_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index

    def get_supp_flatten_input(self, s_x, supp_mask):
        s_x_flatten = []
        supp_valid_mask = []
        supp_obj_mask = []
        supp_mask = F.interpolate(supp_mask, size=s_x.shape[-2:], mode='nearest').squeeze(1) # [bs*shot, h, w]
        supp_mask = supp_mask.view(-1, self.shot, s_x.size(2), s_x.size(3))
        s_x = s_x.view(-1, self.shot, s_x.size(1), s_x.size(2), s_x.size(3))

        for st_id in range(s_x.size(1)):
            supp_valid_mask_s = []
            supp_obj_mask_s = []
            for img_id in range(s_x.size(0)):
                supp_valid_mask_s.append(supp_mask[img_id, st_id, ...]==255)
                obj_mask = supp_mask[img_id, st_id, ...]==1
                if obj_mask.sum() == 0: # To avoid NaN
                    obj_mask[obj_mask.size(0)//2-1:obj_mask.size(0)//2+1, obj_mask.size(1)//2-1:obj_mask.size(1)//2+1] = True
                if (obj_mask==False).sum() == 0: # To avoid NaN
                    obj_mask[0, 0]   = False
                    obj_mask[-1, -1] = False 
                    obj_mask[0, -1]  = False
                    obj_mask[-1, 0]  = False
                supp_obj_mask_s.append(obj_mask)
            supp_valid_mask_s = torch.stack(supp_valid_mask_s, dim=0) # [bs, h, w]
            supp_valid_mask_s = supp_valid_mask_s.flatten(1) # [bs, h*w]
            supp_valid_mask.append(supp_valid_mask_s)

            supp_obj_mask_s = torch.stack(supp_obj_mask_s, dim=0)
            supp_obj_mask_s = (supp_obj_mask_s==1).flatten(1) # [bs, n]
            supp_obj_mask.append(supp_obj_mask_s)

            s_x_s = s_x[:, st_id, ...]
            s_x_s = s_x_s.flatten(2).permute(0, 2, 1)  # [bs, c, h*w] -> [bs, h*w, c]
            s_x_flatten.append(s_x_s)

        s_x_flatten = torch.cat(s_x_flatten, 1) # [bs, h*w*shot, c]
        supp_valid_mask = torch.cat(supp_valid_mask, 1)
        supp_mask_flatten = torch.cat(supp_obj_mask, 1)

        return s_x_flatten, supp_valid_mask, supp_mask_flatten


    def forward(self, x, qry_masks, s_x, supp_mask, init_mask): 
        if not isinstance(x, list):
            x = [x]
        if not isinstance(qry_masks, list):
            qry_masks = [qry_masks.clone() for _ in range(self.num_levels)]

        assert len(x) == len(qry_masks) == self.num_levels
        bs, c = x[0].size()[:2]

        x_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index = self.get_qry_flatten_input(x, qry_masks)

        s_x, supp_valid_mask, supp_mask_flatten = self.get_supp_flatten_input(s_x, supp_mask.clone())

        reference_points = self.get_reference_points(spatial_shapes, device=x_flatten.device)

        ln_id = 0
        ffn_id = 0

        qry_outputs_mask_list = []
        sup_outputs_mask_list = []

        prototype = self.prototype.weight.unsqueeze(1).repeat(bs, 1, 1) 
        k = s_x
        v = k.clone()
        prototype_list = []
        for c_id in range(self.num_layers):
            cross_out = self.cross_layers[c_id](prototype, k, v, supp_valid_mask, supp_mask_flatten)
            prototype = cross_out + prototype
            prototype = self.layer_norms[ln_id](prototype)
            ln_id += 1

            prototype = self.ffns[ffn_id](prototype)
            ffn_id += 1
            prototype = self.layer_norms[ln_id](prototype)
            ln_id += 1
            prototype_list.append(prototype)
            qry_outputs_mask, _ = self.forward_prediction_heads(prototype, x_flatten)
            qry_outputs_mask_list.append(qry_outputs_mask)
        
        q = x_flatten
        pos = pos_embed_flatten
        qry_attn_mask = init_mask.flatten(1)

        for l_id in range(self.num_layers):
            if self.use_self:
                q =  q + self.proj_drop(self.qry_self_layers[l_id](q + pos, reference_points, q, spatial_shapes, level_start_index, qry_valid_masks_flatten))
                q = self.layer_norms[ln_id](q)
                ln_id += 1
       
                if self.use_ffn:
                    q = self.ffns[ffn_id](q)
                    ffn_id += 1
                    q = self.layer_norms[ln_id](q)
                    ln_id += 1

            # attention: cross-attention first
            qry_prototype = self.decoder_cross_attention_layers[l_id](
                prototype_list[l_id], q, qry_attn_mask ) 

            prototype = qry_prototype + prototype_list[l_id]

            qry_outputs_mask, qry_attn_mask = self.forward_prediction_heads(prototype, q)
            sup_outputs_mask, _ = self.forward_prediction_heads(prototype, s_x)

            qry_outputs_mask_list.append(qry_outputs_mask)
            sup_outputs_mask_list.append(sup_outputs_mask)

            ## update query feature map
            if l_id < self.num_layers-1:
                tmp_prototype = prototype.expand_as(q)
                tmp_q = rearrange(torch.cat((q,tmp_prototype),dim=-1), 'b (h w) c -> b c h w',h=60,w=60)
                q = self.update_q[l_id](tmp_q)
                q = q + self.res_q[l_id](q)
                q = rearrange(q, 'b c h w -> b (h w) c')
            
        out = qry_outputs_mask_list[-1].clone()
        return out,qry_outputs_mask_list,sup_outputs_mask_list

    def forward_prediction_heads(self, output, mask_features):

        decoder_output = self.decoder_norm(output) #LayerNorm
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bkc->bqk", mask_embed, mask_features) 
        outputs_mask = rearrange(outputs_mask,'b q (h w) -> b q h w', w=60,h=60)
        attn_mask = (outputs_mask.sigmoid().flatten(1) >= 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_mask, attn_mask
