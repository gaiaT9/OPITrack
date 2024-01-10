import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Any
import math

from .BranchedERFNet import LocationEmbedding

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def get_points_position_sub(pos):
    r"""
    get pairwise sub of every 2 points in pos
    pos: torch array with shape: N, n_points, 2
    return : torch array with shape N, n_points, n_points, 2
    """
    N, n, pos_dim = pos.shape
    pos1 = pos.unsqueeze(1).expand((N, n, n, pos_dim))
    pos2 = pos.unsqueeze(2).expand((N, n, n, pos_dim))
    pos = pos1 - pos2
    return pos

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PointTransformerSlim(nn.Module):
    def __init__(self, ninp, nhid):
        super().__init__()

        self.q_proj = nn.Linear(ninp, ninp, bias=False)
        self.k_proj = nn.Linear(ninp, ninp, bias=False)
        self.v_proj = nn.Linear(ninp, ninp, bias=False)

        self.qk_mlp = nn.Sequential(
            nn.Linear(ninp, ninp),
            # nn.BatchNorm1d(ninp),
            nn.ReLU(),
            nn.Linear(ninp, 1),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, ninp),
            # nn.BatchNorm1d(ninp),
            nn.ReLU(),
            nn.Linear(ninp, ninp),
        )

    def forward(self, x, pos):
        """
        x: N, npoints, C
        pos: N, npoints, npoints, 2
        """ 
        # N, npoints, ninp
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # N, npoints, npoints, ninp
        pos_emb = self.pos_mlp(pos)

        # N, npoints, npoints
        att = self.qk_mlp(q.unsqueeze(1) - k.unsqueeze(2) + pos_emb).squeeze(-1)
        att = F.softmax(att, -1)

        # N, npoints, ninp
        fuse_feature = torch.sum(att.unsqueeze(-1) * v.unsqueeze(1), 2)

        fuse_feature = x + fuse_feature

        return fuse_feature 

class TransformerEncoderLayerSlim(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayerSlim, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerSlim, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src

class PointPosEncoder(nn.Module):
    def __init__(self, ninp, dropout=0.1):
        super().__init__()

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, ninp),
            # nn.BatchNorm1d(ninp),
            nn.ReLU(),
            nn.Linear(ninp, ninp),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, pos):
        x = self.pos_mlp(pos) + x
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.pos_encoder = PointPosEncoder(ninp)
        encoder_layers = TransformerEncoderLayerSlim(ninp, nhead, nhid, dropout)
        # encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        # self.encoder_layers = PointTransformerSlim(ninp, nhid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        # self.encoder = nn.Linear(ntoken, ninp, bias=False)
        self.ninp = ninp
        # self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # nn.init.zeros_(self.decoder.weight)
        # nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True, pos=None):
        # N, C, n_points -> n_points, N, C
        src = src.permute((2, 0, 1))
        # N, C, n_points -> N, n_points, C
        # src = src.permute((0, 2, 1))
        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None

        # src = self.encoder(src) * math.sqrt(self.ninp)
        pos = pos.permute((2, 0, 1))
        src = self.pos_encoder(src, pos)

        output = self.transformer_encoder(src, self.src_mask)
        # output = self.encoder_layers(src, pos)

        # output = self.decoder(output)
        # return F.log_softmax(output, dim=-1)
        # n_points, N, C -> N, C, n_points
        output = output.permute((1, 2, 0))
        # N, n_points, C -> N, C, n_points
        # output = output.permute((0, 2, 1))
        return output

class PointFeatFuse3P(nn.Module):
    # three path
    def __init__(self, num_points=250, ic=7, oc=64, maxpool=True):
        super(PointFeatFuse3P, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)

        self.t_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.t_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.t_conv3 = torch.nn.Conv1d(128, 128, 1)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, oc, 1)

        if maxpool:
            self.pool = torch.nn.MaxPool1d(num_points)
        else:
            self.pool = torch.nn.AvgPool1d(num_points)

        self.num_points = num_points

    def forward(self, x, emb, t, withInd=False):
        x = F.leaky_relu(self.conv1(x))
        emb = F.leaky_relu(self.e_conv1(emb))
        t = F.leaky_relu(self.t_conv1(t))

        x = F.leaky_relu(self.conv2(x))
        emb = F.leaky_relu(self.e_conv2(emb))
        t = F.leaky_relu(self.t_conv2(t))

        x = F.leaky_relu(self.conv3(x))
        emb = F.leaky_relu(self.e_conv3(emb))
        t = F.leaky_relu(self.t_conv3(t))

        pointfeat_2 = torch.cat((x, emb, t), dim=1)

        x1 = F.leaky_relu(self.conv4(pointfeat_2))
        x1 = F.leaky_relu(self.conv5(x1))
        x1 = F.leaky_relu(self.conv6(x1))
        if withInd:
            return self.pool(x1).squeeze(-1), torch.max(x1, dim=2)[1]
        return self.pool(x1).squeeze(-1)


class PoseNetFeatOffsetEmb(nn.Module):
    # bn with border
    def __init__(self, num_points=250, ic=7, border_points=200, border_ic=6, output_dim=64, category=False):
        super(PoseNetFeatOffsetEmb, self).__init__()
        self.category = category
        bc = 256
        self.borderConv = PointFeatFuse3P(ic=border_ic, oc=bc, num_points=border_points)

        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        # self.trans1 = TransformerModel(64, 1, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        # self.trans2 = TransformerModel(128, 1, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        # self.trans3 = TransformerModel(256, 1, 256, 1)

        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3_bn = nn.BatchNorm1d(256)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        # self.e_trans1 = TransformerModel(64, 1, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        # self.e_trans2 = TransformerModel(128, 1, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)
        # self.e_trans3 = TransformerModel(256, 1, 256, 1)

        self.e_conv1_bn = nn.BatchNorm1d(64)
        self.e_conv2_bn = nn.BatchNorm1d(128)
        self.e_conv3_bn = nn.BatchNorm1d(256)

        self.fuse_trans = TransformerModel(512, 1, 1024, 1)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 64, 1)
        self.conv4_bn = nn.BatchNorm1d(256)
        self.conv5_bn = nn.BatchNorm1d(512)

        self.conv7 = torch.nn.Conv1d(512, 256, 1)
        self.conv8 = torch.nn.Conv1d(256, 512, 1)
        self.conv9 = torch.nn.Conv1d(512, 64, 1)
        self.conv7_bn = nn.BatchNorm1d(256)
        self.conv8_bn = nn.BatchNorm1d(512)

        self.conv_weight = torch.nn.Conv1d(128, 1, 1)

        self.last_emb = nn.Sequential(
            nn.Linear(704+bc, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.mp2 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

    def forward(self, inp, borders, spatialEmbs, with_weight=False):
        x, emb = inp[:, -2:], inp[:, :-2]
        pos = x
        # pos_sub = get_points_position_sub(x.permute((0, 2, 1)))
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        # x = self.trans1(x)
        emb = F.leaky_relu(self.e_conv1_bn(self.e_conv1(emb)))
        # emb = self.e_trans1(emb)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        # x = self.trans2(x)
        emb = F.leaky_relu(self.e_conv2_bn(self.e_conv2(emb)))
        # emb = self.e_trans2(x)

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))          # B,256,N
        # x = self.trans3(x)
        emb = F.leaky_relu(self.e_conv3_bn(self.e_conv3(emb)))  # B,256,N
        # emb = self.e_trans3(emb)

        pointfeat_2 = torch.cat((x, emb), dim=1)
        pointfeat_2 = self.fuse_trans(pointfeat_2, pos=pos)

        x1 = F.leaky_relu(self.conv4_bn(self.conv4(pointfeat_2)))
        x1 = F.leaky_relu(self.conv5_bn(self.conv5(x1)))
        x1 = F.leaky_relu(self.conv6(x1))                       # B,64,N
        ap_x1 = self.ap1(x1).squeeze(-1)                        # B,64

        x2 = F.leaky_relu(self.conv7_bn(self.conv7(pointfeat_2)))
        x2 = F.leaky_relu(self.conv8_bn(self.conv8(x2)))
        x2 = F.leaky_relu(self.conv9(x2))                       # B,64,N
        mp_x2 = self.mp2(x2).squeeze(-1)                        # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))   #B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        weight_x3 = (weight.expand_as(pointfeat_2) * pointfeat_2).sum(2)

        if with_weight:
            border_feat, bg_inds = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:], withInd=with_weight)
            x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
            outp = self.last_emb(x)
            return outp, weight, bg_inds
        else:
            border_feat = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:])

        x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
        outp = self.last_emb(x)
        return outp

class TrackerOffsetEmb(nn.Module):
    # for uv offset and category
    def __init__(self, margin=0.3, num_points=250, border_ic=6, env_points=200, category=False, outputD=64, v23=False):
        super().__init__()
        self.point_feat = PoseNetFeatOffsetEmb(num_points=num_points, ic=3, border_points=env_points, border_ic=border_ic, output_dim=outputD, category=True)
        self.num_points = num_points
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.embedding = LocationEmbedding

    def init_output(self, n_sigma=1):
        pass

    def compute_triplet_loss(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        loss = torch.zeros([1]).to(inputs.device)
        if mask.float().unique().shape[0] > 1:
            dist_ap, dist_an = [], []
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            # Compute ranking hinge loss
            # y = torch.ones_like(dist_an)
            y = (targets > 0).type(dist_an.dtype)
            loss = self.ranking_loss(dist_an, dist_ap, y).unsqueeze(0)
        return loss

    def forward(self, points, labels, xyxys, infer=False, visualize=False):
        points, xyxys = points[0], xyxys[0]
        embeds = self.embedding(xyxys)
        envs = points[:,self.num_points:]
        points = points[:,:self.num_points, :5]
        if infer:
            return self.inference(points, envs, embeds, norm=False)
        elif visualize:
            embeds, point_weights, bg_inds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds, with_weight=True)
            return embeds, point_weights, bg_inds
        else:
            embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
            labels = labels[0]
            return self.compute_triplet_loss(embeds, labels)

    def inference(self, points, envs, embeds, norm=False):
        # assert points.shape[0] == 1   
        embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
        if norm:
            embeds = embeds / embeds.norm(2, 1).unsqueeze(1).expand_as(embeds)
        return embeds
