import torch
from torch import nn
import torch.nn.functional as F

from .BranchedERFNet import LocationEmbedding

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


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels):
        super(BasicBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        res = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return F.leaky_relu(x + res)


class SpatialAtt(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAtt, self).__init__()
        # self.att_fc = nn.Sequential(
        #     nn.Conv1d(in_channels, 1, 1)
        # )

    def forward(self, x):
        # att = self.att_fc(x)
        # att = x.mean(dim=1, keepdim=True)
        # att = F.softmax(att, dim=-1).expand(x.shape)
        # return (att * x).max(-1, keepdim=True)[0]
        # return x.sort(-1, descending=True)[0][:, :, :].mean(-1, keepdim=True)
        # x = x / x.norm(p=2, dim=1, keepdim=True)
        v, idx = x.max(-1, keepdim=True)
        # if self.training:
        #     fp = open('test.log', 'a')
        #     for i in range(x.shape[0]):
        #         fp.write('%d ' % torch.unique(idx[i]).shape[0])
        #     fp.write('\n')
        return v
        # idx = torch.randint(0, 10, (1,), device=x.device) if self.training else [0]
        # return x.sort(-1, descending=True)[0][:, :, idx]

class PoseNetFeatOffsetEmb(nn.Module):
    # bn with border
    def __init__(self, num_points=250, ic=7, border_points=200, border_ic=6, output_dim=64, category=False):
        super(PoseNetFeatOffsetEmb, self).__init__()
        self.category = category
        bc = 256
        self.borderConv = PointFeatFuse3P(ic=border_ic, oc=bc, num_points=border_points)

        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.res1 = BasicBlock1D(64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.res2 = BasicBlock1D(128)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3_bn = nn.BatchNorm1d(256)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_res1 = BasicBlock1D(64)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_res2 = BasicBlock1D(128)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)
        self.e_conv1_bn = nn.BatchNorm1d(64)
        self.e_conv2_bn = nn.BatchNorm1d(128)
        self.e_conv3_bn = nn.BatchNorm1d(256)

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
            # nn.Linear(704+bc-256, 256),
            # nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
        self.ap1 = torch.nn.AvgPool1d(num_points)
        # self.mp2 = torch.nn.MaxPool1d(num_points)
        self.mp2 = SpatialAtt(64)
        self.num_points = num_points

    def forward(self, inp, borders, spatialEmbs, with_weight=False):
        x, emb = inp[:, -2:], inp[:, :-2]
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        x = self.res1(x)
        emb = F.leaky_relu(self.e_conv1_bn(self.e_conv1(emb)))
        emb = self.e_res1(emb)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = self.res2(x)
        emb = F.leaky_relu(self.e_conv2_bn(self.e_conv2(emb)))
        emb = self.e_res2(emb)

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))          # B,256,N
        emb = F.leaky_relu(self.e_conv3_bn(self.e_conv3(emb)))  # B,256,N

        pointfeat_2 = torch.cat((x, emb), dim=1)

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
        # total_points = weight.shape[-1]
        # k = int(total_points / 1)
        # topk_idx = weight.argsort(2, descending=True)[:, :, :k]
        # topk_att = weight.gather(2, topk_idx)
        # topk_att = topk_att / topk_att.sum(-1, keepdim=True)
        # weight_x3 = pointfeat_2.gather(2, topk_idx.repeat_interleave(pointfeat_2.shape[1], 1)).squeeze(-1)
        # weight_x3 = (weight_x3 * topk_att.expand_as(weight_x3)).sum(2)

        if with_weight:
            border_feat, bg_inds = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:], withInd=with_weight)
            x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
            outp = self.last_emb(x)
            return outp, weight, bg_inds
        else:
            border_feat = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:])

        x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
        # x = torch.cat([ap_x1, mp_x2, weight_x3, spatialEmbs], dim=1)
        # x = border_feat
        outp = self.last_emb(x)
        # return outp
        return outp

class TrackerOffsetEmb(nn.Module):
    # for uv offset and category
    def __init__(self, margin=0.3, num_points=250, border_ic=6, env_points=200, category=False, outputD=64, v23=False, scsn_layers=3):
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
        loss = torch.zeros([1]).cuda()
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

    def calc_mp_regularzation(self):
        w = self.point_feat.conv9.weight[:, :, 0]
        dot_prod = w.matmul(w.T)
        len_w = torch.sqrt(torch.pow(w, 2).sum(dim=1, keepdim=True)).expand(w.shape[0], w.shape[0])
        len_w = len_w.clamp(min=1e-12)
        cos_sim = dot_prod / (len_w * len_w.T)
        # mean_cos_sim = (cos_sim.sum() - torch.trace(cos_sim)) / (cos_sim.shape[0]**2 - cos_sim.shape[0])
        max_cos_sim = torch.abs(cos_sim - torch.diag(torch.diagonal(cos_sim))).max()
        # print(w.shape)
        # return torch.abs(mean_cos_sim)
        return max_cos_sim

    def forward(self, points, labels, xyxys, infer=False, visualize=False):
        points, xyxys = points[0], xyxys[0]
        embeds = self.embedding(xyxys)
        envs = points[:,self.num_points:]
        points = points[:,:self.num_points, :5]
        if infer:
            return self.inference(points, envs, embeds)
        elif visualize:
            embeds, point_weights, bg_inds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds, with_weight=True)
            return embeds, point_weights, bg_inds
        else:
            embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
            labels = labels[0]
            l_trip = self.compute_triplet_loss(embeds, labels)
            # l_reg = self.calc_mp_regularzation()
            # fp = open('test.log', 'a')
            # fp.write(str(l_reg.item()))
            # fp.write('\n')
            return l_trip# + l_reg

    def inference(self, points, envs, embeds):
        # assert points.shape[0] == 1
        embeds = self.point_feat(points.transpose(2, 1).contiguous(), envs.transpose(2, 1).contiguous(), embeds)
        return embeds
