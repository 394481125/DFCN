import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import Res50Encoder


class FewShotSeg(nn.Module):

    def __init__(self, args):
        super().__init__()

        # Encoder
        self.encoder = Res50Encoder(replace_stride_with_dilation=[True, True, False],
                                    pretrained_weights="COCO")  # or "ImageNet"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.args = args
        self.scaler = 20.0
        self.criterion = nn.NLLLoss(ignore_index=255, weight=torch.FloatTensor([0.1, 1.0]).cuda())

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, opt, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors  (1, 3, 257, 257)
            qry_mask: label
                N x 2 x H x W, tensor
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W
        ## Feature Extracting With ResNet Backbone
        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])
        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  # t for support features
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        outputs_qry = []
        coarse_loss = torch.zeros(1).to(self.device)
        for epi in range(supp_bs):

            """
            supp_fts[[epi], way, shot]: (B, C, H, W) 
            """

            if supp_mask[[0], 0, 0].max() > 0.:

                spt_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                             for shot in range(self.n_shots)] for way in range(self.n_ways)]
                spt_fg_proto = self.getPrototype(spt_fts_)

                # CPG module *******************
                qry_pred = torch.stack(
                    [self.getPred(qry_fts[way], spt_fg_proto[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

                qry_pred_coarse = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

                if train:
                    log_qry_pred_coarse = torch.cat([1 - qry_pred_coarse, qry_pred_coarse], dim=1).log()

                    coarse_loss = self.criterion(log_qry_pred_coarse, qry_mask)

                pred = torch.stack(
                    [self.getPred(qry_fts[way], spt_fg_proto[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

                pred_up = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
                pred = torch.cat((1.0 - pred_up, pred_up), dim=1)
                outputs_qry.append(pred)


            else:
                ########################acquiesce prototypical network ################
                supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                              for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_prototypes = self.getPrototype(supp_fts_)  # the coarse foreground

                qry_pred = torch.stack(
                    [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
                ########################################################################

                # Combine predictions of different feature maps #
                qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)

                outputs_qry.append(preds)

        output_qry = torch.stack(outputs_qry, dim=1)
        output_qry = output_qry.view(-1, *output_qry.shape[2:])

        return output_qry, coarse_loss, None, None, None

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def get_fg(self, fts, mask):

        """
        :param fts: (1, C, H', W')
        :param mask: (1, H, W)
        :return:
        """

        mask = torch.round(mask)
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        mask = mask.unsqueeze(1).bool()
        result_list = []

        for batch_id in range(fts.shape[0]):
            tmp_tensor = fts[batch_id]
            tmp_mask = mask[batch_id]

            foreground_features = tmp_tensor[:, tmp_mask[0]]

            if foreground_features.shape[1] == 1:
                foreground_features = torch.cat((foreground_features, foreground_features), dim=1)

            result_list.append(foreground_features)

        foreground_features = torch.stack(result_list)

        return foreground_features





