import math
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.ops import (
    batched_nms,
    box_iou,
    box_convert,
    generalized_box_iou,
    sigmoid_focal_loss,
)

from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, ImageList
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer


KL_WEIGHT = 0.5
L1_WEIGHT = 5.0
GIOU_WEIGHT = 2.0
FOCAL_WEIGHT = 2.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


def apply_deltas(
    deltas,
    boxes,
    bbox_weights=(2.0, 2.0, 1.0, 1.0),
    scale_clamp=math.log(100000.0 / 16),
):
    """
    Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

    Args:
        deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
            deltas[i] represents k potentially different class-specific
            box transformations for the single box boxes[i].
        boxes (Tensor): boxes to transform, of shape (N, 4)
    """
    boxes = boxes.to(deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = bbox_weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=scale_clamp)
    dh = torch.clamp(dh, max=scale_clamp)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

    return pred_boxes


def min_cost_match(pred_logits, pred_boxes, gt_labels, gt_boxes):
    y = torch.sigmoid(pred_logits)
    neg_cost_cls = (1 - FOCAL_ALPHA) * (y) ** FOCAL_GAMMA * (-(1 - y + 1e-8).log())
    pos_cost_cls = FOCAL_ALPHA * ((1 - y) ** FOCAL_GAMMA) * (-(y + 1e-8).log())
    cost_ce = pos_cost_cls[:, gt_labels] - neg_cost_cls[:, gt_labels]
    cost_l1 = torch.cdist(pred_boxes, gt_boxes, p=1)
    cost_giou = 1 - generalized_box_iou(pred_boxes, gt_boxes)

    cost_total = cost_ce * FOCAL_WEIGHT + cost_l1 * L1_WEIGHT + cost_giou * GIOU_WEIGHT
    assert torch.isfinite(cost_total).all(), cost_total
    rr, cc = linear_sum_assignment(cost_total.detach().to('cpu').numpy())
    rr, cc = torch.from_numpy(rr).long(), torch.from_numpy(cc).long()
    return rr, cc, cost_ce, cost_l1, cost_giou


def detection_loss(pred_logits, pred_boxes, gt_labels, gt_boxes):
    '''
    Args:
        pred_logits: [#pred, #class]
        pred_boxes: (normalized) [#pred, 4]
        gt_labels: [#true,]
        gt_boxes: (normalized) [#true, 4]
    Return:
        loss_ce: sum of losses [scalar]
        loss_l1: sum of losses [scalar]
        loss_giou: sum of losses [scalar]
    '''
    rr, cc, cost_ce, cost_l1, cost_giou = min_cost_match(
        pred_logits, pred_boxes, gt_labels, gt_boxes
    )

    scores_true = torch.zeros_like(pred_logits)
    scores_true[rr, gt_labels[cc]] = 1.0
    loss_ce = sigmoid_focal_loss(
        pred_logits, scores_true, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='sum'
    )
    loss_l1 = cost_l1[rr, cc].sum()
    loss_giou = cost_giou[rr, cc].sum()

    loss_ce = loss_ce * FOCAL_WEIGHT
    loss_l1 = loss_l1 * L1_WEIGHT
    loss_giou = loss_giou * GIOU_WEIGHT

    return loss_ce, loss_l1, loss_giou


def kl_loss(anchors, pred_boxes, pred_alphas, true_boxes):
    whs = anchors[:, 2:] - anchors[:, :2]
    pred_deltas = (pred_boxes - anchors) / torch.cat([whs, whs], dim=1)
    true_deltas = (true_boxes - anchors) / torch.cat([whs, whs], dim=1)
    losses_l1 = F.smooth_l1_loss(pred_deltas, true_deltas, reduction='none')
    losses_kl = torch.exp(-pred_alphas) * losses_l1 + 0.5 * pred_alphas
    loss_kl = losses_kl.sum(dim=1).sum()
    return loss_kl * KL_WEIGHT


class DynamicConv(nn.Module):
    def __init__(self, roi_dim=256, roi_h=7, roi_w=7, dyn_dim=64):
        super().__init__()

        self.dynamic_layer = nn.Linear(roi_dim, 2 * roi_dim * dyn_dim)
        self.norm1 = nn.LayerNorm(dyn_dim)
        self.norm2 = nn.LayerNorm(roi_dim)
        self.norm3 = nn.LayerNorm(roi_dim)
        self.out_layer = nn.Linear(roi_dim * roi_h * roi_w, roi_dim)
        self.roi_dim = roi_dim
        self.dyn_dim = dyn_dim

    def forward(self, pro_feats, roi_feats):
        '''
        Args:
            pro_feats: [N, roi_dim]
            roi_feats: [N, roi_dim, roi_h, roi_w]
        Return:
            obj_feats: [N, roi_dim]
        '''

        params = self.dynamic_layer(pro_feats)

        params1 = params[:, : (self.roi_dim * self.dyn_dim)]
        params1 = params1.view(-1, self.roi_dim, self.dyn_dim)

        params2 = params[:, (self.roi_dim * self.dyn_dim) :]
        params2 = params2.view(-1, self.dyn_dim, self.roi_dim)

        roi_feats = roi_feats.flatten(start_dim=2)  # [N, roi_dim, roi_h * roi_w]
        roi_feats = roi_feats.permute(0, 2, 1)  # [N, roi_h * roi_w, roi_dim]

        # [N, roi_h * roi_w, roi_dim] @ [N, roi_dim, dyn_dim]
        obj_feats = F.relu(self.norm1(torch.bmm(roi_feats, params1)))
        # [N, roi_h * roi_w, dyn_dim] @ [N, dyn_dim, roi_dim]
        obj_feats = F.relu(self.norm2(torch.bmm(obj_feats, params2)))

        obj_feats = obj_feats.flatten(start_dim=1)  # [N, roi_h * roi_w * roi_dim]
        obj_feats = F.relu(self.norm3(self.out_layer(obj_feats)))  # [N, obj_dim]

        return obj_feats


class RCNNHead(nn.Module):
    def __init__(
        self,
        roi_dim=256,
        roi_h=7,
        roi_w=7,
        dyn_dim=64,
        num_classes=1,
        ffn_dim=2048,
        num_heads=8,
        num_cls_linear=1,
        num_reg_linear=3,
    ):
        super().__init__()

        self.roi_dim = roi_dim
        self.self_attn = nn.MultiheadAttention(roi_dim, num_heads)
        self.inst_interact = DynamicConv(roi_dim, roi_h, roi_w, dyn_dim)

        self.linear1 = nn.Linear(roi_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, roi_dim)
        self.norm1 = nn.LayerNorm(roi_dim)
        self.norm2 = nn.LayerNorm(roi_dim)
        self.norm3 = nn.LayerNorm(roi_dim)

        self.cls_module = []
        for _ in range(num_cls_linear):
            self.cls_module.append(nn.Linear(roi_dim, roi_dim, False))
            self.cls_module.append(nn.LayerNorm(roi_dim))
            self.cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(self.cls_module)
        self.class_logits = nn.Linear(roi_dim, num_classes)

        self.reg_module = []
        for _ in range(num_reg_linear):
            self.reg_module.append(nn.Linear(roi_dim, roi_dim, False))
            self.reg_module.append(nn.LayerNorm(roi_dim))
            self.reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(self.reg_module)
        self.bboxes_delta = nn.Linear(roi_dim, 4)

    def forward(self, fpn_feats, pro_boxes, pro_feats, pooler):
        '''
        Args:
            fpn_feats: List
            pro_boxes: [B, P, 4]
            pro_feats: [B, P, roi_dim]
            pooler:
        Return:
            logits: [B, P, C]
            boxes: [B, P, 4]
            obj_feats: [B, P, roi_dim]
        '''
        B, P, _ = pro_boxes.shape
        boxes = [Boxes(x) for x in pro_boxes]
        roi_feats = pooler(fpn_feats, boxes)  # [B * P, roi_dim, roi_h, roi_w]

        # self attention
        pro_feats = pro_feats.permute(1, 0, 2)  # [P, B, roi_dim]
        pro_feats = pro_feats + self.self_attn(pro_feats, pro_feats, pro_feats)[0]
        pro_feats = self.norm1(pro_feats)

        # inst interact
        pro_feats = pro_feats.permute(1, 0, 2)  # [B, P, roi_dim]
        pro_feats = pro_feats.reshape(B * P, -1)
        obj_feats = pro_feats + self.inst_interact(pro_feats, roi_feats)
        obj_feats = self.norm2(obj_feats)  # [B * P, roi_dim]

        # FFN
        obj_feats = obj_feats + self.linear2(F.relu(self.linear1(obj_feats)))
        obj_feats = self.norm3(obj_feats)  # [B * P, roi_dim]

        cls_feats = obj_feats
        for layer in self.cls_module:
            cls_feats = layer(cls_feats)
        cls_pred = self.class_logits(cls_feats)

        reg_feats = obj_feats
        for layer in self.reg_module:
            reg_feats = layer(reg_feats)
        reg_pred = self.bboxes_delta(reg_feats)

        pred_boxes = apply_deltas(reg_pred, pro_boxes.reshape(-1, 4))

        cls_pred = cls_pred.reshape(B, P, -1)
        pred_boxes = pred_boxes.reshape(B, P, -1)
        obj_feats = obj_feats.reshape(B, P, -1)

        return cls_pred, pred_boxes, obj_feats


class Model(pl.LightningModule):
    def __init__(self, num_proposals=300, num_classes=1, num_stages=6):
        super().__init__()

        cfg = model_zoo.get_config(
            'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml', trained=True
        )
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
        frcnn = build_model(cfg)
        DetectionCheckpointer(frcnn).load(cfg.MODEL.WEIGHTS)

        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.num_stages = num_stages

        self.backbone = frcnn.backbone.to(self.device)
        self.head_series = nn.ModuleList(
            [
                RCNNHead(
                    roi_dim=256,
                    roi_h=7,
                    roi_w=7,
                    dyn_dim=64,
                    ffn_dim=2048,
                    num_heads=8,
                    num_classes=1,
                    num_cls_linear=1,
                    num_reg_linear=3,
                )
                for _ in range(num_stages)
            ]
        )
        self.proposal_boxes = nn.Embedding(num_proposals, 4)
        self.proposal_feats = nn.Embedding(num_proposals, 256)
        nn.init.constant_(self.proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.proposal_boxes.weight[:, 2:], 1.0)

        self.fpn_keys = ['p2', 'p3', 'p4', 'p5']
        fpn_shape = self.backbone.output_shape()
        self.pooler = ROIPooler(
            output_size=(7, 7),
            scales=tuple(1.0 / fpn_shape[k].stride for k in self.fpn_keys),
            sampling_ratio=2,
            pooler_type='ROIAlignV2',
        )

        self.unc_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

        del frcnn

    def extract_fpn_feats(self, images):
        sizes = []
        for image in images:
            _, h, w = image.shape
            sizes.append([w, h, w, h])
        sizes = torch.tensor(sizes).float().to(self.device)
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        tensor = TF.normalize(
            images.tensor, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
        )
        tensor = tensor.to(self.device)
        fpn_feats = self.backbone(tensor)
        fpn_feats = [fpn_feats[k] for k in self.fpn_keys]
        return fpn_feats, sizes

    def detect(self, fpn_feats, sizes):
        N = len(sizes)
        P = self.num_proposals
        C = self.num_classes

        proposal_boxes = self.proposal_boxes.weight
        proposal_boxes = box_convert(proposal_boxes, 'cxcywh', 'xyxy')  # [P, 4]
        proposal_boxes = proposal_boxes.unsqueeze(dim=0).expand(N, P, 4)
        proposal_boxes = proposal_boxes * sizes.view(N, 1, 4)
        proposal_feats = self.proposal_feats.weight
        proposal_feats = proposal_feats.unsqueeze(dim=0).expand(N, P, 256)

        stages_logits = [torch.zeros(N, P, C, device=self.device)]
        stages_boxes = [proposal_boxes]
        stages_feats = [proposal_feats]
        for head in self.head_series:
            logits, boxes, feats = head(
                fpn_feats,
                stages_boxes[-1].detach(),
                stages_feats[-1],
                self.pooler,
            )
            stages_logits.append(logits)
            stages_boxes.append(boxes)
            stages_feats.append(feats)
        det_logits = torch.stack(stages_logits, dim=0)  # [S + 1, N, P, C]
        det_boxes = torch.stack(stages_boxes, dim=0)  # [S + 1, N, P, 4]
        det_feats = torch.stack(stages_feats, dim=0)  # [S + 1, N, P, 256]

        return det_logits, det_boxes, det_feats

    def postprocess(
        self,
        det_logits,
        det_boxes,
        det_sigmas,
        det_score_thresh=0.3,
        det_nms_thresh=0.7,
    ):
        results = []
        for i in range(det_logits.shape[1]):
            scores = torch.sigmoid(det_logits[-1, i]).to('cpu')  # [P, C]
            boxes = det_boxes[-1, i].to('cpu')  # [P, 4]
            sigmas = det_sigmas[i].to('cpu')

            max_res = scores.max(dim=1)
            labels = max_res.indices + 1
            scores = max_res.values

            keep = batched_nms(boxes, scores, labels, det_nms_thresh)
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            sigmas = sigmas[keep]

            mask = scores >= det_score_thresh
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]
            sigmas = sigmas[mask]

            results.append((scores, labels, boxes, sigmas))
        return results

    def training_step(self, inps, _):
        self.train()

        images = []
        gt_insts_b = []
        for inp in inps:
            images.append(inp['image'].float())
            gt_insts_b.append(inp['insts'].to(self.device))

        N = len(images)
        P = self.num_proposals
        S = self.num_stages

        fpn_feats, sizes = self.extract_fpn_feats(images)
        logits, boxes, feats = self.detect(fpn_feats, sizes)
        alphas = self.unc_head(feats[-1])

        total_gt_boxes = sum(len(insts) for insts in gt_insts_b)

        losses = {}
        for s in range(1, S + 1):
            losses_ce = []
            losses_l1 = []
            losses_giou = []
            for i in range(N):
                pred_boxes = boxes[s, i] / sizes[i]
                true_boxes = gt_insts_b[i].gt_boxes / sizes[i]
                loss_ce, loss_l1, loss_giou = detection_loss(
                    logits[s, i],
                    pred_boxes,
                    gt_insts_b[i].gt_classes,
                    true_boxes,
                )
                losses_ce.append(loss_ce)
                losses_l1.append(loss_l1)
                losses_giou.append(loss_giou)

                rr, cc, _, _, _ = min_cost_match(
                    logits[s, i],
                    pred_boxes,
                    gt_insts_b[i].gt_classes,
                    true_boxes,
                )
                iou = torch.diag(box_iou(pred_boxes[rr], true_boxes[cc])).mean()
                self.log(f'iou/{s}', iou)

            losses.update(
                {
                    f'loss_det_ce/{s}': sum(losses_ce) / total_gt_boxes,
                    f'loss_det_l1/{s}': sum(losses_l1) / total_gt_boxes,
                    f'loss_det_giou/{s}': sum(losses_giou) / total_gt_boxes,
                }
            )

        for i in range(N):
            losses_kl = []
            rr, cc, _, _, _ = min_cost_match(
                logits[-1, i],
                boxes[-1, i] / sizes[i],
                gt_insts_b[i].gt_classes,
                gt_insts_b[i].gt_boxes / sizes[i],
            )
            losses_kl.append(
                kl_loss(
                    boxes[-2, i][rr].detach(),
                    boxes[-1, i][rr],
                    alphas[i][rr],
                    gt_insts_b[i].gt_boxes[cc],
                )
            )
            self.log('alpha', alphas[i][rr].mean())
        losses['loss_kl'] = sum(losses_kl) / total_gt_boxes

        losses['loss_total'] = sum(v for v in losses.values())
        for k, v in losses.items():
            self.log(k, v)
        return losses['loss_total']
