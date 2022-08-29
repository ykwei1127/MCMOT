import torch
import numpy as np
import motmetrics
from torch.nn import functional as F
from torchvision.ops import box_convert


def fix_invalid_bbox(bbox):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    w = max(w, 1e-3)
    h = max(h, 1e-3)
    return torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).float()


def cosine_similarity(embsA, embsB):
    embsA = F.normalize(embsA, p=2, dim=1)
    embsB = F.normalize(embsB, p=2, dim=1)
    return embsA @ embsB.t()

def camera_compensate(H, boxes):
    if H is None:
        return boxes

    # Homography Transform
    N = len(boxes)
    homo_boxes = torch.zeros(3, 2 * N).float()
    homo_boxes[:2, :] = boxes.reshape(2 * N, 2).t()
    homo_boxes[2, :] = 1.0
    homo_result = H @ homo_boxes
    result = homo_result[:2, :] / (homo_result[2, :] + 1e-12)
    result = result.t().reshape(N, 4)

    # Fix degenerated bbox
    result = box_convert(result, 'xyxy', 'cxcywh')
    result[:, 2:].clamp_(min=1e-3)
    result = box_convert(result, 'cxcywh', 'xyxy')

    return result


def random_subset(dataset, k: int = 64):
    indices = torch.randperm(len(dataset))[:k].tolist()
    return torch.utils.data.Subset(dataset, indices)


class WarmupMultiStepScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, milestones=[], gamma=0.1):
        self.num_warmup_steps = num_warmup_steps
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.num_warmup_steps))
        milesteons_passed = [m for m in self.milestones if current_step >= m]
        return self.gamma ** len(milesteons_passed)


class MOTEvaluator:
    def __init__(
        self,
        target_classes=[1],
        distractor_classes=[2, 7, 8, 12],
        remove_distractor=True,
    ):
        self.accs = []
        self.names = []
        self.target_classes = target_classes
        self.distractor_classes = distractor_classes
        self.remove_distractor = remove_distractor

    def iou_dists_per_frame(self, df_pred, df_true):
        pred_data = {fid: group for fid, group in df_pred.groupby('fid')}
        true_data = {fid: group for fid, group in df_true.groupby('fid')}

        # Remove entries that match a distractor
        for fid, group in true_data.items():
            distractor_mask = group['class'].isin(self.distractor_classes)
            distractor_boxes = group[distractor_mask][['x', 'y', 'w', 'h']].values
            true_data[fid] = group[group['class'].isin(self.target_classes)]

            if (
                self.remove_distractor
                and len(distractor_boxes) > 0
                and fid in pred_data
            ):
                pred_boxes = pred_data[fid][['x', 'y', 'w', 'h']].values
                iou = 1 - motmetrics.distances.iou_matrix(pred_boxes, distractor_boxes)
                remove_mask = iou.max(axis=1) >= 0.5
                pred_data[fid] = pred_data[fid][~remove_mask]

        acc = motmetrics.MOTAccumulator()
        for fid in set(pred_data.keys()) | set(true_data.keys()):
            true_tags = np.empty(0)
            true_boxes = np.empty((0, 4))
            if fid in true_data:
                group = true_data[fid]
                true_tags = group['tag'].values
                true_boxes = group[['x', 'y', 'w', 'h']].values

            pred_tags = np.empty(0)
            pred_boxes = np.empty((0, 4))
            if fid in pred_data:
                group = pred_data[fid]
                pred_tags = group['tag'].values
                pred_boxes = group[['x', 'y', 'w', 'h']].values

            dist = motmetrics.distances.iou_matrix(true_boxes, pred_boxes, 0.5)
            acc.update(true_tags, pred_tags, dist, frameid=fid)

        return acc

    def add(self, name, df_pred, df_true):
        acc = self.iou_dists_per_frame(df_pred, df_true)
        self.accs.append(acc)
        self.names.append(name)
        return acc

    def evaluate(self, use_percentage=True):
        metrics = motmetrics.metrics.create()
        fmt = metrics.formatters
        summary = metrics.compute_many(
            self.accs,
            metrics=[
                'idf1',
                'recall',
                'precision',
                'num_unique_objects',
                'mostly_tracked',
                'partially_tracked',
                'mostly_lost',
                'num_false_positives',
                'num_misses',
                'num_switches',
                'num_fragmentations',
                'mota',
                'motp',
                'num_objects',
                'num_predictions',
            ],
            generate_overall=True,
        )

        if use_percentage:
            div_dict = {
                'num_objects': [
                    'num_false_positives',
                    'num_misses',
                    'num_switches',
                    'num_fragmentations',
                ],
                'num_unique_objects': [
                    'mostly_tracked',
                    'partially_tracked',
                    'mostly_lost',
                ],
            }
            for divisor in div_dict:
                for divided in div_dict[divisor]:
                    summary[divided] = summary[divided] / summary[divisor]

            change_fmt_list = [
                'num_false_positives',
                'num_misses',
                'num_switches',
                'num_fragmentations',
                'mostly_tracked',
                'partially_tracked',
                'mostly_lost',
            ]
            for k in change_fmt_list:
                fmt[k] = fmt['mota']

        summary.index = self.names + ['overall']
        text = motmetrics.io.render_summary(
            summary, formatters=fmt, namemap=motmetrics.io.motchallenge_metric_names
        )

        return summary, text
