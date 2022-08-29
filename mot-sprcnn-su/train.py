import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import  DataLoader, ConcatDataset
from torchvision.ops import box_convert

from detectron2.utils.colormap import random_color
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation.coco_evaluation import COCOeval_opt
from pycocotools.coco import COCO

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.plugins import DDPPlugin

from model import Model
from dataset import TrainData
from utils import random_subset, WarmupMultiStepScheduler




class Visualization(callbacks.Callback):
    def __init__(self, loader, interval=1):
        super().__init__()
        self.loader = loader
        self.interval = interval

    @pl.utilities.rank_zero_only
    def on_train_epoch_end(self, trainer, model: Model):
        if (trainer.current_epoch + 1) % self.interval != 0:
            return

        log_dir = Path(trainer.logger.log_dir)
        vis_dir = log_dir / 'visualize' / f'{trainer.current_epoch:03d}'
        vis_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        model.eval()
        for inps in tqdm(iter(self.loader), desc='Visualize', leave=False):
            with torch.no_grad():
                images = []
                for inp in inps:
                    images.append(inp['image'].float())
                N = len(images)
                P = model.num_proposals
                fpn_feats, sizes = model.extract_fpn_feats(images)
                logits, boxes, feats = model.detect(fpn_feats, sizes)
                alphas = model.unc_head(feats[-1])
                anchors = boxes[-2].detach()
                whs = anchors[..., 2:] - anchors[..., :2]  # [N, P, 2]
                sigmas = torch.exp(0.5 * alphas)  # [N, P, 4]
                sigmas = sigmas * torch.cat([whs, whs], dim=2)

            for i in range(N):
                vis_image = images[i].permute(1, 2, 0).numpy()
                H, W, _ = vis_image.shape

                colors = np.stack([random_color(True, 1) for _ in range(P)])
                pos_mask = torch.sigmoid(logits[-1, i, :, 0]) >= 0.4
                for s in range(len(boxes)):
                    scores = torch.sigmoid(logits[s, i][pos_mask][..., 0]).to('cpu')
                    vis = Visualizer(vis_image)
                    vis.overlay_instances(
                        boxes=boxes[s, i][pos_mask].to('cpu'),
                        labels=[f'{x:.0%}' for x in scores.tolist()],
                        assigned_colors=colors[pos_mask.to('cpu').numpy()],
                    )
                    vis.output.save(vis_dir / f'{idx:03d}.det{s}.jpg')

                vis = Visualizer(vis_image)
                vis.overlay_instances(boxes=inps[i]['insts'].gt_boxes.to('cpu'))
                vis.output.save(vis_dir / f'{idx:03d}.gt.jpg')

                cs = colors[pos_mask.to('cpu').numpy()]
                bs = boxes[-1, i][pos_mask].to('cpu')
                errs = sigmas[i][pos_mask].t().to('cpu').numpy()  # [4, N]
                vis = Visualizer(vis_image)
                vis.overlay_instances(
                    boxes=boxes[-1, i][pos_mask].to('cpu'),
                    assigned_colors=cs,
                )
                vis.output.ax.errorbar(
                    (bs[:, 0] + bs[:, 2]) / 2,
                    (bs[:, 1] + bs[:, 3]) / 2,
                    xerr=errs[[0, 2]],
                    yerr=errs[[1, 3]],
                    ecolor=cs,
                    fmt='none',
                )
                vis.output.ax.set_xlim(0, W)
                vis.output.ax.set_ylim(H, 0)
                vis.output.save(vis_dir / f'{idx:03d}.unc.jpg')

                idx += 1


class COCOEvaluation(callbacks.Callback):
    def __init__(self, loader, gt_json, interval=1):
        super().__init__()
        self.loader = loader
        self.gt_json = gt_json
        self.interval = interval
        self.metric_names = []
        self.metric_names.extend(["AP", "AP50", "AP75", "APs", "APm", "APl"])
        self.metric_names.extend(["AR1", "AR10", "AR100", "ARs", "ARm", "ARl"])

    @pl.utilities.rank_zero_only
    def on_train_epoch_end(self, trainer, model: Model):
        if (trainer.current_epoch + 1) % self.interval != 0:
            return

        log_dir = Path(trainer.logger.log_dir)
        out_dir = log_dir / 'evaluation' / f'{trainer.current_epoch:03d}'
        out_dir.mkdir(parents=True, exist_ok=True)

        image_id = 0
        model.eval()
        predictions = []
        for inps in tqdm(iter(self.loader), desc='Evaluate', leave=False):
            with torch.no_grad():
                images = []
                for inp in inps:
                    images.append(inp['image'].float())
                fpn_feats, sizes = model.extract_fpn_feats(images)
                logits, boxes, feats = model.detect(fpn_feats, sizes)
                alphas = model.unc_head(feats[-1])
                anchors = boxes[-2].detach()
                whs = anchors[..., 2:] - anchors[..., :2]  # [N, P, 2]
                sigmas = torch.exp(0.5 * alphas)  # [N, P, 4]
                sigmas = sigmas * torch.cat([whs, whs], dim=2)
                outs = model.postprocess(logits, boxes, sigmas)

            for i in range(len(images)):
                inp = inps[i]
                scores, labels, boxes, sigmas = outs[i]

                scale_x = inp['rawW'] / images[i].shape[2]
                scale_y = inp['rawH'] / images[i].shape[1]
                boxes[:, 0::2] *= scale_x
                boxes[:, 1::2] *= scale_y

                boxes = box_convert(boxes, 'xyxy', 'xywh')

                for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    predictions.append(
                        {
                            'image_id': inp['image_id'],
                            'bbox': b,
                            'category_id': l,
                            'score': s,
                        }
                    )

                image_id += 1

        pred_json_path = out_dir / 'predictions.json'
        with open(pred_json_path, 'w') as f:
            json.dump(predictions, f)

        coco_gt = COCO(str(self.gt_json))
        coco_det = coco_gt.loadRes(str(pred_json_path))
        coco_eval = COCOeval_opt(coco_gt, coco_det, 'bbox')
        coco_eval.params.imgIds = list(set(ann['image_id'] for ann in predictions))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(self.metric_names)
        }

        # Using `model.log` seems to freeze the program
        tb = trainer.logger.experiment
        for k, v in metrics.items():
            tb.add_scalar(f'detection/{k}', v, global_step=trainer.global_step)


if __name__ == '__main__':
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=Path, required=True)
    parser.add_argument('--valid_json', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, default='weights/p300_coco.pth')
    parser.add_argument('--log_dir', type=Path, default='logs')
    parser.add_argument('--batch_size', type=int, default=6) #8
    parser.add_argument('--callback_interval', type=int, default=5) #8
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40) #40
    parser.add_argument('--num_proposals', type=int, default=300)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(args)

    assert args.train_json.exists()
    assert args.valid_json.exists()
    assert args.pretrained.exists()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    train_set = TrainData(args.train_json, mode='train')
    valid_set = TrainData(args.valid_json, mode='infer')
    visul_set = ConcatDataset(
        [random_subset(train_set, k=32), random_subset(valid_set, k=32)]
    )

    print('#train:', len(train_set))
    print('#valid:', len(valid_set))
    
    if args.debug:
        args.gpus = 1
        args.batch_size = 2
        args.callback_interval = 1
        train_set = random_subset(train_set, k=256)
        valid_set = random_subset(valid_set, k=32)
        visul_set = random_subset(visul_set, k=32)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=lambda x: x,
    )
    valid_loader = DataLoader(
        valid_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=lambda x: x,
    )
    visul_loader = DataLoader(
        visul_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=lambda x: x,
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        terminate_on_nan=True,
        gradient_clip_val=2.0,
        max_epochs=args.epochs,
        log_every_n_steps=5,
        default_root_dir=str(args.log_dir),
        callbacks=[
            callbacks.GPUStatsMonitor(),
            callbacks.lr_monitor.LearningRateMonitor(),
            callbacks.ModelCheckpoint(
                save_top_k=-1,
                every_n_epochs=args.callback_interval,
                save_weights_only=True,
            ),
            Visualization(visul_loader, interval=args.callback_interval),
            COCOEvaluation(
                valid_loader, args.valid_json, interval=args.callback_interval
            ),
        ],
    )

    model = Model(num_proposals=args.num_proposals)
    old_weights = model.state_dict()
    new_weights = torch.load(args.pretrained)
    if 'state_dict' in new_weights:
        new_weights = new_weights['state_dict']
    new_weights = {k: v for k, v in new_weights.items() if v.shape == old_weights[k].shape}
    check = model.load_state_dict(new_weights, strict=False)
    print(check.missing_keys)

    total_steps = len(train_set) * args.epochs / args.gpus / args.batch_size
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=5e-4)
    scheduler = {
        'interval': 'step',
        'scheduler': WarmupMultiStepScheduler(
            optimizer,
            500,
            milestones=[round(total_steps * 0.5)],
        ),
    }
    model.configure_optimizers = lambda: ([optimizer], [scheduler])

    trainer.fit(model, train_loader)
