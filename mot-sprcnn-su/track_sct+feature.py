from email import header
import torch
import numpy as np
import pandas as pd
import argparse
import lapsolver
from collections import namedtuple
from tqdm import tqdm
from pathlib import Path
from torchvision.ops import box_convert, box_iou
from utils import MOTEvaluator, cosine_similarity, fix_invalid_bbox
from filterpy import kalman


Detection = namedtuple('Detection', ['bbox', 'emb', 'score', 'unc'])


class Track:
    F = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0],  # x1
            [0, 1, 0, 0, 0, 1, 0, 0],  # y1
            [0, 0, 1, 0, 0, 0, 1, 0],  # x2
            [0, 0, 0, 1, 0, 0, 0, 1],  # y2
            [0, 0, 0, 0, 1, 0, 0, 0],  # v_x1
            [0, 0, 0, 0, 0, 1, 0, 0],  # v_y1
            [0, 0, 0, 0, 0, 0, 1, 0],  # v_x2
            [0, 0, 0, 0, 0, 0, 0, 1],  # v_y2
        ]
    )  # state transition

    Q = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # x1
            [0, 1, 0, 0, 0, 0, 0, 0],  # y1
            [0, 0, 1, 0, 0, 0, 0, 0],  # x2
            [0, 0, 0, 1, 0, 0, 0, 0],  # y2
            [0, 0, 0, 0, 1e-2, 0, 0, 0],  # v_x1
            [0, 0, 0, 0, 0, 1e-2, 0, 0],  # v_y1
            [0, 0, 0, 0, 0, 0, 1e-2, 0],  # v_x2
            [0, 0, 0, 0, 0, 0, 0, 1e-2],  # v_y2
        ]
    )  # process noise uncertainty

    H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )  # convert x to z

    def __init__(
        self, track_id: int, det: Detection, n_init: int, max_age: int, window: int
    ):
        self.track_id = track_id
        self.n_init = n_init
        self.max_age = max_age
        self.window = window
        self.history = []
        self.history.append(det)

        self.state = 'tentative'
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[:4, 0] = det.bbox.numpy()
        self.P = np.array(
            [
                [10, 0, 0, 0, 0, 0, 0, 0],  # x1
                [0, 10, 0, 0, 0, 0, 0, 0],  # y1
                [0, 0, 10, 0, 0, 0, 0, 0],  # x2
                [0, 0, 0, 10, 0, 0, 0, 0],  # y2
                [0, 0, 0, 0, 1e4, 0, 0, 0],  # v_x1
                [0, 0, 0, 0, 0, 1e4, 0, 0],  # v_y1
                [0, 0, 0, 0, 0, 0, 1e4, 0],  # v_x2
                [0, 0, 0, 0, 0, 0, 0, 1e4],  # v_y2
            ]
        )  # state covariance matrix

    @property
    def unc(self):
        return self.history[-1].unc

    @property
    def imgemb(self):
        return self.history[-1].emb

    @property
    def bbox(self):
        bbox = torch.from_numpy(self.x[:4, 0]).float()
        return fix_invalid_bbox(bbox)

    @property
    def emb(self):
        uncs = []
        embs = []
        for det in self.history[-self.window :]:
            wh = det.bbox[2:] - det.bbox[:2]
            unc = (det.unc / torch.cat([wh, wh])).sum()
            uncs.append(unc.sum())
            embs.append(det.emb)
        uncs = torch.tensor(uncs)
        embs = torch.stack(embs)
        emb = embs[uncs.argmin()]
        return emb

    @property
    def score(self):
        return self.history[-1].score

    def predict(self):
        self.x, self.P = kalman.predict(self.x, self.P, Track.F, Track.Q)
        self.age += 1
        self.time_since_update += 1

    def update(self, det):
        self.x, self.P = kalman.update(
            self.x, self.P, det.bbox.numpy(), np.diag(det.unc), Track.H
        )
        self.history.append(det)
        self.hits += 1
        self.time_since_update = 0
        if self.state == 'tentative' and self.hits >= self.n_init:
            self.state = 'confirmed'

    def mark_missed(self):
        if self.state == 'tentative':
            self.state = 'deleted'
        elif self.time_since_update > self.max_age:
            self.state = 'deleted'


class Tracker:
    def __init__(self, max_age=30, n_init=3, window=25):
        self.max_age = max_age
        self.n_init = n_init
        self.window = window
        self.tracks = []
        self.track_id = 0

    def step(self, detections):
        for track in self.tracks:
            track.predict()

        confirmed_tracks = [
            track for track in self.tracks if track.state == 'confirmed'
        ]
        remaining_tracks = [
            track for track in self.tracks if track.state != 'confirmed'
        ] # tracks that are not matched in appearanace and tracks that are tentative

        if len(confirmed_tracks) > 0 and len(detections) > 0:
            confirmed_boxes = torch.stack([track.bbox for track in confirmed_tracks])
            confirmed_embs = torch.stack([track.emb for track in confirmed_tracks])
            detected_boxes = torch.stack([det.bbox for det in detections])
            detected_embs = torch.stack([det.emb for det in detections])
            iou_sim = box_iou(confirmed_boxes, detected_boxes)
            app_sim = cosine_similarity(confirmed_embs, detected_embs)
            iou_sim[iou_sim < 0.2] = np.nan
            cost = -(0.25 * iou_sim + 0.75 * app_sim)
            rr, cc = lapsolver.solve_dense(cost.numpy())
            for r, c in zip(rr, cc):
                confirmed_tracks[r].update(detections[c])
            unmatched_rr = np.setdiff1d(np.arange(len(confirmed_tracks)), rr)
            unmatched_cc = np.setdiff1d(np.arange(len(detections)), cc)
            unmatched_tracks = [confirmed_tracks[r] for r in unmatched_rr]
            remaining_tracks.extend(unmatched_tracks)
            detections = [detections[c] for c in unmatched_cc]

        if len(remaining_tracks) > 0 and len(detections) > 0:
            remaining_boxes = torch.stack([track.bbox for track in remaining_tracks])
            remaining_embs = torch.stack([track.emb for track in remaining_tracks])
            detected_boxes = torch.stack([det.bbox for det in detections])
            detected_embs = torch.stack([det.emb for det in detections])
            iou_sim = box_iou(remaining_boxes, detected_boxes)
            app_sim = cosine_similarity(remaining_embs, detected_embs)
            iou_sim[iou_sim < 0.4] = np.nan
            cost = -(1.0 * iou_sim + 0.0 * app_sim)
            rr, cc = lapsolver.solve_dense(cost.numpy())
            for r, c in zip(rr, cc):
                remaining_tracks[r].update(detections[c])
            unmatched_rr = np.setdiff1d(np.arange(len(remaining_tracks)), rr)
            unmatched_cc = np.setdiff1d(np.arange(len(detections)), cc)
            unmatched_tracks = [remaining_tracks[r] for r in unmatched_rr]
            detections = [detections[c] for c in unmatched_cc]

            for track in unmatched_tracks:
                track.mark_missed()

        # New tracks
        if len(detections) > 0:
            for det in detections:
                self.track_id += 1
                self.tracks.append(
                    Track(self.track_id, det, self.n_init, self.max_age, self.window)
                )

        # Remove tracks
        self.tracks = [track for track in self.tracks if track.state != 'deleted']

        # Extract result
        result = torch.tensor(
            [
                # (track.track_id, *track.bbox.tolist(), track.score.item())
                (track.track_id, *track.bbox.tolist(), track.score.item(), *track.unc.tolist(), *track.imgemb.tolist())
                for track in self.tracks
                if track.time_since_update == 0 and track.state == 'confirmed'
            ]
        )

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=Path, default='videos/mot17')
    parser.add_argument('--dets', type=Path, required=True)
    parser.add_argument('--outs', type=Path, required=True)
    args = parser.parse_args()

    assert args.videos.exists()
    assert args.dets.exists()
    args.outs.mkdir(parents=True, exist_ok=True)

    for video_paths in sorted(list(args.videos.glob('*.imgs'))):
        video_name = video_paths.stem
        fids = pd.read_csv(video_paths)['fid'].values
        dets = pd.read_csv(args.dets / f'{video_name}.csv')
        embs = torch.load(args.dets / f'{video_name}.emb')

        tracker = Tracker()

        preds = []
        preds_features = []
        for fid in tqdm(fids, desc=video_name):
            mask = dets['fid'] == fid

            curr_group = dets[mask]
            curr_embs = embs[mask.values]
            curr_boxes = torch.tensor(curr_group[['x', 'y', 'w', 'h']].values)
            curr_boxes = box_convert(curr_boxes, 'xywh', 'xyxy')
            curr_scores = torch.tensor(curr_group['s'].values)
            curr_sigmas = torch.tensor(curr_group[['ul', 'ut', 'ur', 'ub']].values)

            curr_dets = [
                Detection(b, e, s, u)
                for b, e, s, u in zip(curr_boxes, curr_embs, curr_scores, curr_sigmas)
            ]

            result = tracker.step(curr_dets)
            if len(result) > 0:
                feature = result[:, 10:].numpy()
                feature = pd.DataFrame(feature)
                preds_features.append(feature)

                result = result[:, :10]
                result[:, 1:5] = box_convert(result[:, 1:5], 'xyxy', 'xywh')
                result = result.numpy()
                result = pd.DataFrame(result, columns=['tag', 'x', 'y', 'w', 'h', 's', 'ul', 'ut', 'ur', 'ub'])
                result['fid'] = fid
                result['tag'] = result['tag'].astype(int)
                preds.append(result)
                
        print(f"saving {video_name}.txt ...")
        df_pred = pd.concat(preds, ignore_index=True)
        df_pred.to_csv(
            args.outs / f'{video_name}.txt',
            index=None,
            header=None,
            columns=['fid', 'tag', 'x', 'y', 'w', 'h', 's', 'ul', 'ut', 'ur', 'ub'],
        )
        
        print(f"saving {video_name}_features.txt ...")
        df_fid_tag = df_pred[['fid','tag']]
        df_feature = pd.concat(preds_features, ignore_index=True)
        df_pred_feature = pd.concat([df_fid_tag, df_feature], axis=1, ignore_index=True)
        df_pred_feature.to_csv(
            args.outs / f'{video_name}_features.txt',
            index=None,
            header=None,
        )

evaluator = MOTEvaluator(remove_distractor=True)
for gt_path in sorted(list(args.videos.glob('*.gt'))):
    video_name = gt_path.stem
    df_pred = pd.read_csv(args.outs / f'{video_name}.txt')
    df_pred.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', 'ul', 'ut', 'ur', 'ub']
    df_true = pd.read_csv(gt_path)
    evaluator.add(video_name, df_pred, df_true)
summary, text = evaluator.evaluate(False)
print(text)