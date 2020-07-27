import numpy as np
import copy

from tracker.base_tracker import BaseTracker


class Tracker(BaseTracker):
    def __init__(self, opt):
        self.opt = opt
        self.init = False
        self.reset()

    def init_track(self, dets):
        items = self.split_results(dets)
        self.init = True
        for item in items:
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
                # active and age are never used in the paper
                item = {'score': item['score'],
                        'age': 1,
                        'inactive': 0,
                        'tracking_id': self.id_count,
                        'bbox': item['bbox'],
                        'class': item['class'],
                        'ct': item['ct']}
                self.tracks.append(item)

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def step(self, dets):
        results = self.split_results(dets)
        N = len(results)
        M = len(self.tracks)

        dets = np.array(
            [det['ct'] + det['tracking'] for det in results], np.float32)  # N x 2

        # Filter out close detections
        closeness = ((dets.reshape((1, -1, 2)) - dets.reshape((-1, 1, 2))) ** 2).sum(axis=2)
        xx, yy = np.where(closeness < 2)
        drop_dets_ids = []
        for (x, y) in zip(xx, yy):
            if y <= x:
                continue
            drop_dets_ids.append(y)

        dets_wth_shift = np.array([det['ct'] for det in results], np.float32)

        track_size = 1. * np.array([((track['bbox'][2] - track['bbox'][0]) *
                                     (track['bbox'][3] - track['bbox'][1]))
                                    for track in self.tracks], np.float32)  # M
        item_size = 1. * np.array([((item['bbox'][2] - item['bbox'][0]) *
                                    (item['bbox'][3] - item['bbox'][1]))
                                   for item in results], np.float32)  # N
        tracks = np.array(
            [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2

        dist = (((tracks.reshape((1, -1, 2)) - dets.reshape((-1, 1, 2))) ** 2).sum(axis=2))  # N x M
        dist_wth_shift = (((tracks.reshape((1, -1, 2)) - dets_wth_shift.reshape((-1, 1, 2))) ** 2).sum(axis=2))
        invalid = ((dist > track_size.reshape(1, M)) +
                   (dist > item_size.reshape(N, 1))
                   ) > 0
        dist = dist + invalid * 1e18
        invalid = ((dist_wth_shift > track_size.reshape(1, M)) +
                   (dist_wth_shift > item_size.reshape(N, 1))
                   ) > 0
        dist_wth_shift = dist_wth_shift + invalid * 1e18

        matched_indices = greedy_assignment(copy.deepcopy(dist))
        matched_indices_wth_shift = greedy_assignment(copy.deepcopy(dist_wth_shift))

        for x in matched_indices_wth_shift:
            if x not in matched_indices:
                matched_indices = np.append(matched_indices, [x], axis=0)

        unmatched_dets = [d for d in range(dets.shape[0])
                          if not (d in matched_indices[:, 0])]
        matches = matched_indices

        ret = []
        # Новые треки, смэтчившиеся с предыдущими
        for m in matches:
            res_idx = m[0]
            track = results[res_idx]
            if res_idx in drop_dets_ids:
                continue
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1 + self.tracks[m[1]]['age']
            track['inactive'] = 0
            ret.append(track)

        # Новые треки без мэтча
        for i in unmatched_dets:
            if i in drop_dets_ids:
                continue
            track = results[i]
            if track['score'] > self.opt.new_thresh:
                self.id_count += 1
                track['tracking_id'] = self.id_count
                track['age'] = 1
                track['inactive'] = 0
                ret.append(track)
        # Старые треки без обновлений
        for i in range(len(self.tracks)):
            if i in matches[:, 1]:
                continue
            track = self.tracks[i]
            track['inactive'] += 1
            track['age'] += 1
            if track['inactive'] > 5:
                del self.tracks[i]

        if len(ret) == 0:
            for track in self.tracks:
                track['age'] += 1
        else:
            self.tracks = ret
        return ret


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)
