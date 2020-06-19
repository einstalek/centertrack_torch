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
                        'active': 1, 'age': 1,
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
        for m in matches:
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1
            track['active'] = self.tracks[m[1]]['active'] + 1
            ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            if track['score'] > self.opt.new_thresh:
                self.id_count += 1
                track['tracking_id'] = self.id_count
                track['age'] = 1
                track['active'] = 1
                ret.append(track)
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
