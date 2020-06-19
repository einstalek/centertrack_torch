from abc import abstractmethod


class BaseTracker:
    @abstractmethod
    def init_track(self, dets):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, dets):
        pass

    @staticmethod
    def split_results(dets):
        items = []
        for k in dets:
            dets[k] = dets[k].detach().numpy()
        for i, score in enumerate(dets['scores'][0]):
            if score < 0.3:
                continue
            item = {'score': float(score),
                    'bbox': dets['bboxes'][0, i],
                    'class': 1,
                    'ct': dets['cts'][0, i],
                    'tracking': dets['tracking'][0, i]}
            items.append(item)
        return items
