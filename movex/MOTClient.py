import time
import numpy as np


class MOTClient(object):
    def __init__(self, client_config):
        """Creates a new detection source that implements the `poll` function.
        Detections are for the MOT16 dataset, sourced from a Faster RCNN model.
        The MOTClient object implements a configurable latency, so it only returns
        detections for a given call to `poll` if the latency time has elapsed.
        """
        self._latency_in_seconds = client_config["latency"] / 1000
        self._frames_before_inference = int(
            self._latency_in_seconds * client_config["video_fps"]
        )
        self._inference_countdown = self._frames_before_inference
        self._detections = np.load(client_config["path_to_detections"])

    def poll(self, frame_idx):
        """Indexes into the predefined dection matrix if enough time has elapsed to
        achieve the simulated latency. If enough time has not passed, return None."""

        if self._inference_countdown == 0:
            detections = self._create_detections(frame_idx)
            self._inference_countdown = self._frames_before_inference

        else:
            detections = None
            self._inference_countdown -= 1

        return detections

    def wait(self, frame_idx):
        last = time.perf_counter()
        detections = self._create_detections(frame_idx)
        now = time.perf_counter()
        created_detections_time = now - last

        time.sleep(self._latency_in_seconds - created_detections_time)
        return detections

    def _create_detections(self, frame_idx):
        frame_indices = self._detections[:, 0].astype(np.int)
        mask = frame_indices == frame_idx

        bboxes = []
        scores = []
        for row in self._detections[mask]:
            bbox, confidence = row[2:6], row[6]
            if bbox[3] < 0:
                # The reason that this checks if bbox[3] is less than 0
                # is just in the off chance that the height of a bbox is negative
                # This obviously should not be allowed
                continue

            else:
                bboxes.append(bbox)
                scores.append(confidence)

        return (bboxes, scores)
