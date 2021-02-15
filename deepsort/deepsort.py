import os, time
import cv2
import numpy as np
import tensorflow as tf

from .sort.detection import Detection
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.tracker import Tracker
from .deep import Model as Extractor


class DeepSortTracker(object):
    def __init__(
        self,
        nn_budget=None,
        max_cosine_distance=0.5,
        nms_max_overlap=0.3,
        min_confidence=0.8,
        max_iou_distance=0.7,
        n_init=3,
    ):
        """asdfasdfasdasdf"""
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)

            except RuntimeError as e:
                print(e)

        self.input_shape = [128, 64]  # Height, width

        # Load the feature extraction network from checkpoints
        parent_dir = os.path.dirname(__file__)
        self.extractor = Extractor(self.input_shape)
        self.extractor.load_weights(
            os.path.join(parent_dir, "checkpoints/extractor.tf")
        ).expect_partial()

        # Configure the SORT tracker
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, n_init=n_init)

    def track(self, img, bboxes_in, scores, tlbr=True):
        """If not tblr assume its MOT testing and use ltwh"""

        patches = []
        in_bboxes = bboxes_in.copy()
        bboxes = []
        for i in range(len(in_bboxes)):
            score = scores[i]

            # Skip if not confident enough
            if score < self.min_confidence:
                continue

            # Unpack the bbox
            bbox = in_bboxes[i]
            if tlbr:
                left, top, right, bottom = map(lambda x: int(x) if x > 0 else 0, bbox)
                in_bboxes[i] = left, top, right - left, bottom - top

            else:
                left, top, width, height = map(lambda x: int(x), bbox)
                right = left + width
                bottom = top + height

            # Extract feature vector
            patch = img[top:bottom, left:right]
            # patch = tf.expand_dims(patch, 0)

            # Check just in case
            if 0 in patch.shape:
                continue

            # Resize and normalize
            patch = tf.image.resize(patch, self.input_shape)
            patch /= 255
            patches.append(patch)

        # Inference on the bbox
        if len(patches) > 0:
            patches = np.stack(patches)
            features = self.extractor.predict(patches)

        else:
            features = []

        # Create detection: bbox in the form x, y, w, h
        detections = [
            Detection(bbox, 1, feature) for bbox, feature in zip(in_bboxes, features)
        ]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)

        # Pack up the output
        bboxes = []
        track_ids = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

                # if tlbr:
                #    bbox = track.to_tlbr()

                # else:
            bbox = track.to_tlwh()

            bboxes.append(bbox)
            track_ids.append(track.track_id)

        return bboxes, track_ids
