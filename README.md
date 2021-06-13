# Overview
This repo contains the accompanying code for our paper [Motion Extrapolation for Video Object Detection](https://arxiv.org/pdf/2104.08918.pdf) 
accepted at the [Robust Video Scene Understanding: Tracking and Video Segmentation](https://eval.vision.rwth-aachen.de/rvsu-workshop21/) CVPR 2021 workshop.

**WARNING**: __There is active work being done on this repository. We make no promises about its stability or backwards compatibility for the time being.__

![MOVEX in action](https://github.com/juliantrue/movex/blob/master/assets/thediagram.png?raw=true)

# Some Differences From the Paper
Since the time of submitting the referenced paper, additional work has been done.

Changes made are:
- MOT16 as an additional evaluation dataset
- global frame compensation implemented for video with a lot of camera movement (like MOT16 sequences 05 and 13)
- ablation options added for removing global frame compensation, aggregation function, multi-processing capabilities, and propagation at large.
- additional optical flow computation types: Robust Local Optical Flow and Dense Pyramid Lucas-Kanade (__unstable__)

# Citation
If you find this work useful, consider citing us:

```
@misc{true2021motion,
      title={Motion Vector Extrapolation for Video Object Detection}, 
      author={Julian True and Naimul Khan},
      year={2021},
      eprint={2104.08918},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


