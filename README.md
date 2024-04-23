# [Predicting Pedestrian Trajectory via Interaction Conditioned Latents](https://arxiv.org/abs/2402.17339)

![contents](figures/framework.png)
Pedestrian trajectory prediction is the key technology in many applications for providing insights into human behavior and anticipating human future motions. Most existing empirical models are explicitly formulated by observed human behaviors using explicable mathematical terms with deterministic nature, while recent work has focused on developing hybrid models combined with learning-based techniques for powerful expressiveness while maintaining explainability. However, the deterministic nature of the learned steering behaviors from the empirical models limits the models' practical performance. To address this issue, this work proposes the social conditional variational autoencoder (SocialCVAE) for predicting pedestrian trajectories, which employs a CVAE to explore behavioral uncertainty in human motion decisions. SocialCVAE learns socially reasonable motion randomness by utilizing a socially explainable interaction energy map as the CVAE's condition, which illustrates the future occupancy of each pedestrian's local neighborhood area. The energy map is generated using an energy-based interaction model, which anticipates the energy cost (i.e., repulsion intensity) of pedestrians' interactions with neighbors. Experimental results on two public benchmarks including 25 scenes demonstrate that SocialCVAE significantly improves prediction accuracy compared with the state-of-the-art methods, with up to 16.85\% improvement in Average Displacement Error (ADE) and 69.18\% improvement in Final Displacement Error (FDE).

## Environments
Python 3.9

Pytorch 1.12.0

## Dataset preparation
Please download the processed data from 
[Baidu Yun](https://pan.baidu.com/s/1H19JCh4uQAbOsmSGrk0iKw?pwd=sc24) or [Google drive](https://drive.google.com/file/d/1razsS-UsrHw-2v6ZR-tAGui4bkjMv2pK/view?usp=sharing).

Place the unzipped data into `./data/SDD/`

## Training
To train new networks, run
`train_SDD.py`

If you find our code useful, please cite:

```bibtex
@inproceedings{xiang2024socialcvae,
  title={SocialCVAE: Predicting Pedestrian Trajectory via Interaction Conditioned Latents},
  author={Xiang, Wei and Yin, Haoteng and Wang, He and Jin, Xiaogang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={6216--6224},
  year={2024}
}
```






