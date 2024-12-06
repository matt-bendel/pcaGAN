# pcaGAN: Improving Posterior-Sampling cGANs via Principal Component Regularization, NeurIPS 2024 [[arXiv]](https://arxiv.org/pdf/2411.00605) [[Open Review]](https://openreview.net/pdf?id=Z0Nq3hHeEG)

This repo contains the official implementation for the paper [pcaGAN: Improving Posterior-Sampling cGANs via Principal Component Regularization](https://openreview.net/forum?id=Z0Nq3hHeEG)

By: Matthew Bendel, Rizwan Ahmad, and Philip Schniter

## Abstract
In ill-posed imaging inverse problems, there can exist many hypotheses that fit
both the observed measurements and prior knowledge of the true image. Rather
than returning just one hypothesis of that image, posterior samplers aim to explore the full solution space by generating many probable hypotheses, which can
later be used to quantify uncertainty or construct recoveries that appropriately
navigate the perception/distortion trade-off. In this work, we propose a fast and
accurate posterior-sampling conditional generative adversarial network (cGAN)
that, through a novel form of regularization, aims for correctness in the posterior
mean as well as the trace and K principal components of the posterior covariance
matrix. Numerical experiments demonstrate that our method outperforms contemporary cGANs and diffusion models in imaging inverse problems like denoising,
large-scale inpainting, and accelerated MRI recovery.

---

This code is still under construction. Please check back later!

[//]: # (## Setup)

[//]: # (See ```docs/setup.md``` for basic environment setup instructions.)

[//]: # ()
[//]: # (## Reproducing our Results)

[//]: # (### MRI)

[//]: # (See ```docs/mri.md``` for instructions on how to setup and reproduce our MRI results.)

[//]: # ()
[//]: # (## Extending the Code)

[//]: # (See ```docs/new_applications.md``` for basic instructions on how to extend the code to your application.)

[//]: # ()
[//]: # (## Questions and Concerns)

[//]: # (If you have any questions, or run into any issues, don't hesitate to reach out at bendel.8@osu.edu.)

[//]: # ()
[//]: # (## References)

[//]: # (This repository contains code from the following works, which should be cited:)

[//]: # ()
[//]: # (```)

[//]: # (@article{zbontar2018fastmri,)

[//]: # (  title={fastMRI: An open dataset and benchmarks for accelerated MRI},)

[//]: # (  author={Zbontar, Jure and Knoll, Florian and Sriram, Anuroop and Murrell, Tullie and Huang, Zhengnan and Muckley, Matthew J and Defazio, Aaron and Stern, Ruben and Johnson, Patricia and Bruno, Mary and others},)

[//]: # (  journal={arXiv preprint arXiv:1811.08839},)

[//]: # (  year={2018})

[//]: # (})

[//]: # ()
[//]: # (@article{devries2019evaluation,)

[//]: # (  title={On the evaluation of conditional GANs},)

[//]: # (  author={DeVries, Terrance and Romero, Adriana and Pineda, Luis and Taylor, Graham W and Drozdzal, Michal},)

[//]: # (  journal={arXiv preprint arXiv:1907.08175},)

[//]: # (  year={2019})

[//]: # (})

[//]: # ()
[//]: # (@inproceedings{Karras2020ada,)

[//]: # (  title={Training Generative Adversarial Networks with Limited Data},)

[//]: # (  author={Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},)

[//]: # (  booktitle={Proc. NeurIPS},)

[//]: # (  year={2020})

[//]: # (})

[//]: # ()
[//]: # (@inproceedings{zhao2021comodgan,)

[//]: # (  title={Large Scale Image Completion via Co-Modulated Generative Adversarial Networks},)

[//]: # (  author={Zhao, Shengyu and Cui, Jonathan and Sheng, Yilun and Dong, Yue and Liang, Xiao and Chang, Eric I and Xu, Yan},)

[//]: # (  booktitle={International Conference on Learning Representations &#40;ICLR&#41;},)

[//]: # (  year={2021})

[//]: # (})

[//]: # ()
[//]: # (@misc{zeng2022github,)

[//]: # (    howpublished = {Downloaded from \url{https://github.com/zengxianyu/co-mod-gan-pytorch}},)

[//]: # (    month = sep,)

[//]: # (    author={Yu Zeng},)

[//]: # (    title = {co-mod-gan-pytorch},)

[//]: # (    year = 2022)

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (## Citation)

[//]: # (If you find this code helpful, please cite our paper:)

[//]: # (```)

[//]: # (@journal{bendel2022arxiv,)

[//]: # (  author = {Bendel, Matthew and Ahmad, Rizwan and Schniter, Philip},)

[//]: # (  title = {A Regularized Conditional {GAN} for Posterior Sampling in Inverse Problems},)

[//]: # (  year = {2022},)

[//]: # (  journal={arXiv:2210.13389})

[//]: # (})
[//]: # (```)