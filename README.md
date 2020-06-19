# ShadowTutor

This repository accompanies the paper:

> "Shadowtutor: Distributed Partial Distillation for Mobile Video DNN Inference", Jae-Won Chung, Jae-Yun Kim, Soo-Mook Moon

[![arXiv:2003.10735](https://img.shields.io/badge/arXiv-2003.10735-b31b1b.svg)](https://arxiv.org/abs/2003.10735)
[![DOI:10.1145/3404397.3404404](https://zenodo.org/badge/DOI/10.1145/3404397.3404404.svg)](https://doi.org/10.1145/3404397.3404404)

ShadowTutor was implemented with Python 3.6, PyTorch 1.3.0, Detectron2, and OpenMPI. The NVIDIA Jetson Nano embedded board was used as the client device.


## Repository Organization

```
.
├── configs/                       // configuration files
│   ├── Student_dod.yaml
│   ├── Student_pretrain.yaml
│   └── Teacher_dod.yaml
├── datasets/                      // datasets and preparation scripts
│   ├── COCO/
│   ├── coco_prepare.sh
│   ├── coco_sem_seg.py
│   ├── coco_prepare.sh
│   ├── lvsdataset/
│   ├── lvs_convert.sh
│   ├── lvs_download.sh
│   └── lvs_sem_seg.py
├── scripts/
│   └── run_log.sh
├── setup/                         // scripts that setup the server and client
│   ├── client
│   └── server
├── detectron2_repo/               // Detectron2 submodule
├── pretrain_student.py            // script that pretrains that student
├── dod_server.py                  // ShadowTutor, the server side (Algorithm 1, 3)
├── dod_client.py                  // ShadowTutor, the client side (Algorithm 4)
├── dod_common.py                  // shared code (includes Algorithm 2)
├── Student_pretrained.pth         // student pretrained on COCO
├── Student_amended.pth            // number of classes modified for the LVS dataset
├── student.py                     // student architecture
└── utils.py
```

## Installation

Running ShadowTutor can be painful in some respects:

1. You should compile OpenMPI from source, so that it supports CUDA.
2. You should compile PyTorch from source, so that it supports 1.
3. You should do 1 and 2 each for the server (AMD64) and the client (aarch64).
4. Compiling OpenCV from source is also quite painful.
5. Making OpenMPI work between the server and the client is very painful.

However, if you insist, take at look at the `setup` directory. Installation scripts for both the server and the client are provided. Run `setup.sh` on each machine. These scripts may not be perfect, because it was written in an ad-hoc manner. I tried my best though.

Measures that took for me to setup OpenMPI between the server and the client include the following.

1. I setup Passwordless SSH between the two machines. In the server, I set the `HostName` of the client to `jetson`.
2. My `.bashrc` used to return right away when running non-interactively (by OpenMPI), which prevented the `PATH` environment variable from being set. I commented this part out.
3. OpenMPI uses random high-ports for communication. Thus, I disabled firewalls (`ufw`).

If you are still unable to make OpenMPI work, you are kindly pointed to [FAQ: Running MPI Jobs](https://www.open-mpi.org/faq/?category=running) and [FAQ: Troubleshooting](https://www.open-mpi.org/faq/?category=troubleshooting) in the OpenMPI homepage.

## Running ShadowTutor

You can run ShadowTutor with the following command after installing it.

```Bash
# At the HOME directory of the server
bash run_log.sh --max-frames 5000 --video-path <path_to_mp4>
```

Note that `run_log.sh` is copied from the `scripts` folder to your home folder in the last part of `setup/server/setup.sh`.