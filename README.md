# Table of Contents
- [Directory Tree](#directorytree)

## Directory Tree

this repository:
|
│  .env
│  main.ipynb
│  README.md
│  
├─checkpoints
│  └─experiment_name
│     └─node_n
│            loss.pkl
│            param.pkl
│
└─module
    ├─draw
    │  └─ draw.py
    │
    ├─for_dataset
    │  │  create_dataset.py
    │  └─ normalization.py
    │
    └─for_model
        │  checkpoint.py
        │  shallow_nn.py
        └─ training.py

.env is for debug only.