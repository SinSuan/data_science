# Table of Contents
- [Directory Tree](#directorytree)

## Directory Tree
<pre>
this repository:  
|  
│  .env  
│  main.ipynb  
│  README.md  
│    
├─data (you need to add this)  
│    
├─GIFs (you need to add this)  
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
</pre>
.env is for debug only.