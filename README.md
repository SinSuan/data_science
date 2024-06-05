# Table of Contents
- [使用說明](#使用說明)
- [Directory Tree](#directorytree)

## 使用說明
步驟:
1. 創建 data/, GIFs/, checkpoints/
2. 先執行 part2_train.ipynb, 再執行 part2_draw.ipynb (part3 同理)
  
注意事項:

1. name_experiment 會作為 checkpoints/ 下子資料夾的名稱
2. 其他檔案有更動，則須重啟 .ipynb 的 kernel，否則不會 import 最新的檔案

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
├─checkpoints (you need to add this)  
│  └─name_experiment  
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
You need to add the three folder by yourself: data/, GIFs/, checkpoints/