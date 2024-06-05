# Table of Contents
- [使用說明](#使用說明)
- [警告](#警告)
- [Directory Tree](#directorytree)

## 使用說明
步驟:
1. 創建 data/, GIFs/, checkpoints/
2. 先執行 part2_train.ipynb, 再執行 part2_draw.ipynb (part3 同理)
  
注意事項:

1. name_experiment 會作為 checkpoints/ 下子資料夾的名稱，*.pybn 中，每個小題都會宣告一次，需要自行手動更改
2. module\draw\draw.py 中的 plot_decision_boundary 有宣告路徑，需要手動更改成 part3_train.ipynb 中的 data_2_save
3. 其他檔案有更動，則須重啟 .ipynb 的 kernel，否則不會 import 最新的檔案

## 警告
part2 依照作業要求匯 train <b>六小時</b>，建議如果只是要 demo 的話可以減少內層迴圈數(layer_initializer)，以免影響其他 code 的運行

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