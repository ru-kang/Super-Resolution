# Super-Resolution
## Environment
tensorflow==2.14.0
## Dataset
[BSD500](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500/data)
[DIV2K](https://www.kaggle.com/datasets/joe1995/div2k-dataset/data)
##  Folder tree structure
```
Super-Resolution
│  .gitignore
│  README.md
│  Super-Resolution.py
│  
└─Data(Need to Download)
    ├─BSD500
    │  ├─ground_truth
    │  │  ├─test
    │  │  │      100007.mat
    │  │  │      ...
    │  │  │      97010.mat
    │  │  │      
    │  │  ├─train
    │  │  │      100075.mat
    │  │  │      ...
    │  │  │      97017.mat
    │  │  │      
    │  │  └─val
    │  │          101085.mat
    │  │          ...
    │  │          97033.mat
    │  │          
    │  └─images
    │      ├─test
    │      │      100007.jpg
    │      │      ...
    │      │      97010.jpg
    │      │      Thumbs.db
    │      │      
    │      ├─train
    │      │      100075.jpg
    │      │      ...
    │      │      97017.jpg
    │      │      Thumbs.db
    │      │      
    │      └─val
    │              101085.jpg
    │              ...
    │              97033.jpg
    │              Thumbs.db
    │              
    └─DIV2K
        ├─DIV2K_train_HR
        │      0001.png
        │      ...
        │      0800.png
        │      
        └─DIV2K_valid_HR
                0801.png
                ...
                0900.png
```
                

