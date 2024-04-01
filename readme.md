# Tuple Perturbation-based Contrastive Learning Framework for Multimodal Remote Sensing Image Semantic Segmentation.
## Abstract




## Dataset Directory Structure
-------
File Structure is as follows:   

    $train_RGB/*.tif     
    $train_SAR/*.tif     
    $train_lbl/*.tif     
    $val_RGB/*.tif 
    $val_SAR/*.tif
    $val_lbl/*.tif    
    train_RGB.txt
    train_SAR.txt    
    train_lbl.txt    
    trainR1_RGB.txt
    trainR1_SAR.txt    
    trainR1_lbl.txt       
    val_RGB.txt
    val_SAR.txt    
    val_lbl.txt
    
## Training
-------         
To pretrain the model with our TMCNet and finetune , try the following command:      
```
python main_ss.py  root=./data_example/Potsdam
    --ex_mode=1  --self_mode=1 \  
    --self_max_epoch=400  --ft_max_epoch=150 \
    --self_data_name=train  --ft_train_name=trainR1
```   
    
