# Tuple Perturbation-based Contrastive Learning Framework for Multimodal Remote Sensing Image Semantic Segmentation.
## Abstract

Deep learning models exhibit promising potential in multi-modal remote sensing image semantic segmentation (MRSISS). However, the constrained access to labeled samples for training deep learning networks significantly influences the performance of these models. To address that, self-supervised learning (SSL) methods have garnered significant interest in the remote sensing community. Accordingly, this article proposes a novel multi-modal contrastive learning framework based on tuple perturbation, which includes the pre-training and fine-tuning stages. Firstly, a tuple perturbation-based multi-modal contrastive learning network (TMCNet) is designed to better explore shared and different feature representations across modalities during the pre-training stage and the tuple perturbation module is introduced to improve the network's ability to extract multi-modal features by generating more complex negative samples. In the fine-tuning stage, we develop a simple and effective multi-modal semantic segmentation network (MSSNet), which can reduce noise by using complementary information from various modalities to integrate multi-modal features more effectively, resulting in better semantic segmentation performance. Extensive experiments have been carried out on the YESeg-OPT-SAR and WHU-OPT-SAR datasets including optical and SAR pairs with different resolutions, and the results show that the proposed TMCNet and MSSNet outperform the current state-of-the-art techniques. In cases of limited labeled samples, the proposed framework exceeds the performance of competing methods in the accuracy of semantic segmentation, achieving higher overall accuracy (OA) and Kappa coefficient. The source code is available at https://github.com/yeyuanxin110/TMCNet-MSSNet.


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
    
