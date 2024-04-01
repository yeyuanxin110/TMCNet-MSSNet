# Global and Local Contrastive Self-Supervised Learning for Semantic Segmentation of HR Remote Sensing Images.
## Abstract


Recently, supervised deep learning has achieved great success in remote sensing image (RSI) semantic segmentation. However, supervised learning for semantic segmentation requires a large number of labeled samples, which is difficult to obtain in the field of remote sensing. A new learning paradigm, self-supervised learning (SSL), can be used to solve such problems by pre-training a general model with a large number of unlabeled images and then fine-tuning it on a downstream task with very few labeled samples. Contrastive learning is a typical method of SSL that can learn general invariant features. However, most existing contrastive learning methods are designed for classification tasks to obtain an image-level representation, which may be suboptimal for semantic segmentation tasks requiring pixel-level discrimination. Therefore, we propose a global style and local matching contrastive learning network (GLCNet) for remote sensing image semantic segmentation. Specifically, 1) the global style contrastive learning module is used to better learn an image-level representation, as we consider that style features can better represent the overall image features. 2) The local features matching contrastive learning module is designed to learn representations of local regions, which is beneficial for semantic segmentation. We evaluate four RSI semantic segmentation datasets, and the experimental results show that our method mostly outperforms state-of-the-art self-supervised methods and the ImageNet pre-training method. Specifically, with 1\% annotation from the original dataset, our approach improves Kappa by 6\% on the ISPRS Potsdam dataset relative to the existing baseline. Moreover, our method outperforms supervised learning methods when there are some differences between the datasets of upstream tasks and downstream tasks. Our study promotes the development of self-supervised learning in the field of RSI semantic segmentation. Since SSL could directly learn the essential characteristics of data from unlabeled data, which is easy to obtain in the remote sensing field, this may be of great significance for tasks such as global mapping. 

You can visit the paper via https://arxiv.org/abs/2106.10605 or 

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
    
