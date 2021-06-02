# ![image-20210602142343433](../images/image-20210602142343433.png)

# P-stage 1) Image Classification

<br/>

## 대회 개요

 **- [link](http://boostcamp.stages.ai/competitions/1/overview/description)** (현재는 회원들에게만 open이 되어있지만 추후 열릴예정이라고 들었습니다)

## Overview

- **문제정의** : 사람의 사진을 보고 mask의 착용여부, gender, 연령대를 구분

- **Input data** 

  - 4500 명의 사람
  - 한사람당 7개의 Image
    - 마스크를 올바르게 착용한 Image : 5장
    - 마스크를 Incorrect 하게 착용한 Image (턱이나 코에 착용) : 1장
    - 마스크를 착용하지 않은 Image : 1장
  - 총 4500 *  7 * 0.6 = 18900개의 train을 위한 image 존재
  - 4500 * 7 * 0.2 씩 public test set과 private test set이 존재
  - 각각의 image의 확장자는 다양하게 존재

- **Output data**

  - 18개의 class를 도출

  <img src="/Users/jowon/workspace/jo-member.github.io/source/images/image-20210407122730687.png" alt="image-20210407122730687" style="zoom:50%;" />

- 최종 평가 기준 : **f1-score**  
- Age, Gender, Age 사이의 연관성?
- Accuracy보다는 **f1-score**를 높히는데 초점

<br/>

## EDA

문제정의 : 과연 각 label들에 대한 분포가 어떻게 될까?

총 2700명의 data (1사람당 7개의 image 존재)

1. Gender label의 분포
   - Female : 1658 * 7 = 11606 장
   - Male : 1042 * 7 = 7294 장
2. Mask label의 분포
   - Correct : 13500 장 
   - Incorrect : 2700 장 
   - Not Wear : 2700 장
3. Age label의 분포
   - 0~30 : 1281 * 7 = 8967 장
   - 30~60 : 1227 * 7 = 8589 장
   - **60~ : 192 * 7 = 1344 장**

전체적인 분포

- 중년 여성 > 청년 여성 > 청년 남성 > 중년 남성 > 노년 여성 > 노년 남성 순으로 데이터가 많음

![image-20210407125736502](/Users/jowon/workspace/jo-member.github.io/source/images/image-20210407125736502.png)



Mask에 대한 분포는 어차피 mask를 쓴 data가 많이 필요하기 때문에 상관이 없다고 생각했습니다.
하지만 age에 대한 분포를 보면 60세 이상의 image가 매우 적다는걸 알 수 있습니다

이를 해결하기 위한 방법으로 몇가지를 생각해 보았습니다.

1. Focal loss :Class Imbalance 문제가 있는 경우, 맞춘 확률이 높은 Class는 조금의 loss를, 맞춘 확률이 낮은 Class는 Loss를 훨씬 높게 부여
2. Weight cross entropy loss : class의 개수에 따른 역수를 가중치로 넣어줘서 각 class마다 loss의 가중치를 두어 imbalance data에 효과적으로 대처
3. 외부 data의 사용 -----> 성능이 대폭 하락
4. 이미지 자체에 augmentation을 적용해서 이미지의 절대적인 개수를 늘림

<br/>

## Model & Loss & Optimizer

1. **Model**

   단순한 classification task에서 Model은 pretrained된 SOTA model을 사용하는게 image classification에서는 정석이라고 들었습니다.

   앙상블시 성능향상을 위해 resnext와 efficientnet을 train 시켰습니다.

   

   **EfficientNet**

   - 고려점

     1. 과연 pretrained된 backbone-network를 freeze해야 할까?

        이는 어느정도 실험을 하지 않아도 결론이 나와있습니다.

        Pretraining 할 때 설정했던 문제와 현재 문제와의 유사성을 고려해보면 유사성이 높다고 할 수는 없다. 오직 사람의 face에 초점이 된 data이기 때문에 freeze시키지 않는것이 좋다고 판단하였습니다.

        실제로 freeze 시키고 training을하면 제출시 정확도가 큰폭으로 하락하였습니다...

        <br/>

     2. 어느정도 크기의 Model을 사용해야 하나?

        EfficientNet에는 다양한 크기를 가지는 Model이 존재합니다.
        EfficientNet - b(0-8) 까지 숫자가 커질수록 Parameter수가 증가하고 Model의 크기가 증가하기 때문에 학습시에 시간이 소요됩니다.

        <br/>

        **큰 모델을 사용할수록 classification의 성능이 증가?**

        만약 크기가 클수록 성능이 좋아진다면 작은 model에서 hyperparamter를 검증해 본다음에 무조건 efficientnet-b8을 사용하는게 좋을것 입니다. 하지만 train시 b4를 넘어가는 크기의 model은 train시간만 오래걸릴뿐 뚜렷한 성능의 향상을 보여주지 못하였습니다.

        **b4 or b5가 가장 적절**

        <br/>

     3. EfficientNet의 마지막 fc layer를 우리의 task에 맞게 바꾸어주어야 합니다.
        물론 이때 이 layer의 parameter를 initialization 해주어야 좀더 학습이 안정적으로 진행이 될 것입니다.
        
        (pytorch에서의 layer weight initialization은 layer의 종류에 따라 다르게 내부에 구현되어있다)(https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

<br/>

2. **Loss**

   어떠한 Loss를 써야할까???

   - 한가지의 loss만을 사용
   - 2개 이상의 loss를 비율을 나누어 합쳐준 Custom Loss를 사용

   이번 task에서 실험해본 Loss Function

   - **Crossentropy Loss**

   - Focal Loss

   - F1 Loss

   

   Model의 최종 output을 8개의 차원으로 뽑아내고, 1-3은 mask classification,4-5는 gender classiification, 6-8은 age classification을 수행하여 각각의 loss를 더한 loss function을 backward 하며 학습하였습니다.

   

3. **Optimizer**

   - Adam 계열 : AdamP, AdamW
   - SGD 계열

   AdamP라는 optimizer 같은 경우에는 비교적 최근에 네이버에서 발표한 optimizer로 image task에서 굉장히 좋은 성능을 보여주고 있어 채택하였습니다. 실제로 실험결과에서도 Adam 계열이 SGD 계열의 optimzier보다 수렴속도와 acc가 좋았으며, 최종적으로  AdamP를 사용했습니다.

   

   다음으로 lr을 scheduling하는 중요한 part가 남아있습니다. 이번 classification을 진행하며 lr scheduling이 꽤나 성능에 큰 영향을 미친다는점을 느꼈습니다.

   

   lr을 초기부터 크게 설정해 준다면, pretrained의 parameter들이 크게 흔들릴수 있으므로, warm up start를 도입한 cosineannealingwarmrestarts scheduler를 사용하였습니다.

   cosineannealing의 lr은 아래 그래프와 같습니다

   <img src="../images/image-20210602165129448.png" alt="image-20210602165129448" style="zoom:50%;" />

   종료시점까지 step마다 lr이 감소하는 StepLR scheduler과 달리 주기마다 lr값을 높혀줌으로서 model이 local minimum에서 빠져나오게 해줍니다.

   

4. **Augmentation**

   아주 다양한 Augmentation을 진행하였는데

   1. CenterCrop
   2. RandomCrop
   3. GaussianNoise
   4. Horizontal Flip
   5. ColorJitter
   6. Normalize
   7. Cutmix

   결국 결과는 놀랍게도 normalize만을 적용시켰을때의 model의 성능이 가장 좋았습니다.

   우리가 가진 data는 매우 잘 정제되어있는?! 데이터입니다. 우리가 가진 이미지에서 사람은 대부분을 차지하고 위치 또한 비슷합니다.  우리가 굳이 augmetation을 안해주더라도 data가 너무 좋아 model에게 어려운 문제를 학습시키지 않아도 되는것 같습니다.

   다음에는 autoaugmentation을 한번 적용해서 model의 성능을 올려보고 싶습니다.

   

5. Validation

   K-fold validataion을 사용하였고, validataion set도 traindata의 분포를 띄도록 뽑아주었다. 

   20퍼센트를 validation으로 사용하였고, 실제 제출시에는 training에 전체를 사용하여 validation set을 만들지 않고, 경험적인 지표들의 변화로 학습을 중단시켜 최종 model을 확정시켰습니다.
   
6. Ensemble

   Competition의 꽃인 앙상블 입니다.
   이전까지 앙상블의 효과에 대해 의구심을 품고있었으나, 성능이 올랐다는 캠퍼분들의 말씀을 듣고 간단한 soft voting을 진행하였습니다.

   - Resnext로 학습한 model   (가중치 : 0.2)

   - Efficientnet-b4로 학습한 model 

     1) image scale = (128, 96) 가중치 0.4  

     2) image scale = (256,192) 가중치 0.4

   이두가지를 model의 결과에 argmax를 취해주기 전 결과끼리 가중치를 곱해주고 모두 더한 이후 argmax를 시켜주었습니다. 이중  efficientnet의 성능이 더 높았기때문에 가중치를 0.4씩으로 주고 더하였습니다. 

   결과적으로 약간의 LB score향상을 얻어냈습니다.
   



쓰고나서 보니까 조금 허무했습니다. 힘을들여 augmentation을 구현하고 실험하는 과정도 해보고, augmentation해서 class 분포도 균일하게 맞춰보고, 여러 버전 만들고, scheduler써보고, custom loss 구현해보고, 외부데이터 써보고 별짓을 다했지만 효과가 없었던 실험이 많았던것 같습니다. 

하지만 좋은 python code로 block들이나 arg들만 바꾸어서 다양한 실험을 해볼수있는 환경을 구축해보았다는 경험과, 여러가지를 시도해봤다는거 자체에 의미를 두고 좋은 competition이였다고 회고해봅니다.



---

code 

notebook file : 간단한 실험용도로 학교서버에서 돌릴수있는 환경으로 구축
python files : 터미널에서 명령창에 arg들을 집어 넣어서 train and inference

​	