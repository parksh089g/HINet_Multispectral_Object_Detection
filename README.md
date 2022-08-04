# HINet_Multispectral-Object-Detection

## Intro

Baseline : https://github.com/DocF/multispectral-object-detection 

Code for High-Frequency Interchange Network for Multispectral Object Detection

Multispectral Object Detection with MHT and Yolov5

## Abstract
* 요  약 

RGB 이미지를 활용하는 다양한 객체 인식 분야에서 조도가 어둡거나 특정 물체에 의해 가려진 환경에서의 RGB 이미지는 객체 인식 성능 저하를 일으킨다. IR 이미지는 가시광선이 아닌 적외선 파동을 감지하기 때문에 이러한 환경에서 강인한 객체 인식 성능을 가질 수 있고, RGB-IR 이미지 쌍을 가지고  각자의 강점을 결합 하는 것을 통해 객체 인식 성능을 향상시킬 수 있다. 본 논문에서는 RGB-IR 이미지 쌍의 강점만을 결합하여 객체 인식 성능을 향상시키는 다중 스펙트럼 융합 모델인 high-frequency interchange network (HINet)을 제안한다. HINet은 RGB-IR 이미지 간 주요 정보를 교환하기 위해 두 가지 객체 인식 모델을 mutual high-frequency transfer (MHT)를 이용하여 연결하였다. MHT에서는 RGB-IR 이미지 쌍 각각을 discrete cosine transform (DCT) 스펙트럼 도메인으로 변환하여 고주파 정보를 추출한다. 추출된 고주파 정보는 서로의 네트워크에 전달되어 객체 인식성능 향상을 위해 활용되어 진다. 실험 결과는 제안하는 네트워크의 우수성을 보이며 다중 스펙트럼 객체 인식 성능을 향상시키는 것을 확인할 수 있다.

* ABSTRACT 

Object recognition is carried out using RGB images in various object recognition studies. However, RGB images in dark illumination environments or environments where target objects are occluded other objects cause poor object recognition performance. On the other hand, IR images provide strong object recognition performance in these environments because it detects infrared waves rather than visible illumination. In this paper, we propose an RGB-IR fusion model, high-frequency interchange network (HINet), which improves object recognition performance by combining only the strengths of RGB-IR image pairs. HINet connected two object detection models using a mutual high-frequency transfer (MHT) to interchange advantages between RGB-IR images. MHT converts each pair of RGB-IR images into a discrete cosine transform (DCT) spectrum domain to extract high-frequency information. The extracted high-frequency information is transmitted to each other's networks and utilized to improve object recognition performance. Experimental results show the superiority of the proposed network and present performance improvement of the multispectral object recognition task. 

### Overview
![](https://velog.velcdn.com/images/parksh089g/post/219a86ff-83e7-4d2c-b09e-54947dbd9667/image.png)![](https://velog.velcdn.com/images/parksh089g/post/d77f2695-005f-4dc4-8223-dd21852c6ce3/image.png)


## Citation
 -


## Installation 
Python>=3.6.0 is required with all requirements.txt installed including PyTorch>=1.7 (The same as yolov5 https://github.com/ultralytics/yolov5 ).

The same as https://github.com/DocF/multispectral-object-detection

## Dataset
-[FLIR] A new aligned version.

-[LLVIP]

The same as https://github.com/DocF/multispectral-object-detection

## Run
#### Download the pretrained weights

MHT weights 

-[FLIR]

-[LLVIP]

#### Change the data cfg
some example in data/multispectral/

#### Change the model cfg
some example in models/transformer/

xxxx_transfomerx3_FLIR_aligned_dct_3.yaml
xxxx_transfomerx3_FLIR_aligned_dct_res.yaml
xxxx_transformerx3_llvip_dct.yaml

### Train Test and Detect
train: ``` python train.py```

test: ``` python test.py```

detect: ``` python detect_twostream.py```

The same as https://github.com/DocF/multispectral-object-detection

## Results

![](https://velog.velcdn.com/images/parksh089g/post/1c74ea77-239f-42b5-9a89-5874ccf40722/image.png)![](https://velog.velcdn.com/images/parksh089g/post/070062fd-5422-4ec9-a861-f3c46616c422/image.png)




#### References

https://github.com/ultralytics/yolov5

https://github.com/DocF/multispectral-object-detection
  
