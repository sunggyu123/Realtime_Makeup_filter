# Realtime_Makeup_filter
Realtime Makeup filter using opencv

> Requirements
>   > - Webcam
>   > - opencv 3.4
>   > - dlib
>   > - tensorflow 1.15
>   > - tensorflow-gpu 1.15
>   > - pretrained haar_cascade_frontalface model
>   > - pretrained MTCNN model
>   > - pretrained shape predicter 68 face landmarks
>   > - beautyGAN model(Our fine tuned model not uploaded)

> 주제 선정 이유
>   > - 작년을 기점으로 코로나 유행이 시작된 후로 학교, 기업 등의 여러 단체에서 비대면 화상 회의 방식으로의 업무 및 수업 횟수가 증가로 저희가 진행하면서 느낀 것은 화상 강의를 진행할 때 얼굴 노출을 부담스러워 하여 마스크를 착용하거나 모자를 푹 눌러쓴 경우 혹은 아예 캠을 키지않는 상황이 발생하는 걸 발견
>   > - 이러한 문제들을 해결하기 위해 사진 어플에서 많이 사용하고 있는 얼굴 필터 기능을 추가하면 편리할 것으로 예상

> 개발하고자 하는 서비스 내용
>   > - 이와 같은 문제를 해결하고자 저희 팀은 실시간 얼굴 필터 적용 서비스를 개발
>   > - 주요 특징으로는 다수 얼굴 검출이 가능하고, 학습된 모델 사용으로 인해 보다 더 자연스러운 필터 적용, 입력을 통한 사용자가 원하는 스타일 적용, 실시간 화면 적용 

> 시나리오
>   > - 먼저 실시간 카메라를 사용하여 사용자의 얼굴 검출을 하게 되고 검출된 얼굴에서 눈,코,입 등의 특징들을 추출하게 됩니다. 이후 학습을 진행하기 위해 검출된 이미지 전처리 과정이 진행되고, 이전에 추출한 얼굴특징들과 전처리된 이미지를 BeautyGAN모델에 적용시켜 실시간 웹캠 화면에 메이크업 필터가 적용되는 결과물이 나오게 됩니다.

> 얼굴 검출
>   > - 다수 얼굴 검출이 가능하며, 정면, 측면, 악세사리 착용 모습 그리고 얼굴이 화면에 짤려서 나오는 경우에도 검출이 가능했던 모델은 MTCNN이어서 이 모델을 최종적으로 선정
>   > - 저희가 원하는 조건에서 잘 검출되는 것을 확인할 수 있었으며, 마스크와 모자 같은 악세사리를 착용한 경우에서 또한 검출이 가능

> BeautyGAN 모델
>   > <img width="1081" alt="beautyGAN" src="https://user-images.githubusercontent.com/49279776/143253608-d0a7c1cd-723c-4f80-83a8-eeee26caf929.png">

> Vgg16 vs Vgg19
>   > - 기존 BeautyGAN모델은 generator의 분류기 부분에서 학습된 vgg16 모델의 일부를 사용하여 구현이 되지만, 저희는 vgg19 모델을 사용하면 더 좋은 성능을 가져올 거라고 판단하여 전이 학습을 통해 저희 프로젝트에 맞게 재정의 

> BeautyGAN 성능
>   > - 아래 그림은 기존 모델의 성능 지표
>   >   > ![original](https://user-images.githubusercontent.com/49279776/143255012-4a6033df-0395-4381-a34c-f87dbb1a3f83.PNG)

>   > - 아래 그림은 vgg16모델을 vgg19모델로 변형시킨 모델의 성능 지표
>   >   >![transfromed](https://user-images.githubusercontent.com/49279776/143254064-aeeaf1e2-85e8-48ac-904e-e20a6ced7894.PNG)

> 최종 결과물 모습 기능
>   > - 따라서 저희는 BeautyGAN 모델의 출력값으로 나온 이미지를 실시간 웹캠 화면에 적용시키고자 이미지 합성을 통해 진행
> 유투브 링크 
> https://www.youtube.com/watch?v=D79aV15hnMc
