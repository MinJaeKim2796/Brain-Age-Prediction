# Brain-Age-Prediction

# *Assessing the Impact of Intensity Normalization in CNN based Brain Age Prediction*


KHU

[*Software Convergence Capstone Design*] *Spring 2022*





### 1. 과제 개요
  뇌의 노화 과정은 인지 능력의 저하인 인지 노화(Cognitive Ageing)에 따라 발생하며, 나이가 들수록 알츠하이머와 같은 퇴행성 신경질환의 위험이 증가한다. 의료계에서 뇌연령(Brain Age)을 직접 구하는 것은 어려운 일이기 때문에, 일반적으로 건강한 사람의 뇌연령과 생물학적 나이(Chronological Age)가 같다는 가정하에 뇌연령 예측을 진행한다. 따라서, 건강한 사람의 데이터로 학습한 후에 뇌 질환을 겪는 사람의 뇌연령을 측정함으로써 뇌 노화의 정도를 파악할 수 있다. MRI 뇌 데이터를 바탕으로 예측한 뇌 연령(Brain-Predicted Age)과 생물학적 나이의 차이 값을 'Brain Age Gap'이라 하며, 이 값이 클수록 뇌 노화가 심각하다고 알려져 있다. 그러므로 Brain Age Gap은 뇌의 노화에 관한 상태를 파악할 수 있는 바이오마커로 활용될 수 있다. 본 과제에서는 건강한 사람의 T1 weighted MRI 데이터를 사용하여 3D-CNN 모델을 통해 뇌연령을 예측함으로써, 의료계에 도움을 주고자 한다.

  해당 과제에서는 18~88세의 건강한 사람의 MRI로 구성된 CAMCAN(Cambridge Centre for Ageing and Neuroscience) Dataset을 사용한다. Raw Data는 192 x 256 x 256(Voxel Size : 1mm)의 크기로, 전처리(Preprocessing)의 정도에 따른 3D-CNN 모델의 뇌연령 예측 성능을 비교한다.



### 2. 과제 수행 내용
![image](https://user-images.githubusercontent.com/59433841/174036490-52586662-b916-4fe4-aa5e-af893c20fc8b.png)

  1) Minimal Preprocessed : Raw 데이터로부터 뇌의 조직만 남기기 위해 HD-BET[1]라는 Tool을 이용하여 Skull-Stripping(Brain Extraction)을 수행한다.
  2) Fully Preprocessed : CAT12라는 Tool을 이용하여 Skull-Stripping뿐만 아니라, 자기장 영역 불균일성 보정(Magnetic Field Inhomogeneity Correction) 및 정합(Registration) 등을 수행한다. 뇌는 회백질(Gray Matter; GM), 백색질(White Matter; WM), 그리고 뇌척수액(Cerebrospinal fluid; CSF)으로 구성된다. Scanner 내부의 자기장은 일정해야 한다. 하지만, 뇌 조직과 자기장이 만나면 자기장의 크기는 감소하며, 뇌 조직의 유형에 따라 감소하는 비율이 다르므로 비정상적으로 밝거나 어두운 영역이 나타나는 자기장 불균일성이 발생한다. 이러한 문제를 해소하고자 Magnetic Field Inhomogeneity Correction을 적용한다. Registration은 서로 다른 기하학적 공간을 가진 영상들을 동일한 공간으로 맞춰주는 프로세스를 의미하고, 모델 학습 중 데이터의 좌표에 따라 미치는 영향을 최소화하기 위해 적용한다. 추가로, Registration에서는 Resizing을 포함하여 데이터 크기를 121 x 145 x 121(Voxel Size : 1.5mm)가 되도록 Downsampling을 한다.

  3) Intensity Normalization : 해당 과제에서는 하나의 Cohort에서만 수집된 데이터이나, 일반적으로 Cohort마다 MRI를 촬영하는 장비는 각각 다르므로 Pulse Sequence, Scan Parameter 등의 차이로 Site Effect가 발생할 수 있다. 이를 방지하기 위해 해당 과제에서는 Pre-Processing(Minimal, Fully)된 데이터를 기반으로 각각 Z-Score, FCM(Fuzzy C-Means), KDE(Kernel Density Estimation), White Stripe, Least Squares, 그리고 Histogram Matching인 총 6가지의 Intensity Normalization을 수행한다.



### 3. 모델 및 학습
![image](https://user-images.githubusercontent.com/59433841/174036777-41be4d2f-16bf-4bd4-8950-e3f7a275c2c8.png)

  전처리한 데이터(총 14종류)를 입력으로 하여, 예측한 Brain Age를 출력하게 된다. 모델은 총 5개의 Convolution Layer로 구성되며, VGGNet과 유사한 구조를 지닌다. 모든 Block은 Convolution 연산(Kernel Size : 3 x 3 x 3), Batch Normalization, 활성화 함수 Rectified Linear Unit(ReLU), Convolution 연산, Batch Normalization, ReLU, Max Pooling으로 구성된다. 이후, Fully Connected Layer를 거쳐 뇌 연령을 예측하게 된다. Out Channel의 수는 첫 Block부터 8, 16, 32, 64, 128이 된다.
  CAMCAN Dataset을 Train Set(N = 500)과 Test Set(N = 101)으로 분리하여 진행하였다. 그 중, Train Set의 20%를 Validation Set으로 두어, Validation MAE가 가장 낮을 때를 Optimal Model로 선택했다. 모델 학습은 PyTorch 환경에서 진행되었으며, Adam Optimizer, MAE Loss Function, Learning Rate 0.01(StepLR Scheduler gamma = 0.97), Batch Size 8을 적용하였다.



### 4. 결과
  1) Minimal Preprocessed with Intensity Normalization(Voxel Size : 1mm)


![image](https://user-images.githubusercontent.com/59433841/174036957-57447c8c-ea09-4bac-9dc3-809493fcf40c.png)


  2) Fully Preprocessed with Intensity Normalization(Voxel Size : 1.5mm)


![image](https://user-images.githubusercontent.com/59433841/174037033-4bce5e6f-ed56-45f9-bd3a-f41c05e11017.png)



### 5. 결론 및 제언
  해당 과제에서는 MRI 데이터의 전처리 정도에 따라 예측한 뇌연령과 생물학적 나이에 대한 MAE와 R을 도출함으로써 전처리 정도에 따른 뇌연령 예측 모델의 성능을 비교하였다. MRI 데이터는 상당히 오랜 전처리 시간이 필요하므로 전처리 과정을 최소화하고자 하였으나, Fully Preprocessed 데이터를 바탕으로 Kernel Density Estimation 기법을 적용한 Intensity Normalization을 수행함으로써 가장 높은 성능(MAE = 7.06, R = 0.90)을 도출하였다. 해당 과제에서는 Intensity Normalization에 따라 3D-CNN의 성능을 정량적으로 제시하였다. 추후 Learning Rate나 Optimizer 등 다양한 Parameter를 적용하거나 다른 3D-CNN 모델을 활용함으로써 뇌연령 예측 모델의 성능 개선에 활용될 수 있다.
  
  
  
### Reference  
[1] https://github.com/MIC-DKFZ/HD-BET

[2] http://www.neuro.uni-jena.de/cat/

[3] Jacob C. Reinhold, Blake E. Dewey, Aaron Carass, and Jerry L. Prince. Evaluating the Impact of Intensity Normalization on MR Image Synthesis, 2018.

[4] James H. Cole, Rudra P.K. Poudel, Dimosthenis Tsagkrasoulis, Matthan W.A. Caan, Claire Steves, Tim D. Spector, and Giovanni Montana. Predicting brain age with deep learning from raw imaging data results in a reliable and heritable biomarker, 2017.
