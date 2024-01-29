# LightGCN 코드 설명

## LGCN 설계 이유

사용자의 알고리즘 별 지식을 추정하도록 모델링하고 LightGCN에서 생성된 임베딩 벡터를 활용한 개인화된 추천 서비스 구축를 위함

## 각 디렉터리별 설명

- datasets : 
    - graph_base_dataset.py : graph관련 모델의 학습을 위한 data set 생성

- model :
    - model.py : lgcn 모델 생성 및 불러오기 + 다른 모델 추가 가능

- trainer :
    - loss_function.py : Loss function 설정
    - metric.py : Metric 설정
    - oprimizer.py : Optimizer 설정
    - trainer.py : train, test 관련 파일

- utils :
    - setting.yaml : 모델의 하이퍼 파라미터 수정, 데이터 파일의 경로 수정
    - util.py
        - 모니터링을 위한 로그 생성 및 저장 코드
        - Negative Sampling

- dataset.py : load dataset
- inference.py : Inference, test 데이터의 예측 결과 저장
- train.py : 모델 학습
- preprocess : 데이터 전처리 : train, valid 분류 및 Negative sampling


## 최종 성능
- LightGCN: AUC : 0.831 // $TN \over {TN + FP}$ : 0.736
