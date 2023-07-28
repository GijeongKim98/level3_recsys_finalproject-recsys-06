# 취준생을 위한 개인화된 알고리즘 문제 및 깃허브 저장소 추천

## 소개

## 파일 설명

### crawling

설명: 크롤링용 코드로 각각 문제, 유저, interaction, 크롤링된 파일을 업데이트

- log 폴더와 파일을 자동 생성

주의:
- interaction에는 유저 데이터, 업데이트에는 유저와 interaction 데이터 필요

### data_manager

설명: 크롤링 후의 데이터를 학습에 사용되도록 처리

- change_data.py를 실행해서 크롤링한 raw_data를 모델에 입력되는 processed_data로 변경

주의:
- setting.yaml에서 raw_data_folder_path 파일 생성과 그 파일 내에 user_data_file, item_data_file, inter_data_file 파일 필요

### mysql

설명: 처리된 데이터를 MySQL로 업로드

- setting.yaml의 host, port, user, password로 DB에 접속 (password는 직접 입력)
- 각 table에 대하여 insert_file_list를 사용하여 프로젝트 생성

주의:
- setting.yaml에 data_path의 이름을 가지는 폴더 생성 필요
- 처음으로 사용시 create_db, create_table을 True로 설정 후 사용, 다음 실행에는 False로 설정
- insert_table_name과 insert_file_list의 값은 위치에 맞게 입력 (현제 yaml 파일에서의 예: insert_file_list의 첫 데이터는 problem table에 입력)
- insert_table_name은 수정 필요 X

### fast-api

설명: fast-api을 설정

- 이미 inference된 모델을 불러와서 유저의 요청이 들어오면 빠르게 추천 문제 출력
- MySQL 서버에 접속하여 빠르게 기본 데이터 출력

주의:
- data 파일에 필요 파일, model에는 .pt 파일 추가 필요

