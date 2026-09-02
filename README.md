# 방한 외국인 관광객 실태조사 대시보드

KTO 외래관광객조사 마이크로데이터(2019, 2023~2025) 기초통계 탐색용 Streamlit 대시보드.

## 데이터 준비 (필수)

이 저장소에는 응답자 단위 원자료(CSV)가 포함되어 있지 **않습니다** (라이선스/재배포 제한). 실행 전 아래 두 파일을 `data/` 폴더에 직접 넣어주세요.

```
dashboard/
  data/
    pre_covid.csv    # 2019 + 2023~2025 통합본
    post_covid.csv   # 2023~2025 통합본
```

필요한 컬럼: `survey_year, pnid, weight, D_NAT, D_SEX, D_AGE, D_NUM, D_GUB, TYP, Q1, M박HAP, M일HAP, 총액1인TOT2` 등 (자세한 컬럼 목록은 `codebook/` 참고).

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 구성

- `app.py` — 3개 탭(데이터 원본 / PreCovid·PostCovid 비교 / 변수 탐색: 분포·상관관계·군집분석)
- `data_utils.py` — 데이터 로딩, 가중통계 계산, 코드북 라벨 매핑
- `codebook/` — 변수 정의서·코드값 매핑 워크북 (응답 데이터 아님, 메타데이터만 포함)
- `data/` — 원자료 CSV 위치 (git에는 포함 안 됨, `.gitignore` 처리)

## Streamlit Community Cloud 배포 시 참고

무료 플랜은 저장소에 없는 파일(원자료)을 자동으로 가져오지 않습니다. 배포하려면 아래 중 한 가지가 필요합니다.

- Streamlit Cloud의 [파일 업로드/외부 스토리지 연동](https://docs.streamlit.io) (S3, GCS 등)으로 `data_utils.py`의 `load_csv`를 원격 경로를 읽도록 수정
- 또는 비공개 데이터 저장 방식(Git LFS + private 저장소, private 배포 등)

가중치(`weight`) 계산 방식, 변수 설명 등은 `codebook/` 워크북과 프로젝트 대화 기록을 참고하세요.
