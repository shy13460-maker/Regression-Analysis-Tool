# 선형·비선형 회귀 분석 프로그램

CSV 또는 Excel 형식의 실험 데이터를 입력받아 **선형 회귀와 다양한 비선형 회귀모델을 적용하고, 회귀곡선·잔차·성능지표를 비교할 수 있는 웹 기반 분석 프로그램**입니다.

Python과 Streamlit으로 개발하였으며, 브라우저에서 데이터를 업로드한 뒤 독립변수(X)와 종속변수(Y)를 선택하고 원하는 회귀모델을 실행할 수 있습니다. 분석 결과는 그래프와 표로 확인할 수 있으며, Excel 파일과 HTML 그래프로 저장할 수 있습니다.

---

## 1. 프로그램 목적

실험 및 공학 데이터는 항상 직선 형태의 관계를 가지지 않으므로, 하나의 선형회귀만으로는 데이터의 경향을 충분히 설명하지 못하는 경우가 있습니다.

이 프로그램은 하나의 데이터에 여러 회귀모델을 동시에 적용하여 다음을 확인할 수 있도록 제작되었습니다.

- 선형회귀 결과 확인
- 비선형 형태의 회귀곡선 확인
- 여러 모델의 적합도 비교
- 잔차를 이용한 오차 확인
- 이상치 후보 확인
- 회귀식을 이용한 새로운 X 값의 Y 예측
- 분석 결과 저장

---

## 2. 주요 기능

### 데이터 입력
- CSV 파일 지원
- Excel `.xlsx` 파일 지원
- Excel `.xls` 파일 지원
- Excel 파일의 시트 선택 가능
- 데이터의 열 제목(Header row) 직접 지정 가능
- 원본 데이터 미리보기 제공

### 변수 선택
- X 열: 독립변수
- Y 열: 종속변수
- Y 오차 열: 선택 사항
- 사례/그룹 열: 선택 사항

### 회귀모델
- 선형 회귀 (Linear)
- 2차 다항 회귀 (Quadratic)
- 3차 다항 회귀 (Cubic)
- 지수 회귀 (Exponential)
- 로그 회귀 (Logarithmic)
- 거듭제곱 회귀 (Power)
- 4-모수 로지스틱 회귀 (Logistic 4P)

### 분석 결과
- 회귀곡선 비교 그래프
- 95% 신뢰구간
- 잔차 그래프
- 이상치 후보 표시
- 회귀식 표시
- 모델별 파라미터 표시
- 입력한 X 값에 대한 Y 예측
- 분석 결과 Excel 다운로드
- 회귀 그래프 HTML 다운로드

### 모델 평가 지표
- R²
- Adjusted R²
- RMSE
- MAE
- SSE
- AIC
- AICc
- BIC

---

## 3. 개발 환경

권장 환경은 다음과 같습니다.

| 항목 | 권장 환경 |
|---|---|
| 운영체제 | Windows 10 / Windows 11 |
| Python | Python 3.10 ~ 3.12 권장 |
| 브라우저 | Chrome, Edge 등 최신 브라우저 |
| 실행 방식 | Streamlit 웹 애플리케이션 |

Python이 설치되어 있는지 다음 명령으로 확인할 수 있습니다.

```bash
python --version
```

pip 설치 상태는 다음 명령으로 확인합니다.

```bash
python -m pip --version
```

---

## 4. 필요한 Python 라이브러리

프로그램 실행에 필요한 외부 라이브러리는 `requirements.txt`에 정리되어 있습니다.

```txt
streamlit>=1.42
numpy>=1.24
pandas>=2.0
scipy>=1.10
plotly>=5.20
openpyxl>=3.1
xlrd>=2.0
```

각 라이브러리의 역할은 다음과 같습니다.

| 라이브러리 | 용도 |
|---|---|
| `streamlit` | 프로그램의 웹 UI 구성, 파일 업로드, 버튼, 표, 다운로드 기능 |
| `numpy` | 수치 계산, 배열 연산, 다항식 회귀 계산 |
| `pandas` | CSV/Excel 데이터 읽기, 열 선택, 데이터 전처리, 결과 표 생성 |
| `scipy` | 비선형 회귀 최적화(`curve_fit`), t 분포 기반 신뢰구간 계산 |
| `plotly` | 회귀 그래프와 잔차 그래프를 인터랙티브하게 표시 |
| `openpyxl` | `.xlsx` Excel 파일 읽기/쓰기 및 분석 결과 Excel 저장 |
| `xlrd` | 구형 `.xls` Excel 파일 읽기 |

다음 모듈도 코드에서 사용하지만 Python 기본 라이브러리이므로 별도 설치가 필요하지 않습니다.

- `io`
- `math`
- `dataclasses`
- `typing`

---

## 5. 라이브러리 설치 방법

### 방법 A. requirements.txt 사용

프로그램 폴더에서 명령 프롬프트(CMD)를 열고 다음 명령을 실행합니다.

```bash
python -m pip install -r requirements.txt
```

설치가 완료되면 Streamlit, NumPy, Pandas, SciPy, Plotly 등이 자동으로 설치됩니다.

### 방법 B. 직접 설치

`requirements.txt`를 사용하지 않는 경우 다음 명령으로 설치할 수 있습니다.

```bash
python -m pip install streamlit numpy pandas scipy plotly openpyxl xlrd
```

---

## 6. 가상환경 사용 권장

기존에 설치되어 있는 다른 Python 패키지와 버전 충돌이 발생할 수 있으므로, 가능하면 가상환경을 사용하는 것을 권장합니다.

Windows CMD 기준:

```bash
python -m venv .venv
```

가상환경 활성화:

```bash
.venv\Scripts\activate
```

pip 업데이트:

```bash
python -m pip install --upgrade pip
```

필요 라이브러리 설치:

```bash
python -m pip install -r requirements.txt
```

가상환경을 사용하면 다른 프로젝트에서 사용하는 라이브러리와 충돌할 가능성을 줄일 수 있습니다.

---

## 7. 프로그램 실행 방법

프로그램이 있는 폴더에서 다음 명령을 실행합니다.

```bash
python -m streamlit run Regression_Analyzer_Web.py
```

정상적으로 실행되면 브라우저에서 프로그램이 열립니다.

일반적으로 로컬 주소는 다음과 같습니다.

```text
http://localhost:8501
```

프로그램을 종료하려면 실행 중인 CMD 창에서 다음 키를 누릅니다.

```text
Ctrl + C
```

---

## 8. 데이터 준비 방법

가장 단순한 데이터 구조는 다음과 같습니다.

| Speed_kn | Bubble_Area_Percent |
|---:|---:|
| 5 | 2.1 |
| 10 | 3.8 |
| 15 | 6.4 |
| 20 | 9.5 |
| 25 | 13.8 |

첫 번째 열을 X, 두 번째 열을 Y로 사용할 수 있습니다.

### X 열
독립변수입니다.

예:
- 속도
- 시간
- 온도
- 압력
- RPM
- 유량

### Y 열
종속변수입니다.

예:
- 저항
- 기포 면적비율
- 소비전력
- 온도 변화
- 효율

### Y 오차 열
선택 사항입니다.

표준편차나 측정오차가 있는 경우 그래프의 **오차막대(Error bar)** 로 표시할 수 있습니다.

> 현재 프로그램에서 Y 오차 열은 그래프 표시용이며, 회귀계수를 계산할 때 가중치로 사용하지 않습니다.

### 사례/그룹 열
여러 실험 Case를 한 파일에 저장한 경우 사용할 수 있습니다.

예:

| Case | Speed_kn | Resistance_N |
|---|---:|---:|
| A | 10 | 100 |
| A | 20 | 250 |
| B | 10 | 120 |
| B | 20 | 280 |

`Case` 열을 사례/그룹으로 선택하면 A와 B를 각각 독립적으로 회귀분석합니다.

---

## 9. 프로그램 사용 순서

1. 프로그램 실행
2. CSV 또는 Excel 파일 업로드
3. Excel인 경우 시트 선택
4. 열 제목이 위치한 행(Header row) 확인
5. X 열 선택
6. Y 열 선택
7. 필요하면 Y 오차 열 선택
8. 필요하면 사례/그룹 열 선택
9. 분석할 회귀모델 선택
10. `회귀 분석 실행` 버튼 클릭
11. 회귀 그래프 확인
12. 잔차 그래프 확인
13. 모델 결과 표 확인
14. 이상치 후보 확인
15. 필요하면 X 값을 입력하여 Y 예측
16. Excel 결과 또는 HTML 그래프 다운로드

---

## 10. 회귀모델 설명

### 10.1 선형 회귀

선형 회귀식:

```text
y = ax + b
```

X와 Y 사이의 관계를 직선으로 근사합니다.

프로그램에서는 최소제곱법을 이용하여 관측값과 회귀선 사이의 오차 제곱합이 작아지도록 계수를 계산합니다.

---

### 10.2 2차 다항 회귀

```text
y = ax² + bx + c
```

데이터가 단순 직선보다 곡선 형태를 보일 때 사용할 수 있습니다.

---

### 10.3 3차 다항 회귀

```text
y = ax³ + bx² + cx + d
```

2차 함수보다 더 복잡한 곡률 변화가 있는 데이터를 표현할 수 있습니다.

---

### 10.4 지수 회귀

```text
y = a·exp(bx) + c
```

X가 증가함에 따라 Y가 빠르게 증가하거나 감소하는 데이터에 사용할 수 있습니다.

---

### 10.5 로그 회귀

```text
y = a·ln(x) + b
```

X가 증가할수록 변화량이 점차 감소하는 형태의 데이터에 사용할 수 있습니다.

**주의:** 로그 함수의 특성상 X는 0보다 커야 합니다.

---

### 10.6 거듭제곱 회귀

```text
y = a·xᵇ
```

스케일 법칙이나 공학 데이터의 비선형 관계를 표현할 때 사용할 수 있습니다.

**주의:** 현재 프로그램에서는 X가 0보다 큰 데이터에서 사용합니다.

---

### 10.7 4-모수 로지스틱 회귀

기본 형태:

```text
y = lower + (upper - lower) /
    (1 + exp(-slope·(x - midpoint)))
```

S자 형태의 데이터에 사용할 수 있습니다.

주요 파라미터:
- `lower`: 하한값
- `upper`: 상한값
- `midpoint`: 중앙 위치
- `slope`: 곡선 기울기

---

## 11. 회귀 계산 방법

### 선형/2차/3차 다항 회귀
NumPy의 다항식 최소제곱 적합을 사용합니다.

즉, 관측값과 모델 예측값의 차이인 잔차의 제곱합이 최소가 되도록 계수를 계산합니다.

### 지수/로그/거듭제곱/Logistic 4P
SciPy의 `curve_fit`을 사용하여 모델 파라미터를 반복적으로 조정하는 비선형 최소제곱 최적화를 수행합니다.

따라서 프로그램은 하나의 회귀방식만 사용하는 것이 아니라, 모델의 수학적 형태에 맞는 적합 방법을 사용합니다.

---

## 12. 평가 지표 설명

### R² — 결정계수

모델이 Y 데이터의 변동을 얼마나 설명하는지를 나타냅니다.

일반적으로 1에 가까울수록 데이터에 잘 맞습니다.

단, R²가 높다고 해서 항상 가장 좋은 모델이라는 의미는 아닙니다. 모델이 지나치게 복잡하면 과적합이 발생할 수 있습니다.

### Adjusted R² — 수정 결정계수

모델의 변수 또는 파라미터 수를 고려해 R²를 보정한 값입니다.

복잡한 모델이 단순히 파라미터가 많다는 이유로 유리해지는 문제를 일부 보완합니다.

### RMSE — Root Mean Squared Error

잔차 제곱의 평균에 제곱근을 적용한 값입니다.

```text
RMSE = sqrt(mean((Y - Ŷ)²))
```

값이 작을수록 관측값과 예측값의 차이가 작습니다.

### MAE — Mean Absolute Error

잔차 절댓값의 평균입니다.

```text
MAE = mean(|Y - Ŷ|)
```

값이 작을수록 좋습니다.

### SSE — Sum of Squared Errors

잔차 제곱합입니다.

```text
SSE = Σ(Y - Ŷ)²
```

작을수록 관측 데이터에 잘 적합된 것입니다.

### AIC — Akaike Information Criterion

모델의 적합도와 복잡도를 함께 고려하는 지표입니다.

**낮을수록 좋은 모델**로 해석합니다.

### AICc

표본 수가 많지 않을 때 AIC를 보정한 값입니다.

이 프로그램에서는 가능한 경우 **AICc가 가장 낮은 모델을 최적 모델로 표시**합니다.

AICc 계산이 불가능한 경우 AIC를 사용합니다.

### BIC — Bayesian Information Criterion

AIC와 비슷하지만 모델 복잡도에 더 큰 패널티를 주는 지표입니다.

마찬가지로 낮을수록 좋습니다.

---

## 13. 잔차 분석

잔차는 다음과 같이 정의합니다.

```text
잔차 = 관측값 - 예측값
```

잔차가 0에 가까우면 해당 점에서 모델의 예측값과 실제 관측값이 비슷하다는 뜻입니다.

잔차 그래프에서는 예측 Y를 X축, 잔차를 Y축으로 표시합니다.

좋은 회귀모델이라면 잔차가 0을 중심으로 특정한 패턴 없이 분포하는 것이 일반적으로 바람직합니다.

---

## 14. 이상치 후보 판정

프로그램은 표준화 잔차를 이용하여 이상치 후보를 표시합니다.

현재 기준:

```text
|표준화 잔차| ≥ 2
```

즉 표준화 잔차의 절댓값이 2 이상인 데이터는 이상치 후보로 표시됩니다.

중요한 점은 **이상치 후보 표시가 해당 데이터를 자동으로 삭제한다는 뜻이 아니라는 것**입니다.

실험 오류, 센서 문제, 실제 물리적 현상 등 다양한 원인이 있을 수 있으므로 사용자가 원본 데이터를 확인한 뒤 판단해야 합니다.

---

## 15. 95% 신뢰구간

최적 회귀모델에 대해 95% 신뢰구간을 표시할 수 있습니다.

프로그램은 적합된 파라미터의 공분산과 t 분포를 이용하여 회귀곡선 주변의 파라미터 불확실성을 근사합니다.

신뢰구간이 좁을수록 해당 구간에서 회귀곡선 추정의 불확실성이 상대적으로 작다는 의미로 해석할 수 있습니다.

---

## 16. 결과 다운로드

### Excel 결과
`분석 결과 Excel 다운로드` 버튼으로 회귀분석 결과를 `.xlsx` 파일로 저장할 수 있습니다.

저장 파일에는 다음과 같은 정보가 포함됩니다.

- 모델 요약
- 모델 파라미터
- 전체 예측값
- 최적 모델 잔차
- 입력 데이터
- 분석 실패 모델 정보(해당 시)

### HTML 그래프
회귀 그래프는 HTML 형식으로 저장할 수 있습니다.

HTML 그래프는 Plotly 기반이므로 브라우저에서 확대, 축소, 마우스 오버 정보 확인 등이 가능합니다.

---

## 17. 예측 기능

회귀분석이 완료된 후 새로운 X 값을 입력하면 최적 모델을 이용하여 해당 X에서의 Y 값을 계산할 수 있습니다.

단, 실험 데이터 범위를 크게 벗어난 값을 입력한 경우 **외삽(Extrapolation)** 이 되므로 실제 현상과 차이가 커질 수 있습니다.

---

## 18. 프로젝트 파일 구성

```text
Regression-Analysis-Tool/
│
├── Regression_Analyzer_Web.py
├── requirements.txt
├── README.md
└── sample_data.csv
```

### Regression_Analyzer_Web.py
실제 Streamlit 회귀분석 프로그램입니다.

### requirements.txt
프로그램 실행에 필요한 외부 Python 라이브러리 목록입니다.

### README.md
현재 문서입니다.

### sample_data.csv
프로그램 테스트에 사용할 수 있는 예제 데이터입니다.

---


---

## 예제 데이터 출처

`sample_data.csv`는 임의로 만든 데이터가 아니라 **UCI Machine Learning Repository의 Yacht Hydrodynamics 공개 데이터셋**에서 가져온 실제 공개 실험 데이터의 일부입니다.

- 데이터셋명: **Yacht Hydrodynamics**
- 원 연구 데이터: Delft Ship Hydromechanics Laboratory
- 전체 실험 수: 308개
- 선형/비선형 회귀에 사용할 수 있는 Regression 데이터셋
- 라이선스: **CC BY 4.0**
- DOI: **10.24432/C5XG7R**

원 데이터는 여러 요트 선형(hull form)에 대해 선형 형상계수와 Froude number를 변화시키면서 **배수량 단위중량당 잉여저항(residuary resistance per unit weight of displacement)** 을 측정한 데이터입니다.

이 저장소의 `sample_data.csv`는 그중 **하나의 동일한 선형 조건**을 선택하여 Froude number가 변할 때 잉여저항이 어떻게 변화하는지를 회귀분석하기 쉽게 정리한 예제입니다.

### 예제 파일 열 설명

| 열 이름 | 의미 |
|---|---|
| `LCB_Position` | 부심의 종방향 위치, 무차원 |
| `Prismatic_Coefficient` | 주형계수(Prismatic coefficient), 무차원 |
| `Length_Displacement_Ratio` | 길이-배수량 비, 무차원 |
| `Beam_Draught_Ratio` | 폭-흘수 비, 무차원 |
| `Length_Beam_Ratio` | 길이-폭 비, 무차원 |
| `Froude_Number` | Froude number, 무차원 |
| `Residuary_Resistance_per_Displacement` | 배수량 단위중량당 잉여저항, 무차원 |

### 프로그램에서 권장하는 선택

예제 파일을 업로드한 뒤 다음과 같이 선택하면 됩니다.

```text
X 열 (독립변수): Froude_Number
Y 열 (종속변수): Residuary_Resistance_per_Displacement
```

이 조합은 Froude number 증가에 따른 잉여저항 변화를 분석하므로, **선형회귀와 비선형회귀의 차이를 비교하기에 적합한 예제**입니다.

### 데이터 인용

> Gerritsma, J., Onnink, R., & Versluis, A. (1981). Yacht Hydrodynamics [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5XG7R.


## 19. 실행 예시

```bash
cd Regression-Analysis-Tool
```

라이브러리 설치:

```bash
python -m pip install -r requirements.txt
```

프로그램 실행:

```bash
python -m streamlit run Regression_Analyzer_Web.py
```

---

## 20. 문제 해결

### `requirements.txt`를 찾을 수 없다는 오류

예:

```text
ERROR: Could not open requirements file:
[Errno 2] No such file or directory: 'requirements.txt'
```

원인:
현재 CMD 위치에 `requirements.txt`가 없습니다.

해결:
프로그램 파일이 있는 폴더로 이동한 뒤 다시 실행합니다.

```bash
cd C:\Users\USER\Downloads\Regression-Analysis-Tool
python -m pip install -r requirements.txt
```

---

### `streamlit` 명령을 찾을 수 없는 경우

다음 방식으로 실행합니다.

```bash
python -m streamlit run Regression_Analyzer_Web.py
```

---

### 라이브러리 버전 충돌 경고가 나타나는 경우

기존 Python 환경에 설치된 다른 패키지와 NumPy 등의 버전이 충돌할 수 있습니다.

가장 안전한 방법은 새 가상환경을 생성하는 것입니다.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

---

### 로그 회귀 또는 거듭제곱 회귀가 실패하는 경우

X 값에 0 또는 음수가 포함되어 있는지 확인합니다.

로그/거듭제곱 모델은 현재 프로그램에서 양수 X 데이터를 요구합니다.

---

### 특정 모델만 분석에 실패하는 경우

모든 데이터가 모든 회귀모델에 적합한 것은 아닙니다.

예를 들어 Logistic 4P 모델은 충분한 데이터 개수와 S자 형태의 경향이 필요할 수 있습니다.

프로그램은 특정 모델이 실패하더라도 다른 모델의 분석 결과는 계속 표시하도록 구성되어 있습니다.

---

## 21. Streamlit Community Cloud 배포

GitHub에 아래 파일을 업로드합니다.

```text
Regression_Analyzer_Web.py
requirements.txt
README.md
sample_data.csv
```

Streamlit Community Cloud에서 GitHub 저장소를 연결한 뒤 Main file path를 다음과 같이 지정합니다.

```text
Regression_Analyzer_Web.py
```

배포가 완료되면 생성된 `streamlit.app` 주소를 Notion의 `/embed` 블록에 넣어 웹 프로그램을 직접 실행할 수 있습니다.

---

## 22. 프로그램 요약

본 프로그램은 실험 데이터를 입력받아 여러 선형·비선형 회귀모델을 적용하고, 각 모델의 회귀곡선과 통계지표를 비교할 수 있도록 제작된 웹 기반 데이터 분석 도구입니다.

단순히 회귀곡선을 그리는 것뿐 아니라 잔차 분석, 이상치 후보 검출, 95% 신뢰구간, 모델 선택, 예측 및 결과 저장 기능을 포함하여 실험 데이터 분석 과정 전체를 하나의 프로그램에서 수행할 수 있도록 구성하였습니다.
