# 10-K Report Similarity

SEC 10-K의 사업 설명(Item 1)으로 기업 간 사업 유사도를 측정하는 연구용 파이프라인입니다.
그 점수가 믿을 만한지 **산업분류(GICS·SIC)를 재현하는지**, **다음 해 주가가 실제로 같이 움직이는 기업을 찾는지**로 검증합니다.

> **연구 질문:** 10-K 사업 설명으로 찾은 '비슷한 기업'이 기존 산업분류보다 더 나은 peer인가?

설계 근거와 지표 정의는 [docs/methodology.md](docs/methodology.md)에 있습니다.

## v1 → v2에서 바뀐 점

| | v1 (2025-09) | v2 |
|---|---|---|
| 대상 | 기술주 5개 | S&P 500 (약 500개, 11개 섹터) |
| 텍스트 | Item 1 + 1A 청크 평균 (1A가 58~77%) | Item 1이 기본, 1A는 따로 비교 |
| 수집 | sec-api.io (유료, API 키) | edgartools로 EDGAR 직접 (무료) |
| 전처리 | 소문자화·구두점 제거 | 대소문자 유지, 표·쪽번호·머리글 제거, 추출 오류 판정 |
| 모델 | OpenAI 임베딩 3종 차원 | TF-IDF 기준선 + 로컬 SBERT (+ OpenAI 선택) |
| 검증 | 없음 | 산업분류 재현(P@k, AUC), 다음 해 수익률 상관 |
| 재현성 | 파일명·키 하드코딩 | 설정 파일, 단계별 캐시, 환경변수, 테스트 |

## 주요 결과: S&P 500, 2024년 10-K → 2025년 주가

476개 기업(추출 품질 검사를 통과한 기업)의 Item 1 텍스트로 유사도를 만들고, 2025년 일별 수익률로 검증했습니다.
전체 표는 [reports/sp500_2024.md](reports/sp500_2024.md)에 있습니다.

핵심 지표는 **텍스트 이웃끼리 주가가 실제로 같이 움직이는가**, 그리고 **GICS가 같은 업종으로 묶지 않는 곳에서도 그런가**입니다.
GICS 분류 재현(P@k)은 텍스트가 사업 내용을 담고 있는지 확인하는 최소 조건으로만 봅니다.
분류 밖의 연결, 예를 들어 Apple의 부품 공급사 Skyworks를 찾아내면 P@k에서는 오답으로 채점되기 때문입니다.

| 방법 | 상위 5 이웃 잔차상관 [95% CI] | 서브산업 밖 이웃: 같은 섹터 대비 | GICS 통제 β | 서브산업 P@5 |
|---|---|---|---|---|
| TF-IDF (단어 빈도 기준선) | 0.305 [0.291, 0.320] | +0.084 | +0.034 | 0.391 |
| MiniLM, Item 1 전체 (+center) | 0.290 [0.275, 0.304] | +0.079 | +0.030 | 0.373 |
| mpnet, 앞 1536토큰 (+center) | 0.285 [0.271, 0.302] | +0.081 | +0.038 | 0.372 |
| OpenAI 3-large, 앞 1536토큰 (+center) | 0.307 [0.290, 0.322] | +0.093 | +0.038 | 0.398 |
| OpenAI 3-large, 8천 토큰 청크 (+center) | 0.188 [0.174, 0.205] | +0.004 (유의하지 않음) | +0.014 | 0.195 |
| MiniLM, **Item 1A** 전체 (+center) | 0.279 [0.263, 0.295] | +0.072 | +0.025 | 0.345 |
| **앙상블: TF-IDF + OpenAI** | **0.318 [0.303, 0.333]** | **+0.102** | +0.025 | **0.416** |
| 앙상블: TF-IDF + MiniLM (무료) | 0.306 [0.291, 0.321] | +0.088 | +0.022 | 0.399 |
| 같은 GICS 서브산업 기업 전체 (기준선) | 0.359 [0.340, 0.380] | – | – | – |
| 무작위 | 0.005 | – | – | 0.012 |

표의 용어는 다음과 같습니다.

- 잔차상관: 시장 전체 움직임을 뺀 일별 수익률 상관. 시장 요인은 유니버스 동일가중 평균(자기 제외)입니다.
- `+center`: 전체 평균 벡터를 뺀 뒤 잰 코사인 유사도.
- '서브산업 밖 이웃: 같은 섹터 대비': 자기 서브산업 밖에서만 고른 상위 5개 이웃의 잔차상관에서, 같은 섹터·다른 서브산업 회사 평균(0.128)을 뺀 값.
- 'GICS 통제 β': 같은 서브산업·섹터인지를 통제한 기업쌍 회귀에서, 유사도가 1 표준편차 높을 때 늘어나는 잔차상관.
- 괄호와 아래 [ ] 구간은 기업 단위 부트스트랩 95% 신뢰구간입니다.

- **텍스트는 GICS가 묶지 않는 연결을 찾고, 그 연결은 실제로 주가에 나타납니다.**
  - 서브산업 밖에서만 고른 이웃도 같은 섹터 회사보다 더 같이 움직였습니다(TF-IDF +0.084 [+0.074, +0.095]).
  - GICS를 통제해도 유사도의 효과가 유의했습니다(β +0.03~0.04, 설명력 R² 0.082 → 0.10~0.12).
  - Apple–Skyworks의 잔차상관은 0.27로, Apple과 가장 같이 움직인 회사 475개 중 2위입니다. TF-IDF는 Skyworks를 Apple의 1위 이웃으로 찾았습니다. NVIDIA–Arista(AI 네트워킹)도 0.54입니다.
- **앙상블(TF-IDF + OpenAI)이 가장 좋았습니다.** 상위 5 이웃 잔차상관이 TF-IDF보다 +0.013 [+0.007, +0.019] 높았고, 분류 밖 연결 지표도 가장 높았습니다. 두 방법의 이웃이 절반 이상 달라서(상위 10개 Jaccard 0.38) 서로 보완합니다.
- **단순한 TF-IDF는 여전히 강력한 기준선입니다.** 단독 방법 중에서는 OpenAI(짧은 청크)와 통계적으로 같았고(+0.001 [-0.006, +0.009]), 무료 임베딩들보다는 유의하게 나았습니다.
- **모델보다 입력 길이가 중요했습니다.** 같은 OpenAI 모델이라도 청크 하나가 최대 8천 토큰이면 최하위였고, 384토큰으로 자르자 최상위권이 됐습니다.
- **평균 벡터 제거는 모든 임베딩에 효과가 있었습니다.** v1에서 점수가 0.8 근처에 몰리던 현상은 겉보기 문제가 아니라 변별력 손실이었습니다.
- **Item 1이 Item 1A보다 나았습니다.** 같은 모델로 비교하면 Item 1A는 모든 지표에서 뒤졌습니다. 평균 벡터를 빼지 않으면 GICS를 넘어서는 설명력도 없었습니다(β -0.002, 유의하지 않음).
- **시장 요인은 SPY가 아니라 동일가중을 씁니다.** 시가총액 가중 지수로 시장 움직임을 빼면 지수 비중이 큰 대형주끼리 상관이 음(-)으로 치우칩니다. 예를 들어 Apple–Microsoft가 SPY 기준 -0.09, 동일가중 기준 +0.19입니다. 방법 간 순위는 어느 쪽을 써도 같습니다.
- **한계:** 상위 5개 이웃 전체로 보면 텍스트 이웃(0.29~0.32)이 GICS 서브산업 전체 평균(0.359)보다 아직 낮습니다. 가장 가까운 1위 이웃만 보면 서브산업 평균보다 높습니다(TF-IDF 0.426). 또 이 지표는 공통 노출(같이 오르내림)을 재므로, 점유율을 다투는 경쟁사처럼 반대로 움직이는 관계는 과소평가합니다(예: Amazon–Walmart -0.03).

## 파이프라인

```
S&P 500 목록 ──▶ EDGAR 10-K ──▶ 정제·품질 판정 ──▶ 기업 벡터 ──▶ 평가 ──▶ reports/<name>.md
(위키피디아)     (edgartools)    표·쪽번호 제거       TF-IDF        다음 해 주가 동조성
                 Item 1 / 1A     추출 오류 걸러냄     SBERT         GICS 밖 연결 (분류 통제)
                                                      OpenAI        GICS·SIC 재현 (최소 조건)
                                                      앙상블        신뢰구간 (부트스트랩)
```

## 설치

[uv](https://docs.astral.sh/uv/)를 권장합니다.

```bash
uv sync --extra sbert --extra returns   # 로컬 임베딩 모델 + 수익률 검증
uv sync --all-extras                    # OpenAI, 개발 도구(pytest, ruff)까지 전부
```

pip를 쓴다면 `pip install -e ".[sbert,returns]"`로 설치합니다. Python 3.10 이상이 필요합니다.

`.env.example`을 `.env`로 복사하고 값을 채웁니다. `.env`는 git에 올라가지 않습니다.

```bash
EDGAR_IDENTITY="이름 이메일"   # 필수. SEC가 요청마다 요구합니다
OPENAI_API_KEY=...            # 선택. methods에 kind: openai를 쓸 때만
```

## 실행

```bash
uv run tenksim run -c configs/smoke.yaml        # 기업 20개, 5~10분. 설치·연결 확인용
uv run tenksim run -c configs/sp500_2024.yaml   # 본 실험. 8코어 노트북 CPU에서 수집 10분 + 임베딩 45분
```

중간에 끊겨도 같은 명령을 다시 실행하면 받아 둔 10-K와 계산한 임베딩은 건너뜁니다.
단계별로 나눠 돌릴 수도 있습니다.

```bash
uv run tenksim ingest   -c configs/sp500_2024.yaml   # 기업 목록 + 10-K 수집 + 정제·품질 판정
uv run tenksim embed    -c configs/sp500_2024.yaml   # --method minilm 처럼 일부만 가능
uv run tenksim evaluate -c configs/sp500_2024.yaml   # metrics.json + reports/sp500_2024.md
```

결과를 살펴보는 명령도 있습니다.

```console
$ uv run tenksim neighbors -c configs/sp500_2024.yaml --method tfidf-openai AAPL -k 5
AAPL 와 비슷한 기업 (tfidf-openai)
  1. BBY    Best Buy                                 score=95.712  pct= 96.9
  2. QCOM   Qualcomm                                 score=95.556  pct= 96.8
  3. MSFT   Microsoft                                score=95.474  pct= 96.7
  4. SWKS   Skyworks Solutions                       score=95.001  pct= 96.4
  5. GRMN   Garmin                                   score=94.639  pct= 96.1

$ uv run tenksim explain -c configs/smoke.yaml --method minilm SNPS CDNS --top 1
[1] cosine=0.813
  SNPS: Company and Segment Overview Synopsys, Inc. (Synopsys, we, our or us) delivers trusted and comprehensive silicon to systems design solutions, from electronic design automation (EDA), ...
  CDNS: Historically, the industry that provided the tools used by IC engineers was referred to as Electronic Design Automation (“EDA”). ...
```

`--method`에는 방법 이름(`minilm`), 변형(`minilm+center` 또는 `--center`), 앙상블 이름(`tfidf-openai`)을 쓸 수 있습니다.
`score`는 방법의 유사도(코사인, 앙상블이면 평균 백분위)이고, `pct`는 전체 기업쌍 중 백분위(0~100)입니다.
코사인 값 자체는 모델마다 분포가 달라 절대값으로 해석하면 안 됩니다.
`explain`은 두 회사가 비슷하다고 나온 근거 문단을 보여줍니다(청크 임베딩 방법만).

## 기업 관계도 (개발 중)

10-K에서 기업 간 관계(경쟁·공급·협력·지분)를 근거 문장과 함께 찾아 보여주는 단독 웹앱입니다.
설계와 진행 상황은 [docs/relation-map-plan.md](docs/relation-map-plan.md)에 있습니다.
회사를 고르면 **관계도**(경쟁 · 공급·협력 · 지분, 방향 없음)와 관계마다 근거 문장이 나옵니다. 판정 모델이 불확실로 남긴 관계는 '관계 검수'에서 맞음·아님으로 정하고, 판정 전 후보(유사도 상위 기업 ∪ 10-K에 이름이 나온 기업)는 회사 화면의 둘째 탭에서 봅니다.

```bash
uv sync --extra app                                        # 웹앱 의존성 (FastAPI)
uv run tenksim export --all -c configs/sp500_2024.yaml     # 이름 언급 → 후보 → graph.db (10초 정도)
cd web && npm install && npm run build && cd ..            # 화면 빌드 (Node.js 필요, 처음 한 번)
uv run tenksim serve -c configs/sp500_2024.yaml            # http://127.0.0.1:8765 이 열립니다
```

- **표본 검수:** `uv run tenksim sample -c configs/sp500_2024.yaml --name dev1 --purpose dev`로 표본 회사 10곳을 뽑은 뒤, 웹앱의 '표본 검수'에서 문장마다 그 회사가 맞는지와 관계(경쟁 · 공급·협력 · 지분, 방향 없이)를 고릅니다(모델 판정은 가려져 있습니다). 이름 없이 `tenksim sample`만 실행하면 표본별 진행 상황이 나옵니다. 검수 기록은 `data/runs/<name>/relations/reviews.sqlite`에 쌓이고, graph.db를 다시 만들어도 지워지지 않습니다.
- **판정과 채점 (1단계, 진행 중):** `.env`에 `TYPESAFE_API_KEY`를 넣고 `uv sync --extra judge` 뒤 `uv run tenksim eval-relations -c configs/sp500_2024.yaml --sample dev1`을 실행하면, 표본의 문장마다 Jev가 회사 식별과 관계(경쟁 · 공급·협력 · 지분)를 판정하고 검수 라벨로 채점합니다(10개사 약 180문장에 1센트 정도). 기본 입력은 근거 문장과 앞뒤 문단이고, `--context span`(문장만)이나 `--context pair`(같은 쌍의 다른 문장까지)로 바꿔 비교할 수 있습니다. 결과는 `data/runs/<name>/relations/eval/`에 표(`.md`)와 라벨과 어긋난 문장 목록(`_disagreements.csv`, 엑셀로 열림)으로 남습니다. 한 번 판정한 입력은 `judgements.sqlite`에 저장돼 다시 돈을 내지 않습니다. 질문별 채택·기각 점수는 설정의 `relations.judge.thresholds`에서 바꾸고, 채점표 끝의 '임계값별' 표가 개발 표본 기준 제안값을 보여줍니다. 합격 판단용 확인 표본(`--purpose confirm`)은 뽑을 때 판정 설정을 고정하고, 설정이 바뀌면 채점을 멈춥니다. 보내기 전에 입력을 보려면 `tenksim judge --sample dev1 --dry-run`.
- **전체 판정과 관계 만들기 (2단계):** `uv run tenksim judge --all -c configs/sp500_2024.yaml`은 모든 이름 언급(S&P 500 약 3,600문장, 약 $0.2)을 판정하고, 두 회사 관계(경쟁 · 공급·협력 · 지분)와 근거 문장을 `graph.db`에 합칩니다. `tenksim export`는 이미 판정한 것만 합치고 모델을 부르지 않습니다. 신용평가사 언급은 관계로 치지 않습니다(`configs/aliases.yaml`의 `credit_rating`).
- 단계별로 돌리려면 `tenksim mentions`, `tenksim candidates`, `tenksim export`를 차례로 실행합니다. `--ticker NVDA`를 붙이면 그 회사 결과를 터미널에 출력합니다.
- 회사 이름 사전(별칭, 분사 시점, 제외할 문맥)은 [configs/aliases.yaml](configs/aliases.yaml)에서 고칩니다.
- 화면을 고치는 중에는 `tenksim serve --no-browser`를 켜 두고 `web/`에서 `npm run dev`를 실행하면 바로 반영됩니다.

## 산출물

| 경로 | 내용 |
|---|---|
| `reports/<name>.md` | 결과 리포트: 데이터 품질, 주가 동조성, GICS 밖 연결, 산업분류 재현, 방법 간 일치도, 예시 ([S&P 500](reports/sp500_2024.md), [smoke](reports/smoke.md)) |
| `data/sections/<year>/<cik>.parquet` | 회사별 10-K 섹션 원문 (모든 실행이 공유) |
| `data/embeddings.sqlite` | 청크 임베딩 캐시 (모델 + 텍스트 해시 기준) |
| `data/runs/<name>/` | universe, 정제된 문서, method별 벡터·청크·이웃, `metrics.json` |

`data/`는 git에 올리지 않습니다. 이 밖에 edgartools의 HTTP 캐시(`~/.edgar`, 10-K 한 건에 평균 수 MB)와
Hugging Face 모델 캐시(`~/.cache/huggingface`)가 쌓입니다.

## 설정 파일

`configs/*.yaml` 하나가 실험 한 번입니다. 주요 항목은 다음과 같습니다.

```yaml
name: sp500_2024
universe:
  source: sp500_wikipedia        # 또는 csv (+ path: ticker 또는 cik 컬럼이 있는 파일)
  tickers: [AAPL, MSFT]          # 일부만 쓸 때
  cik_overrides: {XOM: 34088}    # CIK가 바뀐 회사를 그해 CIK로 바로잡기
filings:
  year: 2024                     # 이 해에 '제출된' 10-K 원본
  sections: [business, risk_factors]
methods:
  - {name: tfidf, kind: tfidf}
  - {name: mpnet-1536, kind: sbert, model: sentence-transformers/all-mpnet-base-v2, doc_tokens: 1536}
  - {name: minilm-risk, kind: sbert, model: sentence-transformers/all-MiniLM-L6-v2, section: risk_factors}
ensembles:                       # 유사도를 기업쌍 백분위로 바꿔 평균낸 조합
  - {name: tfidf-mpnet, members: [tfidf, mpnet-1536+center]}
evaluation:
  k: [1, 5, 10]
  primary_k: 5                   # 신뢰구간과 '가장 좋은 방법' 선택 기준
  baseline: tfidf                # 짝지은 차이의 기준 (기본: 첫 method)
  n_boot: 1000
  labels: [gics_sector, gics_sub_industry, sic2, sic3]
  returns: {start: 2025-01-01, end: 2025-12-31}   # 공시 다음 해 → 미래 정보 없음
```

모르는 키는 오류로 막습니다(오타 방지). 전체 항목은 `src/tenksim/config.py`에 있습니다.

## 테스트

```bash
uv run pytest                                     # 오프라인 테스트 (네트워크 없이)
EDGAR_IDENTITY="이름 이메일" uv run pytest -m live  # 실제 EDGAR에서 Apple 10-K를 받아 확인
```

## 알려진 한계

- 기업 목록·GICS·CIK가 모두 **현재** 기준입니다. 과거 연도에 쓰면 생존 편향이 생기고, CIK가 바뀐 회사는
  `no_filing`으로 나옵니다(`cik_overrides`로 수정).
- GE, Intel처럼 '10-K 상호참조 색인' 형식을 쓰는 회사는 edgartools가 Item 1을 잘못 잘라 와서
  `suspect`로 제외됩니다. 제외 목록은 리포트 1장에 나옵니다.
- 주가는 Yahoo Finance(비공식 API)에서 받습니다.

## 참고

- Vamvourellis et al. (2023), *Company Similarity using Large Language Models*, [arXiv:2308.08031](https://arxiv.org/abs/2308.08031)
- Hoberg & Phillips (2016), Text-Based Network Industries and Endogenous Product Differentiation, *JPE*
- [edgartools](https://github.com/dgunning/edgartools) · [SEC EDGAR 접근 규정](https://www.sec.gov/os/accessing-edgar-data)
