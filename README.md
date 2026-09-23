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

## 파이프라인

```
S&P 500 목록 ──▶ EDGAR 10-K ──▶ 정제·품질 판정 ──▶ 기업 벡터 ──▶ 평가 ──▶ reports/<name>.md
(위키피디아)     (edgartools)    표·쪽번호 제거       TF-IDF        GICS·SIC 재현
                 Item 1 / 1A     추출 오류 걸러냄     SBERT         다음 해 주가 동조성
                                                      OpenAI        방법 간 일치도
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
uv run tenksim run -c configs/sp500_2024.yaml   # 본 실험 (수집 30분~1시간 + 임베딩 1시간 안팎)
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
$ uv run tenksim neighbors -c configs/smoke.yaml --method minilm --center NVDA -k 3
NVDA 와 비슷한 기업 (minilm+center)
  1. AMD    Advanced Micro Devices                   cos=0.604  pct= 95.4
  2. MSFT   Microsoft                                cos=0.485  pct= 93.4
  3. CDNS   Cadence Design Systems                   cos=0.484  pct= 92.8

$ uv run tenksim explain -c configs/smoke.yaml --method minilm SNPS CDNS --top 1
[1] cosine=0.813
  SNPS: Company and Segment Overview Synopsys, Inc. (Synopsys, we, our or us) delivers trusted and comprehensive silicon to systems design solutions, from electronic design automation (EDA), ...
  CDNS: Historically, the industry that provided the tools used by IC engineers was referred to as Electronic Design Automation (“EDA”). ...
```

`pct`는 전체 기업쌍 중 백분위(0~100)입니다. 코사인 값 자체는 모델마다 분포가 달라 절대값으로 해석하면 안 됩니다.
`explain`은 두 회사가 비슷하다고 나온 근거 문단을 보여줍니다.

## 산출물

| 경로 | 내용 |
|---|---|
| `reports/<name>.md` | 결과 리포트: 데이터 품질, 산업분류 재현, 주가 동조성, 방법 간 일치도, 예시 ([smoke 예시](reports/smoke.md)) |
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
evaluation:
  k: [1, 5, 10]
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
