# smoke 결과 리포트

- 생성: 2026-09-24T00:30:49
- 10-K 제출 연도: 2024 · 수집 섹션: business
- 분석 대상: 18개 기업 (universe 20개 중)
- 방법: `tfidf` (tfidf, business), `minilm` (sbert, business)
- `+center`: 전체 평균 벡터를 뺀 뒤 코사인 유사도를 잰 버전

지표 정의와 해석은 [docs/methodology.md](../docs/methodology.md)를 참고하세요.

## 1. 데이터 품질

| 섹션 | ok | suspect | too_short | missing | no_filing | error |
|---|---|---|---|---|---|---|
| business | 18 | 2 | 0 | 0 | 0 | 0 |

분석에서 제외된 문서:

| 티커 | 회사 | 섹션 | 상태 | 첫 줄 / 오류 |
|---|---|---|---|---|
| GE | GENERAL ELECTRIC CO | business | suspect | NOTE 2. BUSINESSES HELD FOR SALE AND DISCONTINUED OPERATIONS. In the fourth quarter of 202 |
| INTC | INTEL CORP | business | suspect | Item Number Item |

## 2. 산업분류 재현

텍스트로 찾은 이웃이 같은 산업에 속하는지 봅니다. AUC는 0.5가 무작위, P@k는 상위 k개 이웃 중 같은 산업 비율입니다.

### GICS 섹터 (n=18)

| 방법 | AUC | P@1 | P@3 |
|---|---|---|---|
| tfidf | 0.940 | 0.833 | 0.722 |
| minilm | 0.975 | 0.889 | 0.667 |
| minilm+center | 0.968 | 0.944 | 0.704 |
| (무작위 기대값) | 0.500 | 0.209 | 0.209 |

### GICS 서브산업 (n=18)

| 방법 | AUC | P@1 | P@3 |
|---|---|---|---|
| tfidf | 0.970 | 0.667 | 0.389 |
| minilm | 0.972 | 0.722 | 0.389 |
| minilm+center | 0.980 | 0.778 | 0.389 |
| (무작위 기대값) | 0.500 | 0.078 | 0.078 |

### SIC 2자리 (n=18)

| 방법 | AUC | P@1 | P@3 |
|---|---|---|---|
| tfidf | 0.973 | 0.778 | 0.426 |
| minilm | 0.972 | 0.778 | 0.426 |
| minilm+center | 0.977 | 0.833 | 0.426 |
| (무작위 기대값) | 0.500 | 0.098 | 0.098 |

## 3. 주가 동조성 (2025-01-01 ~ 2025-12-31, n=18)

텍스트 이웃과의 일별 수익률 상관 평균입니다. 잔차는 시장(베타) 움직임을 뺀 값입니다. 산업분류 기준선은 같은 분류의 회사 전체를 peer로 둔 값입니다.

| 방법 | 잔차@1 | 잔차@3 | 원수익률@1 | 원수익률@3 |
|---|---|---|---|---|
| tfidf | 0.385 | 0.276 | 0.574 | 0.493 |
| minilm | 0.420 | 0.264 | 0.626 | 0.507 |
| minilm+center | 0.432 | 0.257 | 0.615 | 0.500 |
| (같은 GICS 섹터 전체) | 0.309 | 0.309 | 0.530 | 0.530 |
| (같은 GICS 서브산업 전체) | 0.404 | 0.404 | 0.567 | 0.567 |
| (같은 SIC 2자리 전체) | 0.380 | 0.380 | 0.563 | 0.563 |
| (무작위) | 0.012 | 0.012 | 0.307 | 0.307 |

## 4. 방법 간 일치도

Spearman은 전체 기업쌍 점수의 순위 상관, Jaccard@3는 상위 3개 이웃이 겹치는 정도입니다.

| A | B | Spearman | Jaccard@3 |
|---|---|---|---|
| tfidf | minilm | 0.687 | 0.622 |
| tfidf | minilm+center | 0.669 | 0.589 |
| minilm | minilm+center | 0.830 | 0.817 |

## 5. 예시: 유사 기업 상위 5개 (`minilm+center`)

괄호 안은 전체 기업쌍 중 백분위(0~100)입니다.

| 회사 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| AAPL | MSFT (88) | ORCL (80) | SNPS (79) | CDNS (78) | AMD (78) |
| MSFT | NVDA (93) | ORCL (92) | AAPL (88) | SNPS (88) | CDNS (86) |
| NVDA | AMD (95) | MSFT (93) | CDNS (93) | SNPS (89) | ORCL (84) |
| JPM | BAC (97) | WFC (96) | SPG (87) | BRK.B (76) | XOM (74) |
| XOM | CVX (94) | BRK.B (80) | O (77) | SPG (75) | JPM (74) |
| PFE | LLY (100) | MRK (99) | AAPL (67) | BRK.B (66) | XOM (55) |

