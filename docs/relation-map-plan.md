# 10-K 기업 관계도: 구현 계획

> **상태: 검토용 초안 v2.1 (2026-09-24). 0단계 구현 중입니다.**
> 저장소 주인과 다른 AI 모델(GPT 등)이 이 설계를 함께 검토하려고 쓴 문서입니다. 코드를 보지 않아도 이해할 수 있게 배경을 모두 담았습니다.
> v2는 1차 검토 의견(Codex, [codex_review.md](codex_review.md))을 반영했습니다. 무엇이 바뀌었는지는 17장에 있습니다.
> 저장소: <https://github.com/financeis/10-K-Report-Similarity> · 현재 결과: [README](../README.md) · 유사도 설계와 평가: [methodology.md](methodology.md)

표기: **[확인]** 데이터나 문서로 확인한 사실 · **[제안]** 아직 결정하지 않은 설계 · **[미정]** 저장소 주인이 정할 것

## 0. 검토자에게

- **만들려는 것:** 10-K에서 기업 간 관계(경쟁·공급·고객·협력·지분)를 찾아 **근거 문장과 함께** 보여주는 단독 웹앱입니다.
- **핵심 설계 판단**
  1. 1차 후보는 '유사도 상위 K ∪ 본문 이름 언급'으로 잡습니다(3장의 조사 결과가 근거).
  2. 판정은 세 단계로 나눕니다. 언급 하나하나가 그 회사를 가리키는지 보고, 근거 구간마다 어떤 관계와 방향을 말하는지 보고, 마지막에 쌍 단위로 합칩니다(7.3).
  3. 판정 모델은 텍스트를 생성하지 않는 판정 전용 모델 Jev를 기본 후보로 둡니다. 근거 고르기, 숫자 추출, 결과 합치기는 코드가 합니다(6·7장).
  4. 관계는 방향과 시점(현재·과거·계획)을 가지고, 한 쌍에 여러 개일 수 있습니다. 공급과 고객은 방향이 있는 하나의 관계로 합쳐 저장합니다(5장).
  5. 후보·판정 상태·검수 기록을 따로 저장하고, 사람의 검수가 모델 판정보다 우선합니다(7.5).
- **Jev는 검토자 모델이 모를 수 있습니다.** 2026-09-15에 나온 모델이라서, 확인한 사양을 출처와 함께 6장에 정리했습니다.
- **역할 분담:** 저장소 주인은 금융 실무자입니다. 관계 유형의 정의는 저장소 주인이 정합니다. 정답은 관계 목록을 미리 쓰는 대신, 표본 회사의 근거 문장을 저장소 주인이 검수해서 만듭니다(7.6). AI가 만든 판정을 정답으로 쓰지 않습니다.
- 남은 질문 목록은 15장에 있습니다.

## 1. 배경: 지금 저장소가 하는 일 [확인]

`tenksim`은 SEC 10-K의 사업 설명으로 기업 간 사업 유사도를 재는 연구용 파이프라인입니다.

```
S&P 500 목록 → EDGAR 10-K     → 정제·품질 판정   → 기업 벡터         → 유사도 행렬 → 평가
(위키피디아)    (edgartools)     표·쪽번호 제거     TF-IDF, SBERT,                    주가 동조성
                Item 1, 1A       추출 오류 판정     OpenAI, 앙상블                    GICS 밖 연결
                                                                                      분류 재현(최소 조건)
```

**대상과 방법**
- 2024년에 제출된 S&P 500 10-K 가운데 추출 품질 검사를 통과한 476개사입니다.
- 가장 좋은 방법은 TF-IDF와 OpenAI 임베딩의 앙상블(`tfidf-openai`)입니다.
  - 임베딩은 text-embedding-3-large로, Item 1 앞 1,536토큰을 384토큰 청크로 나눠 만들고 평균 벡터를 뺍니다.
  - 두 유사도를 기업쌍 백분위로 바꿔 평균냅니다.

**주요 결과.** 2025년 일별 수익률에서 동일가중 시장 요인을 뺀 잔차의 상관입니다.

| | 상위 5 이웃 잔차상관 [95% CI] |
|---|---|
| 앙상블 TF-IDF + OpenAI | 0.318 [0.303, 0.333] |
| TF-IDF 단독 | 0.305 [0.291, 0.320] |
| 같은 GICS 서브산업 기업 전체 | 0.359 |
| 무작위 | 0.005 |

**GICS가 묶지 않는 연결도 찾습니다.**
- Apple–Skyworks는 부품 공급 관계이고 GICS 서브산업이 다릅니다.
- 두 회사의 잔차상관은 0.27로, Apple과의 475개 쌍 중 2위입니다.
- TF-IDF는 Skyworks를 Apple의 1위 이웃으로 찾았습니다.

**한계: 유사도는 관계의 종류를 모릅니다.** 경쟁사와 공급사가 모두 '가까운 회사'로 섞여 나옵니다. 주가 동조성도 간접 증거일 뿐입니다. 이 문서는 그 한계를 푸는 계획입니다.

**관련 코드(`src/tenksim/`)와 데이터**

| 위치 | 역할 |
|---|---|
| `filings.py` | 수집. 회사별 원문을 `data/sections/<연도>/<cik>.parquet`에 저장(원문 `text_raw`, accession 번호, 제출일, 보고 기간, EDGAR URL 포함) |
| `text.py` | 정제. HTML 엔티티 해제, NFKC 정규화, 쪽번호·표·반복 머리글 제거. 정제 텍스트는 `data/runs/<name>/documents.parquet`에 저장(원문은 빠짐) |
| `embed/` | 방법별 벡터. 교체 가능한 어댑터 |
| `similarity.py` | 유사도, 앙상블, 두 회사의 가장 비슷한 청크 쌍을 찾는 `explain_pair` |
| `returns.py` | 잔차 상관 |
| `pipeline.py` | 단계별 산출물 저장. 다시 실행하면 한 일은 건너뜀 |
| `cli.py` | 명령줄 |

**개발 환경:** Windows 11, Python 3.13(uv), GPU 없는 노트북.

## 2. 목표와 범위

**목표**
1. 회사 하나를 고르면 관련 기업을 관계 유형별로 보여줍니다.
2. 모든 관계에 근거를 붙입니다. 근거는 10-K 원문 문장과 출처(몇 년 10-K의 어느 항목)이고, 그 문장이 그 관계를 뒷받침한다는 판정도 함께 기록합니다.
3. 관계 판정이 얼마나 맞는지 저장소 주인의 표본 검수로 잽니다.

**범위 밖**
- 저장소 주인이 개발 중인 다른 웹앱(미국 기업 리포트 앱)과의 연동. 그 앱은 설계 초기라 이번에는 고려하지 않습니다.
- 투자 신호와 백테스트.
- 여러 사용자와 외부 배포. 로컬 단일 사용자 앱입니다.

## 3. 사전 조사: 유사도만으로 1차 필터를 하면 무엇을 놓치나 [확인]

**방법**
1. 476개사의 Item 1·1A 정제 텍스트에서 다른 S&P 500 기업의 이름을 찾았습니다. 회사명 사전은 이렇게 만들었습니다.
   - 회사명에서 Inc. 같은 접미사를 떼고, 별칭 약 60개를 손으로 추가했습니다.
   - 'Target'·'Delta'처럼 일반 단어인 이름은 뺐습니다.
   - 거래소·지수·신용평가사(NDAQ, SPGI, MCO, MSCI, CBOE, ICE, CME)도 뺐습니다. 관계와 상관없이 거의 모든 10-K에 나오기 때문입니다. (이 방식은 실제 관계까지 지우므로 본 구현에서는 문맥 기준 제외로 바꿉니다. 7.1 참고.)
2. 문장 안의 단어로 맥락을 대략 나눴습니다.
   - 경쟁: compet
   - 고객·공급: customer, supplier, vendor, foundry, sole source 등
   - 제휴: partner, alliance, joint venture, licens 등
3. 상대 회사가 앙상블 유사도로 몇 위였는지 확인했습니다. 순위는 475개 중이고, 1위가 가장 비슷한 회사입니다.

**규모**
- 언급은 3,348건이고, (A가 B를 언급한) 쌍으로는 1,561개입니다.
- 476개사 중 368개사가 다른 S&P 500 기업을 하나 이상 이름으로 적었습니다.
- 이름 사전 검색은 476개사 전체에 7초 걸렸습니다.

| 언급 맥락 | 쌍 | 유사도 상위 10 | 상위 20 | 상위 50 | 순위 중앙값 | 섹터가 다른 비율 |
|---|---|---|---|---|---|---|
| 경쟁 | 396 | 64% | 77% | 89% | 5 | 26% |
| 고객·공급 | 235 | 29% | 46% | 70% | 25 | 48% |
| 경쟁 + 고객·공급 | 99 | 54% | 73% | 91% | 9 | 18% |
| 제휴·라이선스 | 129 | 29% | 49% | 64% | 22 | 52% |
| 기타(맥락 단어 없음) | 702 | 29% | 40% | 57% | 37 | 52% |

두 문장 이상에서 언급된 쌍만 보면 경쟁은 77%, 고객·공급은 56%가 상위 20 안이었습니다.

**주의: 이 표의 맥락은 단어 규칙으로 나눈 것입니다.** 실제 관계가 있다는 판정이 아니므로, 이 비율을 관계 재현율로 읽으면 안 됩니다. '유사도만 쓰면 공급망 언급의 상당 부분이 후보에서 빠진다'는 방향만 보여줍니다.

**유사도 순위는 낮지만 실제로 있는 공급망 관계의 예**

| 언급한 회사 → 상대 | 상대의 유사도 순위 | 10-K 문장 |
|---|---|---|
| UPS → Amazon | 209 | "one customer, Amazon.com, Inc. and its affiliates, represented approximately 11.8% of our consolidated revenues" |
| Hasbro → Amazon | 232 | "...including our largest customers, Wal-Mart Stores, Inc. and Amazon.com" |
| Veeva → Amazon | 178 | "...we utilize Amazon Web Services." |
| Nucor → ExxonMobil | 149 | "In 2023, we signed an agreement with Exxon Mobil to capture, transport, and store carbon from our DRI plant in Convent, Louisiana." |

**함께 확인된 것**
- **언급은 대부분 한쪽 10-K에만 있습니다.** A가 B를 언급한 쌍 가운데 B도 A를 언급한 경우는 17%입니다. 그래서 두 회사의 10-K를 모두 봐야 합니다.
- **이름이 나왔다고 관계가 있는 것은 아닙니다.**
  - '기타' 쌍 15개를 무작위로 보니 7개가 임원 경력이었습니다. 예: GM 10-K의 CFO 약력 "Delta Air Lines, Executive Vice President — Chief Financial Officer (2013)". 이 Delta는 실제 Delta Air Lines가 맞지만 관계의 근거는 아닙니다. 그래서 '같은 회사인가'와 '관계의 근거인가'를 따로 판정해야 합니다.
  - 같은 이름의 다른 회사나 지명도 있었습니다.
    - Exelon의 "Constellation"은 분사한 Constellation Energy인데, 사전이 Constellation Brands로 연결했습니다.
    - PPL의 "Trimble County"는 발전소 이름입니다.
    - Con Edison의 "Dover Station"도 회사가 아닙니다.
- **한 쌍에 관계가 여러 개입니다.**
  - Qualcomm의 10-K에는 두 문장이 함께 있습니다. 하나는 "revenues from Apple, Samsung and Xiaomi each comprised 10% or more of our consolidated revenues"입니다. 다른 하나는 "competition ... from products internally developed by our customers, including some of our largest customers, such as Apple and Samsung"입니다.
  - NVIDIA는 HPE를 네트워킹 경쟁사로 적습니다. 반면 HPE는 자사 제품을 "co-developed with AI market leader Nvidia"라고 적어 공동 개발 파트너로 소개합니다.
- **분석 대상 밖 기업과 익명 공시도 중요합니다.**
  - NVIDIA의 파운드리는 "TSMC, and Samsung Electronics"로, 둘 다 S&P 500 밖입니다.
  - 가장 큰 고객은 "Customer A, represented 13% of total revenue"로 익명입니다.
- Item 1의 80%(488개 중)에 'Competition' 같은 경쟁 소제목이 있습니다. 경쟁 근거 문장을 고를 때 쓸 수 있습니다.

**이 조사의 한계**
- 이름 사전 방식이라 정확도와 재현율이 모두 대략치입니다. 이름 패턴이 있는 회사는 419개이고, 맥락 분류는 단어 규칙입니다.
- S&P 500 안의 관계만 셉니다.
- Item 7이나 재무제표 주석에 있는 고객 공시는 빠져 있습니다.
- 조사 스크립트는 저장소 밖에서 돌렸고, 0단계에서 `tenksim mentions`로 정식화합니다.

**설계에 주는 의미**
1. **1차 후보는 유사도 상위 K와 이름 언급의 합집합이어야 합니다.** 방향 없는 쌍으로 세면 상위 20은 6,102쌍이고, 언급 쌍을 더하면 6,737쌍(+10%)입니다.
2. **언급 하나하나에 대해 '같은 회사인가'와 '관계의 근거인가'를 따로 판정해야 합니다.**
3. **관계 모델**은 다중 라벨·방향·시점을 지원하고, 두 회사의 근거를 합쳐서 판정해야 합니다.
4. **외부 기업과 익명 고객**도 노드로 보여줘야 합니다.

## 4. 설계 원칙 [제안]

1. **근거 우선.** 관계에는 10-K 원문 문장을 그대로 붙입니다. 모델이 쓴 문장을 인용처럼 보여주지 않습니다. 또 문장을 붙였다는 사실만으로 근거가 되지 않습니다. **그 문장이 그 관계와 방향을 뒷받침한다는 판정**이 있어야 관계에 연결합니다.
2. **코드는 찾고, 모델은 판단합니다.**
   - 코드가 하는 일: 후보 찾기, 근거 구간 고르기, 숫자 후보 추출, 결과 합치기.
   - 모델이 하는 일: 이 이름이 그 회사인지, 이 구간이 어떤 관계를 말하는지 판단.
3. **모델은 교체할 수 있게 만듭니다.**
   - 지금의 `embed/` 어댑터처럼 판정 모델도 설정 한 줄로 바꿉니다.
   - 결과는 (모델 버전, 질문 버전, 입력 해시)를 키로 캐시합니다.
4. **평가 우선.** 판정 모델과 임계값은 사람이 검수한 표본으로 정합니다. 질문을 다듬는 데 쓴 표본으로 최종 성능을 재지 않습니다.
5. **사람이 우선.** 사람의 검수는 모델 판정보다 우선하고, 절대 덮어쓰지 않습니다.
6. **웹앱은 계산된 결과만 읽습니다.**
   - 모델 호출은 배치 파이프라인에서만 합니다.
   - 화면에서는 사용자가 버튼을 누를 때만 호출합니다(예: 한국어 설명 생성).
7. **기존 파이프라인 방식을 따릅니다.** 단계별 산출물을 디스크에 저장하고, 다시 실행하면 한 일은 건너뜁니다.

## 5. 관계 스키마 [제안 · 미정]

유형 정의는 저장소 주인이 확정합니다. 아래는 초안입니다.

| 코드 | 이름 | 방향 | 하위 속성 | 10-K 예시 [확인] |
|---|---|---|---|---|
| `competitor` | 경쟁 | 없음 | – | NVIDIA: "...such as AMD, Arista Networks, Broadcom, Cisco Systems, Inc., Hewlett Packard Enterprise Company, Huawei, Intel, Lumentum Holdings, and Marvell Technology Group" |
| `supplies` | 공급 (X가 Y에 공급) | 공급자 → 고객 | 제품·부품 / 서비스 / 제조 위탁 / 기타 | NVIDIA: "We utilize foundries, such as Taiwan Semiconductor Manufacturing Company Limited, or TSMC, and Samsung Electronics Co., Ltd., or Samsung, to produce our semiconductor wafers." |
| `partner` | 협력 | 없음 | 라이선스 / 유통·재판매 / 합작(JV) / 공동 개발 / 기타 제휴 | ServiceNow: "one of these partnerships is with NVIDIA, with which we plan to develop powerful, enterprise-grade generative AI capabilities" |
| `equity` | 지분 보유 | 보유자 → 피투자사 | – | Berkshire: "Berkshire or a subsidiary owned 26.7% of the outstanding common stock of The Kraft Heinz Company" |
| `similar` | 사업 유사 | 없음 | – | 유사도 상위인데 검토 범위(해당 연도 Item 1·1A)에서 직접 관계를 확인하지 못한 쌍 |

**고객 관계는 따로 저장하지 않습니다.**
- "UPS의 고객 Amazon"은 `supplies(UPS → Amazon)`로 저장합니다.
- 화면에서는 지금 보고 있는 회사를 기준으로 공급사와 고객사를 나눠 보여줍니다.
- 같은 사실을 두 번 저장했다가 서로 어긋나는 일을 막기 위해서입니다.
- 두 방향의 공급은 동시에 참일 수 있습니다(서로 공급하는 관계).

**한 쌍에 여러 관계를 허용합니다.** 예를 들어 Qualcomm–Apple은 `supplies(QCOM → AAPL)`와 `competitor`를 함께 가집니다.

**`similar`의 뜻.** '관계가 없다'가 아니라 '검토한 근거 범위에서 직접 관계를 확인하지 못했다'는 뜻입니다. 화면에도 이렇게 설명합니다.

**시점(`status`)**

| 값 | 뜻 | 예 |
|---|---|---|
| `current` | 공시 시점에 유효한 관계 | "one customer, Amazon.com ... represented approximately 11.8%" |
| `historical` | 끝난 관계, 과거 사실 | 종료된 공급계약, 과거 지분 보유, 과거 인수 |
| `planned` | 발표만 된 관계 | "we plan to develop ..." |
| `unclear` | 판단할 수 없음 | – |

관계마다 근거 공시의 제출일과 보고 기간도 함께 저장합니다.

**근거 수준(`basis`)**
- `disclosed`: 어느 한쪽 10-K 문장에 상대 이름과 관계가 적혀 있고, 그 문장이 관계를 뒷받침한다고 판정됐습니다. 화면에서 실선입니다.
- `inferred`: 이름 언급은 없고, 두 회사의 사업 설명으로 판단했습니다. 점선입니다.
  - 초기 버전에서는 **경쟁만** `inferred`로 표시합니다.
  - 공급 관계는 사업 설명이 맞아떨어진다는 이유만으로 있다고 보기 어렵습니다. 그래서 공급 추론은 '탐색 후보'로만 저장하고 기본 화면에는 숨깁니다.
- `similarity`: 유사도만 있습니다. 흐린 점선입니다.

**노드 종류와 ID.** 노드 ID는 안정적이어야 하며, 웹앱 API도 티커가 아니라 이 ID를 씁니다.

| 종류 | 뜻 | ID 예 |
|---|---|---|
| `company` | 분석 대상 회사 | `cik:1045810` |
| `external` | 분석 대상 밖 회사. TSMC처럼 20-F를 내는 회사는 CIK를 쓰고, Samsung처럼 SEC에 없는 회사는 이름 슬러그를 씀 | `cik:1046179`, `ext:samsung-electronics` |
| `anonymous` | 익명 공시. (공시 기업, accession, 익명 표기) 범위 안에서만 같은 노드이고, 다른 연도의 'Customer A'와 자동으로 연결하지 않음 | `anon:1045810:<accession>:customer-a` |

**관계 속성:** 판정 점수, 판정 모델과 질문 버전, 근거 구간 목록, 매출 비중(7.1의 구조), 시점, 기준 공시(accession), 유사도 백분위, 잔차 상관.

## 6. Jev: 확인한 사양 (2026-09-24 기준) [확인]

Jev는 TypeSafe AI가 2026-09-15에 얼리 액세스로 낸 'System One' 모델입니다. 판정 전용이라 **텍스트를 생성하지 않습니다.**

**입력과 출력**
- 입력은 `state`(문자열, JSON 객체, 텍스트 배열)와 이름을 붙인 질문들입니다. 질문은 세 종류가 있습니다.

  | 종류 | 묻는 것 | 돌려주는 것 |
  |---|---|---|
  | `Noul` | 예/아니오 | 0~1 확률 |
  | `Choice` | 보기 중 하나 | 선택, 보기별 확률, confidence |
  | `Score` | 순서 있는 등급 | 점수, 확률, confidence |

- 출력은 미리 정한 형식의 값과 확률뿐입니다. 이유, 인용문, 새 이름은 나오지 않습니다.
- 한 요청 안의 여러 질문은 같은 state를 공유하며 병렬로 평가됩니다. 그래서 문장별 판정을 해도 요청을 문장마다 따로 보낼 필요는 없습니다. 질문 이름에 근거 ID를 넣으면 됩니다.
- 공식 문서는 **질문 사이의 논리적 일관성을 보장하지 않는다**고 밝힙니다. 서로 모순되는 답은 코드에서 검사해야 합니다.

**한도·가격·속도**
- 요청당 64k 토큰까지 넣을 수 있습니다. 그중 state와 가장 긴 질문을 합쳐 32k 토큰까지입니다.
- 초당 25만 토큰, 분당 1,200 요청까지 보낼 수 있습니다(수요에 따라 조정 중).
- 입력 100만 토큰당 $0.042이고 출력은 무료입니다.
- 요청당 70~500ms 걸립니다. 질문을 늘려도 시간이 거의 늘지 않습니다.
- 영어에서 가장 정확합니다. 10-K는 영어입니다.

**사용법** (공식 quickstart와 Python SDK 문서 기준)

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient   # pip install typesafe-sdk
client = TypeSafeClient()            # 키는 TYPESAFE_API_KEY 환경변수, 기본 모델 jev-latest
resp = client.system_one(model="jev-1.13.0", state=..., questions={"q1": Noul(instructions="...")})
resp.answers["q1"].noul              # 0~1
```

- HTTP로는 `POST https://api.typesafe.ai/v1/systemone`(Bearer 인증)이고, 본문은 `{"state", "model", "questions"}`입니다.
- 임계값을 맞출 때는 모델 버전을 고정하라고 권합니다(현재 `jev-1.13.0`).

**알려진 약점** (공식 문서와 리뷰)
- 질문을 글자 그대로 해석합니다. 부정어와 범위 표현에 주의해야 합니다.
- 셈과 날짜 계산을 하지 못합니다.
- 무관한 내용이 많으면 정확도가 떨어집니다.
- state 안의 적대적 문구에 흔들릴 수 있습니다.
- 서로 모순되는 기준에 약합니다.

**정확도** (개발사 자체 측정, 4개 업무)

| 모델 | 정확도 | 건당 비용 |
|---|---|---|
| Jev | 67.8% | $0.0004 |
| GPT-5.6 Terra | 67.9% | $0.0304 |
| GPT-5.6 Sol | 74.1% | $0.0836 |
| Claude Opus 5 | 73.1% | $0.1761 |

제3자 검증은 아직 없습니다. 개발사는 확률이 보정돼 있다고 하지만, **이 과제에서 보정됐는지 확인하기 전까지는 Jev 점수를 실제 정답 확률로 취급하지 않습니다.** 문서에서도 '확률' 대신 '점수'라고 부릅니다.

**이 프로젝트에 맞는 부분과 맞지 않는 부분 [제안]**
- **맞는 일:** 언급과 근거 구간마다 닫힌 질문(같은 회사인가, 경쟁을 말하는가, 누가 누구에게 공급하는가…)에 점수를 매기는 일입니다.
  - 비용이 거의 들지 않아 수천 개 회사로 넓혀도 부담이 없습니다.
  - 근거 문장은 코드가 골라 넣으므로, 없는 인용을 지어낼 수 없습니다.
  - 점수가 나오니 검수 순서를 정하기 좋습니다.
- **맞지 않는 일:**
  - 사전에 없는 회사 이름 찾기(열린 질문)와 한국어 설명 쓰기는 텍스트 LLM이 맡습니다.
  - 매출 비중 같은 숫자는 정규식으로 후보만 뽑습니다(7.1).
- **확인이 필요한 점:** 개발사 자체 측정에서도 최상위 모델보다 약 6%p 낮았습니다. 이 과제에 쓸 만한지는 검수 표본으로 비교해 정합니다(7.6).

## 7. 파이프라인 설계 [제안]

```
data/sections (원문, 기존) + documents.parquet (정제 텍스트, 기존) + 유사도 행렬 (기존, tfidf-openai)
   │
   ├─ ① mentions       문장 분리 → 회사명 사전 매칭 → 임원 약력 제외 → 언급·근거 구간·숫자 후보
   ├─ ② candidates     (유사도 상위 K) ∪ (언급 쌍) → 방향 없는 쌍, 상태 = pending
   ├─ ③ judge          1단계 언급별 · 2단계 근거 구간별 · (별도) 추론 판정 → 점수 (캐시)
   ├─ ④ relations      3단계 쌍 단위 종합 → 관계 + 근거 연결, 재판정 대상과 검수 대기열
   ├─ ⑤ export         graph.db (SQLite) + 검수 기록 적용
   └─ ⑥ eval-relations 검수 표본 채점(개발·확인 표본 분리), 모델 비교, 관계 유형별 주가 동조성
                             │
                        웹앱 (tenksim serve)
```

- CLI: `tenksim mentions | judge | export | serve | eval-relations -c configs/sp500_2024.yaml`
- `tenksim run`은 설정에 `relations:`가 있으면 ①~⑤를 이어서 돌립니다.

### 7.1 ① 이름 언급 (`relations/names.py`, `relations/mentions.py`)

**회사명 사전**
- 0단계: 분석 대상 회사명에서 접미사를 떼어 씁니다. 여기에 별칭 파일 `configs/aliases.yaml`을 더합니다. 사람이 고치는 파일입니다.
- **제품명·자회사명 별칭**도 넣습니다(예: AWS → Amazon, Instagram → Meta, YouTube → Alphabet). 별칭마다 모회사와 적용 기간을 기록합니다. 인수·분사로 소속이 바뀌기 때문입니다.
- 3단계: SEC 전체 등록사 이름(20-F를 내는 해외 기업 포함)과 SEC 밖 외부 기업 목록을 더합니다.

**제외는 이름 단위가 아니라 문맥 단위로 합니다.**
- 사전 조사처럼 Target이나 금융정보 회사를 통째로 빼면 실제 관계(예: 데이터 공급 계약)까지 사라집니다.
- 대신 관계와 무관한 문맥 패턴을 뺍니다. 예: "listed on the Nasdaq", "S&P 500 Index", 신용등급 문구("rated ... by Moody's"), 성과 비교 그래프 설명.
- 일반 단어인 이름(Target, Delta, Block 등)은 정식 명칭이나 앞뒤 문맥(Inc., Corporation, 따옴표 정의 등)이 있을 때만 잡습니다.
- 자기 자신에 대한 언급과 Item 1의 임원 약력 부분("Information about our Executive Officers" 같은 소제목부터 다음 소제목까지)은 뺍니다.

**찾는 방법**
- 이름 수백 개를 글자 트라이로 묶어 정규식 하나로 찾습니다. 476개사에 7초 걸렸습니다.
- 대소문자를 구분하고 단어 경계를 지킵니다.

**근거 구간.** 언급 하나마다 `mention_id`를 붙이고, 그 언급을 담은 근거 구간(`span_id`)을 만듭니다.
- 기본은 한 문장입니다.
- 다음 경우에는 앞 문장을 함께 넣습니다.
  - 경쟁사 목록의 도입문("Our competitors include:"처럼 목록 앞 문장)
  - 대명사가 가리키는 선행 문장
  - 시점을 나타내는 문장("In 2023, ...")

**위치 기준.** 위치는 **정제 텍스트 기준**임을 명시합니다.
- 근거마다 `accession + section + text_hash(정제 텍스트) + char_start/char_end`를 저장합니다.
- 원문(`text_raw`)은 `data/sections`에 이미 있습니다. 원문 위치로 되짚는 매핑은 EDGAR 원문 화면에 강조 표시가 필요해질 때 추가합니다.

**숫자는 '후보'로만 뽑습니다.** 같은 문장에 이름과 "N% of ... revenue(s)/sales"가 함께 나오면 매출 비중 후보로 저장하되, 다음 구조를 씁니다.

| 필드 | 뜻 | 예 (Qualcomm) |
|---|---|---|
| `value` | 수치 | 10 |
| `operator` | `=` / `≥` / `≈` / `<` | `≥` ("10% or more") |
| `period` | 대상 기간 | fiscal 2024 |
| `denominator` | 무엇의 비중인가 | consolidated revenues |
| `subject` | 누구의 비중인가 (한 회사 / 여러 회사 합계) | 각 회사 (Apple, Samsung, Xiaomi 각각) |
| `evidence_id` | 근거 구간 | `span:...` |

- 여러 회사 합계("together represented 34%")나 여러 연도가 섞인 문장처럼 귀속이 불분명하면, 수치 필드를 비우고 문장만 보여줍니다.
- 익명 고객("Customer A ... 13%")도 같은 구조로 저장하고, `anonymous` 노드(5장)에 붙입니다.

**산출물**
- `mentions.parquet`: mention_id, 언급한 회사, 언급된 노드, 매칭된 이름, span_id
- `spans.parquet`: span_id, accession, section, text_hash, 위치, 구간 텍스트
- `figures.parquet`: 매출 비중 후보

### 7.2 ② 후보 (`relations/candidates.py`)

**후보**
- 후보는 방향 없는 쌍 {A, B}이고, 양 끝은 노드 ID 순서로 정렬해 저장합니다.
- 쌍마다 출처를 표시합니다: A 기준 유사도 순위, B 기준 순위, A가 B를 언급했는지, B가 A를 언급했는지.
- 모든 후보는 상태를 가집니다: `pending`(판정 전) → `judged` → `accepted` / `rejected` / `uncertain`(재판정·검수 대기). 0단계 화면의 '판정 전 언급'은 이 상태로 보여줍니다.

**규모 [확인]:** S&P 500 476개사에서 언급 쌍을 포함하면 다음과 같습니다. 처음에는 K=20으로 시작하고, 검수 표본이 생기면 K=10·20·30의 후보 재현율과 추가 판정 비용을 비교해 정합니다.

| K | 판정할 쌍 |
|---|---|
| 10 | 4,031 |
| 20 | 6,737 |
| 30 | 9,421 |

**판정 입력은 두 종류로 나눕니다.** 명시된 근거를 판단할 때 회사 개요나 GICS가 선입견을 줄 수 있어서입니다.

| 입력 | 쓰는 곳 | 내용 |
|---|---|---|
| 근거 입력 | 1·2단계 판정 (7.3) | 근거 구간과 두 회사의 이름만. 개요와 GICS를 넣을지는 시범 단계에서 넣은 것과 뺀 것을 비교해 정함 |
| 추론 입력 | 추론 판정 (7.3) | 두 회사 개요(Item 1 첫 200단어 정도, GICS), 가장 비슷한 청크 두 쌍(기존 `explain_pair`) |

- 근거 구간은 한 회사 쪽에서 최대 6개로 시작합니다. 개수를 고정하기보다 관계 유형별로 골고루 들어가게 고릅니다.
- [확인] 언급 문장은 중앙값 27단어이고 90%가 52단어 이하입니다. 한 쌍의 언급 문장 수는 중앙값 1개이고 90%가 4개 이하입니다.
- 한 요청은 약 1,500~3,000 토큰으로, Jev의 한도(32k) 안에 넉넉합니다. Jev는 무관한 내용이 많으면 정확도가 떨어지므로 짧게 유지합니다.

### 7.3 ③ 판정 (`relations/judge/`)

판정은 세 단계로 나눕니다.

| 단계 | 단위 | 묻는 것 | 누가 |
|---|---|---|---|
| 1 | 언급 | 이 이름이 지정한 회사를 가리키는가 | 모델 |
| 2 | 근거 구간 | 이 구간이 관계의 근거인가, 어떤 관계·방향·시점을 말하는가 | 모델 |
| 3 | 기업 쌍 | 유효한 근거를 합치면 어떤 관계가 성립하는가 | 코드 (7.4) |

추론 판정(`inferred`)은 이와 별도로, 언급이 없는 쌍에 대해 추론 입력으로 묻습니다.

**질문은 회사 이름을 직접 넣어 대칭으로 씁니다.** 'A', 'B' 같은 임의 기호 대신, 질문 템플릿의 `{X}`, `{Y}`에 실제 회사명을 넣습니다. 그러면 어느 회사의 10-K에서 나온 문장이든 같은 질문으로 처리됩니다. 아래는 모두 영어 초안입니다(Jev는 영어가 가장 정확).

**1단계: 언급별** (`Noul`)

| 이름 | 질문 (초안) |
|---|---|
| `m{i}_is_entity` | In mention {i}, does the name "{name}" refer to the company {Y} ({Y_description}), and not to a place, a product line of another company, or a different company with a similar name? |

- 이 질문은 **회사 식별만** 묻습니다. 임원 약력의 Delta처럼 회사는 맞지만 관계의 근거가 아닌 경우는 2단계에서 거릅니다.

**2단계: 근거 구간별** (한 요청에 한 쌍의 모든 구간 질문을 넣고, 질문 이름에 구간 번호를 붙임)

| 이름 | 형식 | 질문 (초안) |
|---|---|---|
| `s{j}_is_relation_evidence` | Noul | Does span {j} describe a business relationship between {X} and {Y}, rather than only a person's career history or a list of names without any relationship stated? |
| `s{j}_competes` | Noul | Does span {j} state that {X} and {Y} compete, or list one of them as a competitor of the other? |
| `s{j}_x_supplies_y` | Noul | Does span {j} state that {X} sells, provides, or manufactures products or services for {Y}? |
| `s{j}_y_supplies_x` | Noul | Does span {j} state that {Y} sells, provides, or manufactures products or services for {X}? |
| `s{j}_partner` | Noul | Does span {j} state a partnership, joint venture, licensing, distribution or resale, or co-development arrangement between {X} and {Y}? |
| `s{j}_x_owns_y` | Noul | Does span {j} state that {X} owns shares or an equity stake in {Y}? |
| `s{j}_y_owns_x` | Noul | Does span {j} state that {Y} owns shares or an equity stake in {X}? |
| `s{j}_status` | Choice | Is the relationship in span {j} current at the time of the filing, historical (ended or in the past), planned (announced but not yet in effect), or unclear? |
| `s{j}_partner_type` | Choice | (partner일 때) licensing / distribution or resale / joint venture / co-development / other |

바뀐 점과 이유:
- **공급 방향은 대칭 문구 두 개로 나눴습니다.** 전에는 "relies on"을 넣었는데, 이 표현은 의존을 말할 뿐 실제 공급 거래를 뜻하지 않을 수 있어 뺐습니다.
- **지분도 방향별 질문 두 개로 나눴습니다.** 전 질문은 누가 누구의 지분을 가졌는지 알 수 없었습니다.
- **다중 관계라서 유형별 `Noul`을 씁니다.** 반대로 시점처럼 서로 배타적인 상태는 `Choice`를 씁니다.
- **서로 다른 형식의 점수에는 같은 임계값을 쓰지 않습니다.**
- **A/B를 바꿔 묻는 방향 일관성 검사**는 시범 단계에서만 합니다. 모든 요청을 두 번 돌리지는 않습니다.

**추론 판정** (언급이 없는 쌍, 추론 입력)

| 이름 | 형식 | 질문 (초안) | 쓰임 |
|---|---|---|---|
| `competes_likely` | Noul | Based on the two business descriptions, do {X} and {Y} likely sell similar products or services to the same customers? | competitor, inferred (화면 표시) |
| `supply_chain_likely` | Noul | Based on the two business descriptions, is one company plausibly a supplier to the other? | 탐색 후보로만 저장, 기본 화면에 숨김 |

**공통 출력 형식.** Jev와 텍스트 LLM이 같은 형식으로 답하게 해서 비교와 화면 표시가 틀어지지 않게 합니다.

```python
@dataclass
class Answer:
    question: str                 # "s2_x_supplies_y"
    decision: Literal["yes", "no", "abstain"]
    score: float | None           # Jev 점수. 텍스트 LLM은 None (예를 1.0으로 바꾸지 않음)
    evidence_ids: list[str]       # 판단에 쓴 근거 구간. Jev는 질문이 가리키는 구간, LLM은 스스로 고른 구간
    model_id: str                 # "jev-1.13.0", "llm:<모델명>"
    question_version: str

class RelationJudge(Protocol):
    model_id: str
    def judge(self, requests: list[JudgeRequest]) -> list[list[Answer]]: ...
```

- Jev의 `decision`은 점수와 질문 유형별 임계값으로 코드가 정합니다.
- 캐시는 `data/relations/judgements.sqlite`이고, 키는 (모델 ID, 질문 버전, 입력 해시)입니다.

**Jev 호출 예** (UPS–Amazon, 2단계)

```python
resp = client.system_one(
    model="jev-1.13.0",
    state={
        "companies": {"X": "United Parcel Service (UPS)", "Y": "Amazon.com (AMZN)"},
        "spans": [
            {"id": 1, "source": "UPS 10-K (filed 2024), Item 1",
             "text": "For the year ended December 31, 2023, one customer, Amazon.com, Inc. and its affiliates, represented approximately 11.8% of our consolidated revenues, substantially all of which was within our U.S. ..."},
        ],
    },
    questions=span_questions(span_ids=[1], x="United Parcel Service", y="Amazon.com"),  # s1_competes, s1_x_supplies_y, ...
)
scores = {name: (a.noul if hasattr(a, "noul") else a.choice) for name, a in resp.answers.items()}
```

**텍스트 LLM 어댑터** (비교와 재판정용)
- 같은 질문에 JSON 스키마로 답하게 합니다. 근거는 주어진 구간 번호 중에서만 고르게 하고, 한국어 한 줄 이유를 덧붙입니다.
- 번호로만 인용하므로, 지어낸 인용이 화면에 나올 수 없습니다.
- 비용이 커지므로 기본값에서는 끕니다. 비교 표본과 재판정 대상(7.4)에만 씁니다.

### 7.4 ④ 관계 만들기 (`relations/merge.py`)

**3단계 종합 규칙**
1. 1단계에서 `is_entity`가 기각된 언급의 구간은 버립니다.
2. 2단계에서 `is_relation_evidence`가 기각된 구간도 버립니다.
3. 남은 구간 가운데 어떤 관계 질문이 채택되면, 그 관계를 만들고 **채택된 구간만** 근거로 연결합니다(`edge_evidence`).
4. 관계의 시점은 연결된 구간들의 `status`로 정합니다. 구간끼리 시점이 다르면(예: 한 구간은 current, 다른 구간은 historical) `unclear`로 두고 검수 대기열에 넣습니다.
5. 공급 양방향이 모두 채택되면 둘 다 저장합니다.
6. 언급이 없는 쌍은 `competes_likely`가 채택되면 `competitor(inferred)`로, 아무것도 채택되지 않고 유사도 상위 K면 `similar`로 둡니다. `supply_chain_likely`는 탐색 후보 표시만 합니다.
7. 모순 검사: 같은 구간에서 '관계의 근거가 아님'과 '공급을 말함'이 함께 채택되는 식의 모순은 `uncertain`으로 보냅니다(Jev는 질문 간 일관성을 보장하지 않음).

**임계값.** 질문 유형별로 개발 표본(7.6)에서 정합니다. 시작값의 예는 다음과 같습니다.
- 채택: 점수 0.8 이상
- 기각: 0.3 미만
- 그 사이: `uncertain`

**재판정 대상.** 채택 후보만 다시 보면 Jev가 놓친 관계(거짓 음성)는 되살릴 수 없습니다. 그래서 재판정 대상을 이렇게 잡습니다.
- 임계값 근처(`uncertain`)
- 맥락 단어("competitor", "customer", "supplier", "% of revenue" 등)가 있는데 점수가 낮은 경우
- 1단계 식별이나 방향 판정이 서로 충돌하는 경우
- **기각된 결과에서 무작위로 뽑은 감사 표본.** 이 표본으로 거짓 음성 비율도 추정합니다.

재판정은 텍스트 LLM이 먼저 하고, 그래도 불확실하면 검수 대기열로 보냅니다.

### 7.5 ⑤ 저장 구조 (`relations/export.py`)

**graph.db** (SQLite, 파이프라인이 만들고 웹앱은 읽기만 함)

| 테이블 | 주요 컬럼 |
|---|---|
| `nodes` | node_id, kind(company/external/anonymous), cik, ticker, name, gics_sector, gics_sub_industry |
| `filings` | accession, node_id, form, filing_date, period_of_report, filing_url |
| `documents` | accession, section, status, text_hash, 정제 텍스트 (근거 위치의 기준, 원문 확인 화면용) |
| `mentions` | mention_id, doc_node, target_node, matched_name, span_id |
| `spans` | span_id, accession, section, text_hash, char_start, char_end, text |
| `figures` | figure_id, span_id, subject, value, operator, period, denominator |
| `candidates` | pair_key(정렬된 두 node_id), 출처 표시, 유사도 순위, status(pending/judged/accepted/rejected/uncertain) |
| `candidate_spans` | pair_key, span_id, 순서, 단서 단어 (쌍별 판정 입력) |
| `answers` | 7.3의 `Answer` 전부 (질문별 판정 기록) |
| `edges` | edge_id, src, dst, relation, subtype, directed, status(시점), basis, score, model_id, question_version, similarity_pct, resid_corr, as_of(accession), review_state |
| `edge_evidence` | edge_id, span_id, 판정한 질문 |
| `meta` | 설정 이름, 파이프라인 버전, 스키마 버전, 생성 시각 |

- 방향 없는 관계(`competitor`, `partner`, `similar`)는 src·dst를 node_id 순서로 정렬해 저장합니다.
- 기업 노드에 공시를 하나만 붙이지 않고, 공시(`filings`)와 근거(`spans`)가 각자 accession을 가집니다. 여러 연도로 넓히기 쉽게 하려는 것입니다.

**검수 기록**은 별도 파일 `data/relations/reviews.sqlite`에 둡니다. graph.db는 만들 때마다 새로 써지기 때문입니다.

| 컬럼 | 뜻 |
|---|---|
| src, dst, relation, direction | 자연 키 (방향 없는 관계는 정렬) |
| as_of | 검수 대상 공시(accession) 또는 연도 |
| evidence_hash | 검수 당시 연결된 근거 구간들의 해시 |
| schema_version | 검수 당시 스키마 버전 |
| verdict | accept / reject / retype / redirect(방향 수정) / add(누락 관계 추가) / add_evidence(근거 추가) |
| new_relation, new_direction, new_status | 수정 내용 |
| note, reviewed_at | 메모, 시각 |
| model_score, model_id | 검수 당시 모델 판정 |

**검수 적용 규칙** (export 때마다)
1. 사람의 검수가 모델 판정보다 우선합니다. 검수 기록은 절대 덮어쓰지 않고 새 줄로만 쌓습니다(같은 키는 최신 기록 적용).
2. `retype`은 기존 관계를 거절하고 새 관계를 추가한 것으로 처리합니다. `redirect`도 같습니다.
3. 근거가 바뀌면(`evidence_hash`가 다르면) 그 검수는 자동 적용하지 않고 `review_state = needs_recheck`로 표시해 다시 검수 대기열에 올립니다.
4. 스키마 버전이 바뀌어 관계 유형이 달라졌다면 변환 규칙을 적용하고, 변환할 수 없으면 역시 재검토로 보냅니다.
5. 검수로 추가된 관계(`add`)는 모델 판정이 없어도 표시하고, `basis`는 검수자가 붙인 근거로 정합니다.

### 7.6 ⑥ 평가 (`relations/evaluate.py`)

**정답 만들기: 미리 쓰는 정답셋 대신 표본 검수** (v2.1에서 바꿈)

저장소 주인이 관계 목록을 처음부터 쓰는 일은 부담이 너무 큽니다. 그렇다고 AI가 만든 판정을 정답으로 쓰면 평가가 모델끼리의 비교가 됩니다. 그래서 **파이프라인이 표본 회사의 근거 구간을 뽑고, 저장소 주인이 그 구간을 검수**합니다. 검수 기록이 곧 정답 라벨입니다.

- **검수 단위는 근거 구간입니다.** "이 문장은 X와 Y의 어떤 관계를 말하나(경쟁/공급 방향/협력/지분/관계 아님, 시점)"를 고릅니다. 판정 모델의 2단계 질문과 같은 단위라 그대로 채점에 씁니다. 쌍 단위 정답은 구간 라벨을 합쳐 만듭니다.
- **표본 회사는 채택된 것만이 아니라 언급 구간 전부를 검수합니다.** 모델이 기각한 구간도 포함되므로, 정밀도뿐 아니라 판정 단계의 재현율(놓친 관계)도 잴 수 있습니다.
- **가림 검수:** 표본 검수에서는 모델 판정을 가리고 원문만 보여줍니다. 모델 답을 보고 '승인'만 누르면 판단이 모델 쪽으로 끌려가기 쉽기 때문입니다. 전체 실행 뒤의 운영 검수(8.1의 검수 화면)에서는 모델 판정을 보여줍니다.
- **빠진 관계 추가(선택):** 표본 회사의 관계도를 보고 아는 관계가 빠졌으면 추가합니다(`add`). 이름 사전 누락이나 공시 범위 밖 관계를 찾는 용도이고, 필수는 아닙니다.
- 텍스트 LLM의 재판정 결과는 정답으로 쓰지 않습니다. 비교 대상일 뿐입니다.

**표본 두 개로 나눕니다.** 질문을 다듬은 표본으로 합격 여부를 정하면 결과가 좋게 나옵니다.

| 표본 | 용도 | 규모 (초안) |
|---|---|---|
| 개발 표본 | 질문 문구, 판정 입력 구성, 임계값 조정 | 약 10개사 |
| 확인 표본 | 질문·임계값을 고정한 뒤 새로 뽑아 검수. 합격 기준(아래)을 넘으면 전체 실행을 허가 | 새 약 10개사 |

- 두 표본은 **회사 단위로** 나누고, 섹터별로 층화해서 고릅니다. 같은 10-K의 구간이 두 표본에 함께 들어가지 않게 합니다.
- **검수량 [확인]:** 검수 단위(근거 구간 × 언급된 회사)는 S&P 500 전체에 3,617개입니다. 한 회사가 관련된 단위(자기 10-K의 언급 + 남의 10-K가 그 회사를 언급한 것)는 중앙값 6개, 상위 10%가 32개 이상이고, Microsoft처럼 많이 언급되는 회사는 200개가 넘습니다. 10개사 표본은 보통 100~200개로, 건당 15~20초면 1시간 안팎입니다. 많이 언급되는 회사는 회사당 40개까지만 무작위로 뽑습니다.
- 확인 표본에서 떨어지면 개발 표본을 늘려 고치고, 확인 표본은 **새로** 뽑습니다(이미 본 확인 표본으로 다시 재지 않음).

**정답의 두 범위.** 검수한 구간에서 나온 관계는 '확인 가능한 관계'(해당 연도 Item 1·1A에 있음)입니다. 저장소 주인이 추가한 관계는 '알려진 관계'로 표시합니다. 파이프라인의 추출 성능은 '확인 가능한 관계'로 재고, '알려진 관계'와의 차이는 공시 범위의 한계로 따로 보고합니다.

**평가 항목 분해.** 어디서 놓쳤는지 알 수 있게 단계별로 나눠 잽니다.

| 측정 대상 | 확인할 내용 |
|---|---|
| 후보 재현율 | 확인 가능한 정답 관계 가운데 후보 생성 단계가 찾은 비율 (K=10·20·30 비교) |
| 근거 검색 재현율 | 관계를 입증하는 구간이 판정 입력에 포함된 비율 |
| 판정 성능 | 올바른 근거가 주어졌을 때 유형·방향·시점을 맞히는 정도 (1단계 식별 정확도 포함) |
| 전체 성능 | 후보 누락부터 최종 판정 오류까지 포함한 결과 |

**표본과 불확실성**
- 표본 회사 안에서는 구간을 전부 검수하므로 가중이 필요 없습니다. 전체 실행 뒤 운영 검수에서 정밀도를 다시 잴 때는 관계를 유형과 점수 구간으로 층화해 뽑고, 층별 추출 확률로 가중합니다.
- 기각 쪽 감사 표본(7.4)으로 거짓 음성 비율을 추정합니다.
- 시범 규모로는 결론을 확정하기 어렵습니다. 예를 들어 한 유형에서 60건 중 54건을 맞혀도 95% 신뢰구간은 약 80~95%입니다(Wilson). 같은 회사의 쌍끼리는 독립이 아니므로, 구간은 **회사 단위 부트스트랩**으로도 계산합니다.

**지표**
- 유형별 정밀도·재현율·F1, 방향·시점 정확도
- 1단계 식별 정확도
- 점수 보정: 점수 구간별 실제 정답 비율, Brier 점수
- 모델 간 일치도
- 1,000쌍당 비용과 시간

**모델 비교.** 같은 입력으로 Jev와 텍스트 LLM 1~2개를 돌려 확인 표본에서 비교합니다.

**합격 기준 (초안).** 모든 기준에 점추정치와 함께 유형별 표본 수, 신뢰구간, 자동 채택 비율(검수 없이 채택되는 비율)을 적습니다.
- 화면에 실선으로 보여주는 `disclosed` 관계의 정밀도가 0.9 이상이어야 합니다. 신뢰구간 하한도 함께 봅니다.
- Jev를 기본 모델로 쓸지는 F1 차이와 재현율을 같이 봅니다. Jev의 재현율이 낮으면 기각 쪽 재판정(7.4)을 넓히는 비용과 텍스트 LLM을 기본으로 쓰는 비용을 비교해 정합니다.
- 시범 규모의 결과는 잠정 판단으로 기록하고, 3단계에서 표본을 늘려 다시 확인합니다.

**연구 분석** (기존 `returns.py`를 다시 씁니다)
- 관계 유형별(경쟁, 공급망, 협력, 사업 유사만) 잔차 상관을 비교합니다.
- 교란 요인을 함께 다룹니다: 섹터, 회사 규모, 기존 유사도, 공시 성향(이름을 많이 적는 회사). 한 쌍이 여러 관계를 가지거나 같은 회사가 여러 쌍에 반복되는 의존성도 처리합니다(회사 단위 부트스트랩).
- 일별 상관의 부호만으로 경쟁 효과를 판별하기는 어렵습니다. 그래서 이벤트 스터디로 넓힙니다.
  - 실적 발표일의 반응으로, 같은 방향인 전염 효과와 반대 방향인 경쟁 효과를 나눕니다.
  - 발표 시각(장 전·장 후), 같은 날 동시 발표, 시장·산업 충격을 따로 다뤄야 합니다.

## 8. 웹앱 [제안]

### 8.1 화면

1. **검색:** 티커나 이름으로 회사를 찾습니다.
2. **관계도 (메인):** 고른 회사를 가운데 두고 1단계 이웃을 보여줍니다. 아래는 NVIDIA 예시이고, 이름은 모두 실제 10-K 문장에 있습니다.

   ```
                경쟁            AMD · Intel · Broadcom · Arista · Cisco · HPE
                경쟁(자체 칩)    Alphabet · Amazon · Microsoft
                                     │
    공급  TSMC · Samsung  ─────▶   NVIDIA   ─────▶  고객  ○ Customer A (매출 13%)
                                     │
                협력            ServiceNow · HPE   (상대 회사 10-K에 적힌 관계)
         ┄┄ 사업 유사 (유사도만): 바깥 링에 흐리게
   ```

3. **근거 패널:** 관계 선을 누르면 다음이 나옵니다.
   - 근거 구간 원문(상대 이름 강조)과 출처(회사, 제출 연도, 항목), EDGAR 원문 링크
   - 판정 점수, 판정 모델, 시점(현재·과거·계획)
   - 유사도 백분위, 2025년 잔차 상관, 매출 비중(연산자 포함, 예: "≥10%")
   - 한국어 한 줄 설명은 버튼을 누를 때만 만들고, 만든 뒤 저장합니다.
4. **검수:** 불확실한 관계부터 보여줍니다. 대기열에는 `uncertain`, 재판정 뒤에도 불확실한 것, `needs_recheck`, 기각 쪽 감사 표본이 들어갑니다.
   - 승인, 거절, 유형 수정, 방향 수정, 근거 추가, 빠진 관계 추가를 할 수 있습니다.
   - 키보드 단축키를 씁니다.
   - 결과는 `reviews.sqlite`에 쌓입니다.
5. **판정 전 후보 보기 (0단계):** 판정 모델을 붙이기 전에도 후보와 언급 원문을 확인할 수 있게 합니다. 표본 검수(7.6)도 이 화면을 넓혀서 합니다.
6. **나중에:** 전체 시장 지도(섹터를 넘는 관계만 보기), 연도별 관계 변화.

### 8.2 시각 규칙

- **배치:** 가치사슬 방향을 따릅니다. 공급사는 왼쪽, 고객사는 오른쪽, 경쟁사는 위, 협력·지분은 아래, 사업 유사는 바깥 링입니다.
- **여러 역할인 회사(예: HPE는 경쟁과 협력):**
  - 점은 하나만 그립니다.
  - 위치는 우선순위로 정합니다: 공급·고객 > 경쟁 > 협력·지분 > 사업 유사.
  - 나머지 역할은 점 옆 배지로 표시합니다.
  - 여러 관계선은 곡선으로 나눠 겹치지 않게 합니다.
- **양방향 공급:** 화살표 두 개로 그립니다.
- **선:**
  - 색은 관계 유형입니다.
  - 모양은 근거 수준입니다: `disclosed`는 실선, `inferred`는 점선, `similarity`는 흐린 점선.
  - 굵기는 일정하게 둡니다. 굵기를 점수로 쓰면 거래의 중요도로 오해하기 쉽습니다. 점수와 매출 비중은 숫자나 배지로 표시합니다.
  - 과거·계획 관계는 흐리게 그리고 배지를 붙입니다.
- **점:**
  - 색은 GICS 섹터입니다. 섹터를 넘는 연결이 바로 보이게 하려는 것입니다.
  - 회색은 외부 기업, 빈 원은 익명 공시입니다.
- **필터:** 관계 유형, 근거 수준, 시점, 최소 점수, '섹터가 다른 관계만'.
- **이동:** 점을 누르면 그 회사가 가운데로 옵니다. 외부 기업과 익명 노드도 누를 수 있습니다. '한 단계 더 펼치기'로 2단계 이웃(예: 공급사의 공급사)도 봅니다.

### 8.3 기술 구성

- **백엔드:** FastAPI로, 이 저장소 패키지 안(`src/tenksim/app/`)에 둡니다.
  - 127.0.0.1에서만 실행합니다.
  - graph.db는 읽기만 하고, reviews.sqlite에만 씁니다.
- **프론트엔드:** React + Vite + TypeScript로, `web/` 폴더에 둡니다.
  - 빌드 결과를 FastAPI가 서빙합니다.
  - 저장소 주인이 다른 로컬 웹앱에서 이미 쓰고 있는 구성입니다.
- **그래프:** Cytoscape.js를 씁니다.
  - 관계 유형별 선 스타일과 좌표 지정(`preset`) 배치를 지원하고, 수십~수백 노드의 에고 네트워크에 맞습니다.
  - 수천 노드의 전체 시장 지도가 필요해지면 WebGL 기반 Sigma.js를 따로 씁니다.

**API.** 외부 기업과 익명 노드도 탐색할 수 있게 티커가 아니라 `node_id`를 주소로 씁니다.

| 요청 | 하는 일 |
|---|---|
| `GET /api/nodes?q=` | 회사 검색 (티커·이름) |
| `GET /api/nodes/{node_id}/graph?depth=1&types=&basis=&status=&min_score=` | 노드와 관계 |
| `GET /api/edges/{edge_id}` | 근거와 지표 |
| `GET /api/candidates?node_id=&status=pending` | 판정 전·보류 후보 |
| `GET /api/review/queue` | 검수 대기열 |
| `POST /api/reviews` | 검수 저장 |
| `POST /api/edges/{edge_id}/explain` | 한국어 설명 생성 (버튼 전용, 결과 캐시) |

## 9. 저장소 변경 [제안]

```
src/tenksim/
  relations/
    names.py        회사명 사전, 별칭(모회사·적용 기간), 문맥 제외 규칙
    mentions.py     문장 분리, 이름 매칭, 임원 약력 제외, 근거 구간, 숫자 후보
    candidates.py   유사도 상위 K ∪ 언급 → 후보 쌍 + 판정 입력(근거 입력·추론 입력)
    judge/
      base.py       Answer, RelationJudge, 질문 템플릿, 캐시
      jev.py        Jev 어댑터 (typesafe-sdk)
      llm.py        텍스트 LLM 어댑터 (JSON 출력)
    merge.py        3단계 종합, 모순 검사, 재판정 대상 선정
    export.py       graph.db, 검수 적용 규칙
    evaluate.py     개발·확인 표본 채점, 단계별 재현율, 모델 비교, 유형별 주가 동조성
  app/server.py     FastAPI
web/                React + Vite + Cytoscape.js
configs/aliases.yaml
tests/              test_mentions.py, test_candidates.py, test_merge.py, test_reviews.py, test_judge_cache.py, test_app.py
```

**의존성:** 새 의존성은 extra로 분리해 기존 설치에 영향을 주지 않습니다.
- `relations`: typesafe-sdk, 텍스트 LLM SDK
- `app`: fastapi, uvicorn

**설정:** 기존 설정 파일에 `relations:` 절을 추가합니다.

```yaml
relations:
  similarity: tfidf-openai        # 1차 후보에 쓸 유사도 (방법·변형·앙상블 이름)
  top_k: 20
  aliases: configs/aliases.yaml
  spans: {max_per_side: 6, include_lead_in: true}
  judge:
    model: jev-1.13.0             # 또는 llm:<모델명>
    questions: v2
    evidence_input_overview: false   # 근거 판정에 회사 개요를 넣을지 (시범에서 비교)
    thresholds: {default: {accept: 0.8, reject: 0.3}}   # 질문 유형별로 덮어씀
    recheck:
      model: null                 # 예: llm:<모델명>
      rejected_audit_rate: 0.05   # 기각 결과 중 감사 표본 비율
  show_inferred_supplies: false
  samples: configs/relations_samples.yaml   # 개발·확인 표본 회사 목록
```

**테스트:** 기존처럼 네트워크 없이 돕니다.
- 판정 모델은 가짜 어댑터로 바꿔 끼웁니다.
- 검수 적용 규칙(재검토 전환, retype 처리)은 따로 테스트합니다.
- Jev와 LLM을 실제로 부르는 테스트는 `live` 표시를 붙여 따로 돌립니다.

**키:** `TYPESAFE_API_KEY`와 (선택) 텍스트 LLM 키는 `.env`에만 둡니다.

## 10. 단계별 계획 [제안]

스키마·저장 구조를 먼저 확정합니다. 표본 검수는 화면에서 하므로, 0단계에서 후보·원문 확인 화면과 가림 검수 화면을 함께 만듭니다.

| 단계 | 할 일 | 끝났다는 기준 | 모델 비용 |
|---|---|---|---|
| **0. 스키마 확정과 뼈대** | 관계·근거·판정 상태·검수 키 확정(이 문서 5·7.5장). mentions·candidates 정식화와 테스트. FastAPI와 '판정 전 후보 보기'·근거 확인·가림 검수 화면 | NVIDIA를 열면 후보와 언급 원문이 위치·출처와 함께 보이고, 원문 추적이 맞음 | 없음 |
| **0 끝: 개발 표본 검수** | 저장소 주인이 개발 표본(약 10개사)의 언급 구간을 가림 검수 화면에서 라벨링 (7.6) | 개발 표본 라벨 완료 | 없음 |
| **1. 판정과 평가** | Jev·텍스트 LLM 어댑터, 질문 v2, 캐시와 `eval-relations`를 함께 구현. 개발 표본으로 질문·임계값을 정하고, 확인 표본을 새로 뽑아 검수 | 확인 표본이 합격 기준을 넘음 → 전체 실행 허가. 기본 판정 모델 결정 | Jev 몇 달러, 텍스트 LLM 비교 수십 달러 이하 |
| **2. 전체 판정과 화면 완성** | 평가를 통과한 설정으로 476개사 전체 판정. 검수 화면, 관계도 표현(8.2) 완성. 유형별 주가 동조성 분석, 리포트(`reports/relations_sp500_2024.md`) | 검수 적용 규칙이 동작하고, 관계도에 유형·시점·근거가 표시됨 | Jev 약 $1~2 + 재판정 |
| **3. 확장** | 10-K를 내는 전 종목(약 4~5천 개사), SEC 전체 이름 사전, Item 7·재무제표 주석의 고객 공시, 외부 기업 노드, 여러 연도(2019~2025), 전체 시장 지도, 이벤트 스터디. 표본을 늘려 성능 재확인 | 공급망이 S&P 500 밖까지 이어지고, 연도별 변화를 볼 수 있음 | Jev 연도당 약 $10~20 |

**0단계 세부 순서와 진행 상황**

| | 할 일 | 상태 |
|---|---|---|
| 0-1 | 이름 언급(`tenksim mentions`): 회사명 사전·별칭·분사 시점, 문맥 제외, 임원 약력 제외, 근거 구간·도입문, 익명 고객, 매출 비중 후보 | 완료 (2026-09-24) |
| 0-2 | 후보(`tenksim candidates`): 유사도 상위 K ∪ 언급 쌍, 상태 `pending`, 쌍별 근거 구간 선택(10-K 쪽마다 최대 6개, 단서 종류별로 번갈아). K=20에서 6,864쌍(유사도만 5,327 · 둘 다 775 · 언급만 762, 외부·익명 기업 포함 153). 추론 입력(회사 개요, 비슷한 청크)은 판정과 함께 1단계에서 | 완료 (2026-09-24) |
| 0-3 | `graph.db` 내보내기(`tenksim export`): 스키마 버전 1 확정, 노드·공시·정제 문서·언급·구간·매출 비중·후보·쌍별 근거 구간. 판정·관계 테이블은 빈 채로 만들어 둠. 참조 무결성과 근거 위치(정제 텍스트 기준)를 내보낼 때 검사. S&P 500 기준 72MB | 완료 (2026-09-24) |
| 0-4a | 웹앱 뼈대(`tenksim serve`, FastAPI + React/Vite): 회사 검색, 판정 전 후보 목록(출처·양쪽 유사도 순위·방향별 언급 수, 필터), 후보 쌍의 근거 문장(판정 입력 여부, 제외 이유, 이름 강조), 매출 비중 후보, 정제 본문 안에서 근거 보기 | 완료 (2026-09-24) |
| 0-4b | 가림 검수 화면과 `reviews.sqlite`(구간 라벨), 표본 회사 뽑기 | |

## 11. 규모와 비용 추정

가정: 요청 하나에 쌍 하나를 넣고, 요청당 입력은 약 3,000 토큰입니다(근거 구간 + 질문).

| | S&P 500 (476개사, K=20) | 전 종목 (약 4,500개사, K=20, 추정) |
|---|---|---|
| 판정할 쌍 | 6,737 [확인] | 약 6만~7만 |
| 입력 토큰 (기본 판정 1회) | 약 2천만 | 약 2억 |
| Jev (100만 토큰당 $0.042) | 약 $0.85 | 약 $8 |
| 텍스트 LLM (입력 100만 토큰당 $3 가정, 출력 별도) | 약 $60 | 약 $600 |
| Jev 소요 시간 (분당 1,200 요청 한도를 모두 쓴다고 가정) | 약 6분 | 약 1시간 |

**위 표에 들어가지 않은 비용**
- 구간별 질문은 구간 수에 비례해 질문 토큰이 늘어납니다.
- 추론 판정용 별도 요청, 재판정, 재시도, 질문 개발 중의 반복 실행이 더 듭니다.
- 텍스트 LLM의 출력 비용도 따로입니다.
- 이를 모두 합쳐도 S&P 500 규모에서 Jev 비용은 몇 달러 수준으로 봅니다. 텍스트 LLM 비용은 재판정 범위(7.4)에 따라 크게 달라집니다.

**그 밖의 규모**
- S&P 500 규모에서는 비용 차이가 작으니 정확도로 모델을 고르면 됩니다. Jev의 가격 이점은 전 종목과 여러 연도로 넓힐 때 커집니다.
- 전 종목으로 넓히면 edgartools 캐시가 약 20~25GB가 됩니다(지금 500개사에 2.5GB). 수집에는 1~2시간 걸릴 것으로 봅니다.

## 12. 리스크와 대응

| 리스크 | 대응 |
|---|---|
| Jev가 이 과제에서 부정확함 (자체 측정에서도 최상위 모델보다 약 6%p 낮음) | 확인 표본으로 비교한 뒤 결정. 기각 쪽 감사 표본으로 거짓 음성 확인. 어댑터로 교체 가능 |
| Jev 점수가 이 과제에서 보정되지 않음 | 점수 보정을 확인하기 전까지 확률로 표시하지 않음. 임계값은 개발 표본으로 정함 |
| Jev가 신생 서비스임 (얼리 액세스라 가격·API가 바뀌거나 중단될 수 있음) | 모델 버전 고정, 결과 캐시, 텍스트 LLM으로 대체 가능 |
| 질문 사이 답이 모순됨 | 코드에서 모순 검사 후 `uncertain` 처리 |
| 이름 매칭 오탐 (동명이인, 지명, 임원 약력) | 1단계 식별 질문, 2단계 관계 근거 질문, 임원 약력 제외, 별칭 파일 수동 관리 |
| 한쪽만 공시함 (언급의 83%) | 쌍 단위로 두 회사의 근거를 합쳐 판정 |
| 과거·계획 관계가 현재 관계와 섞임 | 구간별 시점 판정, 화면에서 구분 표시 |
| 매출 비중을 잘못 귀속함 | 연산자·기간·대상 구조로 저장, 불분명하면 비움 |
| 모델의 사전지식이 섞임 (10-K에 없는 관계를 '알고' 답함) | 근거 판정은 근거 입력만 사용. 사전지식이 섞일 수 있는 판단은 `inferred`로 따로 표시하고, 공급 추론은 숨김 |
| 검수 기록이 새 그래프에 잘못 적용됨 | 근거 해시·스키마 버전 확인, 달라지면 재검토로 전환 |
| 공시 범위의 한계 (Item 1·1A 밖의 고객 공시, 익명 공시) | 정답을 '알려진 / 확인 가능'으로 나눠 따로 보고. 3단계에서 Item 7과 주석 추가 |
| 평가가 낙관적으로 나옴 | 개발·확인 표본을 회사 단위로 분리, 확인 표본은 한 번만 사용, 회사 단위 부트스트랩 |
| 기존 한계 (현재 기준 S&P 500 목록·GICS, 생존 편향) | methodology.md 2장 참고. 여러 연도로 넓힐 때 과거 구성종목이 필요 |

## 13. 미결정 사항 (저장소 주인)

1. 관계 유형의 정의, 하위 속성, 이름 (5장).
2. 시점 구분(`current / historical / planned / unclear`)이 충분한지.
3. 개발·확인 표본 회사 선정과 규모 (7.6).
4. Jev API 키 발급 (`.env`의 `TYPESAFE_API_KEY`).
5. 비교와 재판정에 쓸 텍스트 LLM (Claude 계열 또는 OpenAI 계열).
6. 임계값을 정밀도 우선으로 잡을지, 자동 채택 비율을 얼마까지 받아들일지.

## 14. 하지 않기로 한 것과 이유

- **유사도만으로 후보 만들기:** 공급망 언급의 상당 부분이 후보에서 빠집니다(3장).
- **쌍 전체에 대한 답 하나로 판정하기:** 어느 문장이 근거인지, 어느 문장을 버려야 하는지 알 수 없습니다(7.3).
- **텍스트 LLM으로 모든 쌍 판정:**
  - S&P 500은 감당할 만합니다(약 $60). 하지만 전 종목과 여러 연도로 가면 수백~수천 달러가 됩니다.
  - LLM이 말로 내는 확신도는 보정이 약한 편이라, 검수 순서를 정하기도 어렵습니다.
  - 그래서 비교 기준과 재판정에만 씁니다.
- **LLM이 10-K 전체를 읽고 관계를 자유롭게 뽑기:**
  - 비용이 크고, 지어낸 인용이나 사전지식이 섞이는 것을 막기 어렵습니다.
  - 대신 코드가 후보와 근거를 고르고, 모델은 판정만 합니다.
  - 예외로, 사전에 없는 외부 기업 이름을 찾는 데는 3단계에서 회사당 한 번 쓰는 방안을 검토합니다.
- **사업 설명만으로 공급 관계 표시하기:** 초기 버전에서는 탐색 후보로만 두고 숨깁니다(5장).
- **이름 단위로 통째 제외하기:** 실제 관계까지 지웁니다. 문맥 단위로 제외합니다(7.1).
- **다른 웹앱과의 연동 설계:** 범위 밖입니다(2장).

## 15. 남은 검토 질문

1차 검토에서 답을 받은 질문(후보 범위, Noul 대 Choice, A/B 교환, 근거 길이, 웹앱 도구)은 반영했습니다. 남은 질문은 다음과 같습니다.

1. **3단계 판정 구조:** 언급별 → 구간별 → 쌍별 종합 규칙(7.4)에 빈틈이 있을까요? 특히 시점이 서로 다른 구간을 합치는 규칙이 적절한가요?
2. **질문 문구 v2(7.3):** Jev가 '글자 그대로' 잘못 해석할 만한 곳이 남아 있을까요? 예: `is_relation_evidence`에서 "a list of names"가 경쟁사 목록까지 배제하지는 않을까요?
3. **근거 입력에서 회사 개요 빼기:** 이름만 주면 1단계 식별(같은 이름의 다른 회사 구분)이 오히려 어려워지지 않을까요? 1단계에만 짧은 회사 설명을 주는 방식이 나을까요?
4. **표본 검수 방식(7.6):** 미리 쓰는 정답셋 대신 표본 회사의 언급 구간을 가림 검수합니다. 이름 언급이 없는 관계(`inferred`)와 이름 사전이 놓친 관계는 이 방식으로 재현율을 잴 수 없는데, 부담을 크게 늘리지 않고 보완할 방법이 있을까요? 표본 각 10개사로 유형별 판단이 가능할까요?
5. **검수 적용 규칙(7.5):** 근거 해시가 바뀐 이유가 사소한 정제 변경일 때도 재검토로 보내면 부담이 커집니다. 이를 줄일 방법이 있을까요?
6. **연구 설계:** 공시 성향(이름을 많이 적는 회사)을 통제하는 구체적인 방법은?

## 16. 참고

**Jev** (2026-09-24 확인)
- TypeSafe AI 공식 문서: [Models](https://docs.typesafe.ai/models), [Quickstart](https://docs.typesafe.ai/introduction/quickstart), [Python SDK](https://docs.typesafe.ai/sdk/python/api/clients/sync), [알려진 한계(jev-1.13)](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- Simon Willison, [Jev introduces a new shape of LLM](https://simonwillison.net/2026/Sep/21/jev/) (2026-09-21)
- [Jev (AI model) — Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model))
- DataCamp, [Jev: TypeSafe's System One Model](https://www.datacamp.com/blog/system-one-models-jev): 개발사 자체 벤치마크 표의 출처
- DEV Community, [How to Use Jev: A practical guide](https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e): 알려진 약점과 사용 패턴

**웹앱**
- Cytoscape.js [preset 배치](https://js.cytoscape.org/#layouts/preset)

**관련 연구** (methodology.md의 참고 문헌과 같음)
- Cohen & Frazzini (2008), *Economic Links and Predictable Returns*: 고객·공급 관계와 수익률
- Hoberg & Phillips (2016), *Text-Based Network Industries and Endogenous Product Differentiation*: 10-K 텍스트로 만든 산업 네트워크
- Foster (1981), Lang & Stulz (1992): 정보 전이와 경쟁 효과 (이벤트 스터디)

## 17. 검토 기록

| 날짜 | 검토자 | 요지 | 반영 |
|---|---|---|---|
| 2026-09-24 | 저장소 주인 | 관계 목록을 미리 쓰는 정답셋은 부담이 너무 큼. AI에게 정답을 만들게 하고 싶지도 않음 | v2.1: 표본 회사 언급 구간의 가림 검수로 대체(7.6, 10장) |
| 2026-09-24 | Codex (GPT), [codex_review.md](codex_review.md) | 전체 방향(후보 합집합, `supplies` 통합, 다중 관계, 배치 판정, 어댑터·캐시)은 타당. 판정 단위, 방향·시점, 평가 분리, 재판정 대상, 매출 비중 귀속, 상태·검수 저장, 근거 위치 기준을 보완하라는 의견 | v2에 반영. 아래 참고 |

**v2에서 바뀐 것**
- **판정 단위(7.3·7.4):** 쌍 단위 `same_entity` 하나를 언급별 식별 → 구간별 관계 근거·유형·방향·시점 → 코드의 쌍 단위 종합으로 바꿨습니다. 관계에는 채택된 구간만 근거로 연결합니다. 질문은 실제 회사명을 넣는 대칭 템플릿으로 바꿨습니다.
- **방향·시점(5장·7.3):** 지분을 방향별 질문 둘로 나눴고, "relies on"을 뺐습니다. 시점(`status`)을 추가했고, 공급 추론은 숨기며, `similar`의 뜻을 바로잡았습니다. 공급·협력에는 하위 속성을 추가했습니다.
- **평가(7.6):** 개발셋·평가셋을 회사 단위로 분리했습니다. 정답을 '알려진 / 확인 가능' 범위로 나누고, 평가를 후보·근거·판정·전체의 네 단계로 분해했습니다. 층화 가중, 회사 단위 부트스트랩, 합격 기준의 신뢰구간·표본 수·자동 채택 비율도 추가했습니다.
- **재판정(7.4):** 채택 후보만이 아니라 임계값 근처, 맥락 단어가 있는데 점수가 낮은 경우, 판정 충돌, 기각 쪽 감사 표본까지 넓혔습니다. 공통 출력 형식(`decision`/`score`/`abstain`/`evidence_ids`)을 두고, LLM의 '예'를 확률 1로 바꾸지 않습니다.
- **매출 비중(7.1):** 연산자·기간·분모·대상 구조로 바꿨고, 불분명하면 비웁니다. 익명 고객은 공시 단위로만 식별합니다.
- **저장(7.5):** 후보·판정 기록·근거 구간·공시 테이블을 추가했습니다. 검수 기록에 `as_of`, 근거 해시, 스키마 버전을 넣고, 검수 적용 규칙을 명시했습니다.
- **근거 위치(7.1):** 정제 텍스트 기준임을 명시하고 `text_hash`를 저장합니다.
- **웹앱(8장):** API를 `node_id` 중심으로 바꿨고, 선 굵기를 고정했습니다. 여러 역할 회사의 배치 규칙과 '판정 전 후보 보기' 화면을 추가했습니다.
- **순서(10장):** 스키마 확정과 0단계 화면을 병행하고, 정답셋 작성을 함께 시작하게 했습니다. 판정과 평가는 1단계에 함께 구현합니다.

**수정해서 받아들인 부분**
- **데이터 세트 수:** 세 세트(질문 개발·임계값·최종 평가) 대신, 라벨링 부담을 고려해 두 세트로 시작합니다.
- **근거 위치:** 원문(`text_raw`)이 `data/sections`에 이미 있으므로, 원문을 새로 보관하지는 않습니다. 원문 위치 매핑은 필요할 때 추가합니다.
- **구현 순서:** 정답셋이 완성될 때까지 화면을 미루지 않습니다.
