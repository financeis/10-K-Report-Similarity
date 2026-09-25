# sp500_2025 관계도 연도 비교 (2024 → 2025년 제출 10-K)

- 생성: 2026-09-26T01:54:38
- 비교: `sp500_2024`(2024년 제출 10-K)와 `sp500_2025`(2025년 제출 10-K)의 관계도입니다. 제출 연도 기준이라, 대부분은 한 해 전 회계연도의 10-K입니다.
- 판정 설정: `jev-1.13.0`, 질문 v3.1, 입력 nearby, 채택 0.8 / 기각 0.3 (질문별: is_entity reject 0.5, competitor accept 0.7/reject 0.5, business reject 0.5). 두 해가 같습니다.
- 관계도에 보이는 관계(모델 채택 + 검수로 확인)끼리 비교합니다. 회사는 티커로 맞춥니다(CIK가 바뀐 회사 포함).
- 분석한 회사: 2024년 488곳, 2025년 494곳, 두 해 모두 487곳. 2024년에만: HAL. 2025년에만: GEV, GLW, RDDT, SMCI, SNDK, SOLV, SW.
- 익명 고객('Customer A' 등)과의 관계는 해마다 같은 고객인지 알 수 없어 비교에서 뺐습니다(2024년 2개, 2025년 6개).
- **'보이지 않음'은 관계가 끝났다는 뜻이 아닙니다.** 그해 10-K에 적히지 않았거나, 판정이 달라졌거나, 그해 10-K를 분석하지 못한 경우입니다. 이유를 함께 셉니다.

## 1. 유형별 변화

| 관계 | 2024년 | 2025년 | 유지 | 새로 보임 | 보이지 않음 | 2024년 관계 중 유지 | 유지된 관계 중 같은 문장 |
|---|---|---|---|---|---|---|---|
| 경쟁 | 564 | 546 | 490 | 56 | 74 | 87% | 53% |
| 공급·협력 | 512 | 509 | 436 | 73 | 76 | 85% | 49% |
| 지분 (검증 전) | 19 | 21 | 15 | 6 | 4 | 79% | 93% |

- 2024년 수는 검수로 확인한 관계를 포함합니다.
- '같은 문장'은 두 해 근거 문장 가운데 글자까지 같은 문장이 하나라도 있는 비율입니다. 10-K는 해마다 같은 문단을 되풀이하는 경우가 많습니다.

## 2. 새로 보이거나 보이지 않게 된 이유

새로 보인 관계는 2024년에, 보이지 않게 된 관계는 2025년에 왜 없었는지입니다.

| 관계 | 변화 | 합계 | 그해 10-K 섹션을 분석하지 못함 | 그해 10-K에 상대 이름 없음 | 검수에서 거절 | 불확실(검수 대기)로 남음 | 언급은 있으나 관계로 채택 안 됨 |
|---|---|---|---|---|---|---|---|
| 경쟁 | 새로 보임 | 56 | 22 | 22 | 0 | 4 | 8 |
| 경쟁 | 보이지 않음 | 74 | 4 | 55 | 0 | 4 | 11 |
| 공급·협력 | 새로 보임 | 73 | 13 | 36 | 0 | 15 | 9 |
| 공급·협력 | 보이지 않음 | 76 | 3 | 57 | 0 | 7 | 9 |
| 지분 (검증 전) | 새로 보임 | 6 | 3 | 1 | 0 | 2 | 0 |
| 지분 (검증 전) | 보이지 않음 | 4 | 0 | 0 | 0 | 4 | 0 |

- **'그해 10-K에 상대 이름 없음'이 공시 문구가 바뀐 경우입니다.** 그해 두 회사 10-K의 Item 1·1A에 상대 회사를 가리키는 이름이 없습니다(제외 문맥의 언급과, 판정 모델이 그 회사가 아니라고 본 언급은 세지 않음).
  - 이렇게 보이지 않게 된 관계를 많이 적었던 회사(2024년 근거 기준): Broadcom (AVGO) 14개, Accenture (ACN) 12개, Microsoft (MSFT) 8개, Coherent Corp. (COHR) 5개, Equifax (EFX) 4개.
  - 새로 이름을 적은 회사(2025년 근거 기준): Axon Enterprise (AXON) 4개, CDW Corporation (CDW) 2개, Ciena (CIEN) 2개, Entergy (ETR) 2개, Hewlett Packard Enterprise (HPE) 2개.
- '분석하지 못함'에는 그해 10-K가 없거나 추출 품질 검사를 통과하지 못한 경우와, 근거가 있던 섹션이 요약만 추출되어 다른 해의 4분의 1 아래로 줄어든 경우가 들어갑니다.
- '불확실'·'채택 안 됨'·'검수에서 거절'은 두 해 모두 이름은 나왔는데 한 해만 채택된 관계입니다. 두 해 문장이 글자까지 같은데도 판정이 달라진 것: 경쟁 27개 중 3개 · 공급·협력 40개 중 7개 · 지분 (검증 전) 6개 중 0개. 판정 입력의 앞뒤 문단이 달라지면 경계값 근처의 점수가 바뀝니다.

## 3. 텍스트 유사도의 연도 안정성

관계도 후보에 쓰는 `tfidf-openai` 유사도(Item 1)로, 두 해 모두 유사도가 있는 472개사의 상위 20 이웃을 비교했습니다(이웃 순위는 이 회사들 안에서 다시 잼).

- 상위 20 이웃 가운데 이듬해에도 상위 20인 비율: 평균 88% (사분위 85%, 90%, 95%).
- 가장 비슷한 회사(1위)가 같은 비율: 79%.
- 모든 기업쌍 유사도의 순위 상관(스피어만): 0.954.

## 4. 예시

분석하지 못한 10-K 때문에 생긴 변화는 뺐습니다. 근거 문장은 상대 회사 이름을 가운데 두고 앞뒤 150자씩 보여줍니다.

### 2025년에 새로 보인 공급·협력

| 회사 | 2024년에 없던 이유 | 시점 | 2025년 근거 문장 |
|---|---|---|---|
| Baker Hughes (BKR) – GE Aerospace (GE) | 불확실(검수 대기)로 남음 | 현재 | The partial or complete loss of GE Vernova or GE Aerospace as suppliers, as well as contracts with our aeroderivative joint venture (the "Aero JV") with GE Vernova may adversely affect our busines… |
| Boeing (BA) – Teledyne Technologies (TDY) | 불확실(검수 대기)로 남음 | 현재 | …oeing in 2024 lasted almost two months and resulted in a pause in aircraft production. These factors have negatively impacted our sales to Airbus and Boeing and any future pauses or reductions in manufacturing could negatively impact our business. |
| Clorox (CLX) – Procter & Gamble (PG) | 그해 10-K에 상대 이름 없음 | 현재 | In February 2025, the Company announced that the Venture Agreement with The Procter & Gamble Company (P&G) for the Company's Glad bags and wraps business will wind down by January 31, 2026. |
| Coherent Corp. (COHR) – Huawei | 그해 10-K에 상대 이름 없음 | 현재 | The Bureau of Industry and Security of the U.S. Department of Commerce (“BIS”) has issued final rules under the EAR that restrict access by Huawei Technologies Co. Ltd. and certain of its affiliates (collectively, “Huawei”) to items produced domestically and abroad from certain U.S. techno… |
| CoStar Group (CSGP) – Alphabet Inc. (Class A) (GOOGL) | 불확실(검수 대기)로 남음 | 현재 | For example, starting on July 1, 2024, Universal Analytics (UA), Google’s legacy analytics platform on which we historically relied for calculating monthly average unique visitors, was discontinued by Google. |
| Apple Inc. (AAPL) – EchoStar (ECHO) | 언급은 있으나 관계로 채택 안 됨 | 현재 | At the end of the third quarter of 2023, we began offering premium wireless devices, including Apple products. |
| Broadcom (AVGO) – CDW Corporation (CDW) | 그해 10-K에 상대 이름 없음 | 현재 | …,000 products and services from more than 1,000 vendor partners, including well-established companies such as Adobe, APC, Apple, Amazon Web Services, Broadcom Inc., Cisco, Dell Technologies, Google, Hewlett Packard Enterprise, HP Inc., IBM, Intel, Lenovo, Microsoft, NetApp, Nutanix, Palo Alto Networ… |
| Boeing (BA) – Howmet Aerospace (HWM) | 그해 10-K에 상대 이름 없음 | 현재 | Boeing production rates have had and are expected to have a material impact on the financial performance of Howmet. |

### 2025년에 새로 보인 경쟁

| 회사 | 2024년에 없던 이유 | 시점 | 2025년 근거 문장 |
|---|---|---|---|
| Ciena (CIEN) – Hewlett Packard Enterprise (HPE) | 그해 10-K에 상대 이름 없음 | 현재 | Our competitors include Nokia, Huawei, Cisco, Hewlett Packard Enterprise, and ZTE. |
| Fox Corporation (Class A) (FOXA) – Alphabet Inc. (Class A) (GOOGL) | 그해 10-K에 상대 이름 없음 | 현재 | …FOX Business also face competition online from CNN.com, NBCNews.com, NYTimes.com, CNBC.com, Bloomberg.com, Yahoo.com, The Wall Street Journal Online, YouTube, social media platforms and audio and podcast networks, among others. |
| Fortinet (FTNT) – Microsoft (MSFT) | 언급은 있으나 관계로 채택 안 됨 | 현재 | … F5 Networks, Inc. (“F5 Networks”), Hewlett-Packard Enterprise (“HPE”), Huawei Technologies Co., Ltd. (“Huawei”), Juniper Networks, Inc. (“Juniper”), Microsoft Corporation (“Microsoft”), Netskope Inc. (“Netskope”), Palo Alto Networks, Inc. (“Palo Alto Networks”), SonicWALL, Inc. (“SonicWALL”), Sopho… |
| Amcor (AMCR) – 3M (MMM) | 그해 10-K에 상대 이름 없음 | 현재 | Competitors include 3M, AptarGroup, Inc., Ball Corporation, Inc, CCL Industries Inc., Crown Holdings, Inc., Graphic Packaging Holding Company, Huhtamaki Oyj, Internationa… |
| Advanced Micro Devices (AMD) – Qualcomm (QCOM) | 그해 10-K에 상대 이름 없음 | 현재 | …rs such as Broadcom Corporation, Marvell Technology Group, Ltd., Analog Devices, Texas Instruments Incorporated and NXP Semiconductors N.V., and from Qualcomm Incorporated and NVIDIA. |
| Amgen (AMGN) – Regeneron Pharmaceuticals (REGN) | 언급은 있으나 관계로 채택 안 됨 | 현재 | In addition, biosimilar versions of EYLEA have been approved both in and outside the United States. These include Amgen's PavbluTM (aflibercept-ayyh), which recently launched in the United States. |
| Amazon (AMZN) – EchoStar (ECHO) | 언급은 있으나 관계로 채택 안 됨 | 현재 | SpaceX, Amazon’s Project Kuiper (“Kuiper”) and others have obtained FCC authority to launch and operate, or provide service from, NGSO satellite systems using… |
| Arista Networks (ANET) – Huawei | 그해 10-K에 상대 이름 없음 | 현재 | …with competition also coming from other large network equipment and system vendors, including Dell/EMC, Extreme Networks, Hewlett Packard Enterprise, Huawei, Juniper Networks, Nvidia and white box networking vendors utilizing open-source operating systems. |

### 2025년에 보이지 않게 된 공급·협력

| 회사 | 2025년에 없던 이유 | 시점 | 2024년 근거 문장 |
|---|---|---|---|
| American International Group (AIG) – BlackRock (BLK) | 그해 10-K에 상대 이름 없음 | 현재 | In addition, beginning in April 2022, certain AIG and Corebridge insurance company subsidiaries entered into investment management agreements with BlackRock and as of December 31, 2023, BlackRock manages $135 billion of our investment portfolio, consisting of liquid fixed income and certain priva… |
| AppLovin (APP) – Meta Platforms (META) | 불확실(검수 대기)로 남음 | 현재 | Several of these platforms, including Facebook, Google, Amazon, and Unity Software, are also our partners and clients. |
| Expand Energy (EXE) – Valero Energy (VLO) | 그해 10-K에 상대 이름 없음 | 과거 | For the 2023 Successor Period, sales to Valero Energy Corporation and Shell Energy North America accounted for approximately 17% and 10%, respectively, of total revenues (before the effects … |
| Teradyne (TER) – Taiwan Semiconductor Manufacturing | 그해 10-K에 상대 이름 없음 | 현재 | In 2021, revenues from Taiwan Semiconductor Manufacturing Company Ltd., a customer of our Semiconductor Test segment, accounted for 12% of our consolidated revenues. |
| Teradyne (TER) – Huawei | 언급은 있으나 관계로 채택 안 됨 | 현재 | However, we do not expect these actions will mitigate the impact of the regulations on our sales to Huawei, HiSilicon and other suppliers. |
| Broadcom (AVGO) – Dell Technologies (DELL) | 불확실(검수 대기)로 남음 | 현재 | Subsequent to the acquisition, Broadcom announced changes to its go-to-market approach for VMware offerings, resulting in a change in our commercial relationship with VMware. |
| Apple Inc. (AAPL) – Booking Holdings (BKNG) | 언급은 있으나 관계로 채택 안 됨 | 현재 | As the primary smartphone manufacturers, Google and Apple could leverage their operating systems to give a competitive advantage to their services that overlap with ours. |
| Amazon (AMZN) – AppLovin (APP) | 불확실(검수 대기)로 남음 | 현재 | Several of these platforms, including Facebook, Google, Amazon, and Unity Software, are also our partners and clients. |

### 2025년에 보이지 않게 된 경쟁

| 회사 | 2025년에 없던 이유 | 시점 | 2024년 근거 문장 |
|---|---|---|---|
| Alphabet Inc. (Class A) (GOOGL) – Microsoft (MSFT) | 그해 10-K에 상대 이름 없음 | 현재 | Our AI offerings compete with AI products from hyperscalers such as Amazon and Google, as well as products from other emerging competitors, including Anthropic, OpenAI, Meta, and other open source offerings, many of which are als… |
| Apple Inc. (AAPL) – Booking Holdings (BKNG) | 언급은 있으나 관계로 채택 안 됨 | 현재 | See - "Consumer adoption and use of mobile devices creates challenges and may enable device companies such as Google and Apple to compete directly with us." |
| Apple Inc. (AAPL) – Microsoft (MSFT) | 그해 10-K에 상대 이름 없음 | 현재 | Windows faces competition from various software products and from alternative platforms and devices, mainly from Apple and Google, and Microsoft Defender for Endpoint competes with CrowdStrike on endpoint security solutions. |
| Broadcom (AVGO) – Microsoft (MSFT) | 그해 10-K에 상대 이름 없음 | 현재 | … CyberArk Software, Ltd., Dino-Software Corporation, Fortinet, Inc., Hewlett Packard Enterprise Company, International Business Machines Corporation, Microsoft Corporation, New Relic, Inc., OpenText Corporation, Oracle Corporation, Palo Alto Networks, Inc., Proofpoint, Inc., Rocket Software, Inc., S… |
| Amazon (AMZN) – Microsoft (MSFT) | 그해 10-K에 상대 이름 없음 | 현재 | Azure faces diverse competition from companies such as Amazon, Broadcom, Google, IBM, Oracle, and open source offerings. |
| Abbott Laboratories (ABT) – Boston Scientific (BSX) | 그해 10-K에 상대 이름 없음 | 현재 | Our primary competitors include Abbott Laboratories and Medtronic plc, as well as a wide range of medical device companies that sell a single or limited number of competitive product… |
| Broadcom (AVGO) – Coherent Corp. (COHR) | 그해 10-K에 상대 이름 없음 | 현재 | …s and the internal resources of large integrated OEMs, such as Advanced Micro Devices, Inc., Amlogic Inc., Analog Devices, Inc., Cisco Systems, Inc., Coherent Corp., Hamamatsu Photonics K.K., Heidenhain Corporation, iC-Haus GmbH, Intel Corporation, Lumentum Holdings Inc., MACOM Technology Solutions … |
| Boston Scientific (BSX) – Medtronic (MDT) | 그해 10-K에 상대 이름 없음 | 현재 | Our primary competitors include Abbott Laboratories and Medtronic plc, as well as a wide range of medical device companies that sell a single or limited number of competitive products or participate in only… |

## 5. 해석할 때 주의할 점

- 관계도는 10-K Item 1·1A에 이름이 적힌 관계만 담습니다. 회사가 거래처·경쟁사 이름을 적는 방식은 해마다 바뀌므로, 공시 변화가 곧 거래 변화는 아닙니다.
- 두 해 모두 같은 판정 모델과 임계값을 씁니다. 판정 정확도는 2024년 확인 표본으로만 검증했습니다.
- 회사 목록은 두 해 모두 현재 S&P 500입니다(생존 편향).
