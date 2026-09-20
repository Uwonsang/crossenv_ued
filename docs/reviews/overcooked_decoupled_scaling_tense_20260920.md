# 시제 일관성 검토

대상: `overcooked_decoupled_scaling.pdf` (저장소 루트, 24쪽). 작성일: 2026-09-20.

논문 전체(본문 p.1–11, 부록 p.15–24)를 문장 단위로 읽고 시제만 따로 본 기록이다. 줄 번호는 PDF 여백에 인쇄된 번호다. 논문은 수정하지 않았다. 내용·수치·논리 문제는 별도 파일 [overcooked_decoupled_scaling_review_20260920.md](overcooked_decoupled_scaling_review_20260920.md)에 있다.

> **적용 순서 주의.** 인쇄된 줄 번호는 본문을 고치면 밀린다. 아래 수정은 **뒤에서 앞으로**(L1153 → L041 순서) 적용하거나, 줄 번호 대신 인용된 영어 원문을 검색해서 찾는 편이 안전하다.

## 0. 명백하게 틀린 것 (5곳)

아래 다섯은 **논문 안에서 같은 사실을 두 시제로 쓴 내부 모순**이다. 두 문장을 나란히 놓으면 둘 다 맞을 수 없으므로 취향 문제가 아니다. 나머지 항목(선행연구 단순과거, Appendix E 과거형, Figure 9 캡션, L072, L974 등)은 문법적으로 방어 가능한 일관성 문제이므로 여기서 제외했다.

| # | 고칠 곳 | 충돌 상대 | 수정 |
|---|---|---|---|
| 1 | **L491** "DCEC **improved** coordination with unseen partners and generalization to unseen environments" | **L025 (Abstract)** "DCEC **improves** coordination with unseen partners and generalization to unseen environments" — 거의 같은 문장 | `improved` → `improves` |
| 2 | **L323** "We **evaluated** DCEC in two cooperative environments" | **L023 (Abstract)** "We **evaluate** DCEC on two cooperative benchmarks" — 같은 문장 | `evaluated` → `evaluate` |
| 3 | **L377** "IPPO and E3T, which **were trained** directly on the corresponding fixed layouts" | **L337** "Each single-task baseline **is trained** separately on each fixed evaluation task" + **L348** "the single-task baselines **are trained** separately on each corresponding fixed task" — 같은 사실 3번 서술, 2번이 현재형 | `were trained` → `are trained` |
| 4 | **L477** "Each participant **interacted** with all six agent variants" | **L989** "Participants **interact** with agents from each evaluated method" — 같은 사실, 정반대 시제 | 둘 중 하나로 통일 (인간 실험이므로 L989를 `interacted`로 맞추는 쪽 권장) |
| 5 | **L084–086** "When we **applied** IDAAC to CEC, we **observed** consistent performance scaling as we **increased** the batch size from 8,192 to 65,536, while vanilla CEC **showed** no improvement" | **L194** "we **increase** the minibatch size from 8,192 to 65,536" — 같은 실험·같은 수치 / **L082 바로 앞 문장** "we **discover** that IDAAC **unlocks** large-batch training" | 4개 동사 전부 현재형으로 |

1·2·5는 논문이 직접 자기 문장을 반복하면서 시제만 바꾼 경우라 반박 여지가 없다. 특히 1은 Abstract와 Conclusion을 나란히 읽는 리뷰어가 바로 본다.

---

## 1. 결론부터

**이 논문의 기본 문체는 이미 현재형이다.** 과거형이 전체적으로 섞여 있는 게 아니라, 현재형 문서에 과거형이 **섬처럼 18군데** 박혀 있는 구조다. 근거:

| 구역 | 시제 |
|---|---|
| Abstract (L011–027) | 전부 현재형 |
| 3절 Preliminaries (L139–170) | 전부 현재형 |
| 4절 Scaling (L191–255) — **실험 결과 포함** | 전부 현재형 |
| 5절 DCEC 방법 (L262–317) | 전부 현재형 |
| 6.1 실험 설정 (L324–357) | L323 한 곳 빼고 전부 현재형 |
| 6.2–6.5 결과 (L376–450) | L377 한 곳 빼고 전부 현재형 |
| Appendix A (L758–804) | 전부 현재형 |
| Appendix B (L827–869) | 전부 현재형 |
| Appendix C.1–C.2 (L876–945) | 전부 현재형 |
| Appendix C.3 (L974–990) | L974 한 곳 빼고 전부 현재형 |
| **Appendix E 인간 실험 (L1136–1173)** | **거의 전부 과거형** |

즉 4절은 배치 스케일링 실험 결과를 전부 현재형으로 쓰고 있다("we **increase** the minibatch size", "CEC **shows** lower training value loss"). 그러니 "실험한 건 과거형"이라는 규칙을 엄격히 적용하면 4절과 6.2–6.5절을 통째로 과거형으로 바꿔야 하고, 그러면 Abstract와 충돌한다.

**권장: 현재형을 기본값으로 두고, 과거형은 인간 실험(Appendix E)에만 남긴다.** 고칠 곳은 18군데이고 대부분 한두 단어다.

## 2. 적용할 규칙

1. **기본값은 현재형.** 방법 설명, 환경·구조 정의, 평가 프로토콜, 실험 결과, 그림 설명, 논문의 기여 서술 전부 현재형.
2. **과거형은 인간 실험 절차에만.** 참가자 모집, IRB 승인, 동의, 참가자가 실제로 한 행동, 설문 실시. 한 번 일어난 사건이고 HCI 논문의 관례다.
3. **설문 문항 원문(Q1–Q7)은 과거형 그대로 둔다.** 참가자에게 제시한 문구 자체이므로 바꾸면 안 된다.
4. **선행연구는 현재완료로 통일.** 이미 다수가 현재완료(`has proposed`, `has been shown`)이므로 단순과거 6곳을 맞춘다.
5. **그림·표를 가리킬 때는 현재형.** `Figure 2 shows`, `Figure 5(a) presents`처럼 이미 일관되어 있다.

## 3. 같은 내용을 두 시제로 쓴 곳 (최우선)

이 네 쌍은 논리 문제가 아니라 같은 문장이 두 번 다른 시제로 나오는 경우다. 리뷰어가 가장 먼저 알아챈다.

### 3.1 Abstract L025 vs Conclusion L491 — 거의 동일한 문장

- **Abstract L025–027 (현재형):** "Across both benchmarks, DCEC **improves** coordination with unseen partners and generalization to unseen environments, while also **achieving** strong coordination performance with human–AI."
- **Conclusion L491–493 (과거형):** "Across Dual Destination and Overcooked-AI, DCEC **improved** coordination with unseen partners and generalization to unseen environments, while also achieving strong human–AI coordination performance."

게다가 Conclusion 같은 문단 안에서도 앞 문장 L489는 `DCEC **enables**`(현재), 뒤 문장 L493은 `These results **highlight**`(현재)다. 과거형은 가운데 한 문장뿐이다.

→ **`improved` → `improves`**

### 3.2 Abstract L023 vs 6.1절 L323 — 거의 동일한 문장

- **Abstract L023 (현재형):** "We **evaluate** DCEC on two cooperative benchmarks, Dual Destination and Overcooked-AI."
- **6.1절 L323 (과거형):** "**Environment.** We **evaluated** DCEC in two cooperative environments: Dual Destination and Overcooked-AI."

L323은 6.1절 전체(L324–357)에서 유일한 과거형이다. 바로 뒤 L335 `We **compare**`, L344 `We **evaluate**`, L352 `we **evaluate**`, L353 `we additionally **evaluate**` 전부 현재형이다.

→ **`We evaluated` → `We evaluate`**

### 3.3 본문 L084 vs 4절 L194 — 동일한 실험을 두 시제로

- **1절 L084–086 (과거형 4개):** "**When we applied** IDAAC to CEC, we **observed** consistent performance scaling **as we increased** the batch size from 8,192 (default in CEC) to 65,536, while vanilla CEC **showed** no improvement."
- **4절 L194–195 (현재형):** "To examine the effect of larger-batch training on generalization, we **increase** the minibatch size from 8,192 to 65,536."
- **6.5절 L440–445 (현재형):** "For CEC, performance **does not improve** with larger batches... In contrast, DCEC **outperforms** CEC at 8K and **continues** to improve..."

L084의 바로 앞 문장 L082도 현재형이다: "we **discover** that IDAAC **unlocks** large-batch training in CEC."

→ 수정안: "**When IDAAC is applied to CEC, we observe** consistent performance scaling **as we increase** the batch size from 8,192 (default in CEC) to 65,536, while vanilla CEC **shows** no improvement."

### 3.4 본문 L477 vs 부록 L989 — 인간 실험 절차를 두 시제로

- **6.6절 L477–479 (과거형):** "Each participant **interacted** with all six agent variants across five Overcooked layouts, with the agent-layout combinations presented in a randomized order."
- **Appendix C.3 L989 (현재형):** "Participants **interact** with agents from each evaluated method on the fixed Overcooked tasks."

둘 중 하나는 반드시 바꿔야 한다. 규칙 2를 따르면 인간 실험은 과거형이므로 **L989를 과거형으로** 맞추는 쪽을 권한다: `Participants **interacted** with agents...`. 단, 6.6절 본문 전체를 현재형으로 유지하고 싶다면 반대 방향(L477→현재형)도 가능하다. 아래 4절 표에 두 선택지를 모두 적었다.

## 4. 본문에서 고칠 곳 (현재형으로)

| # | 위치 | 현재 원문 | 수정안 | 근거 |
|---|---|---|---|---|
| 1 | p.2 L072 | "We **conducted** our experiments on top of the CEC framework" | "We **conduct** our experiments..." | 같은 동사가 L430에서 현재형 — "We **conduct** this analysis on both the five fixed tasks..." |
| 2 | p.2 L084–086 | "When we **applied**... we **observed**... as we **increased**... CEC **showed**" | "When IDAAC **is applied**... we **observe**... as we **increase**... CEC **shows**" | 3.3 참조. 앞 문장 L082가 현재형 |
| 3 | p.6 L323 | "We **evaluated** DCEC in two cooperative environments" | "We **evaluate** DCEC..." | 3.2 참조. 6.1절 유일한 과거형 |
| 4 | p.7 L377 | "CEC performs worse than IPPO and E3T, which **were trained** directly on the corresponding fixed layouts" | "which **are trained** directly on..." | 같은 사실이 L337 "Each single-task baseline **is trained** separately", L348 "the single-task baselines **are trained** separately"로 두 번 현재형 |
| 5 | p.9 L476–477 | "we **conducted** a user study involving 20 participants" | (A) "we **conduct** a user study..." 또는 (B) 과거형 유지 | 앞뒤 L475·L480은 현재형. 규칙 2를 택하면 (B) 유지 가능 |
| 6 | p.9 L477–479 | "Each participant **interacted** with all six agent variants..." | (A) "Each participant **interacts** with..." 또는 (B) 과거형 유지 + **L989를 과거형으로** | 3.4 참조. #5와 같은 선택지를 골라야 함 |
| 7 | p.10 L488–489 | "In this work, we **addressed** a limitation of large-batch cross-environment training" | "we **address** a limitation..." | 연구 내용 서술은 전부 현재형 — L070 `we investigate`, L079 `we adopt`, L114 `we focus`, L131 `we study`, L132 `We further examine` |
| 8 | p.10 L491–492 | "DCEC **improved** coordination with unseen partners and generalization to unseen environments" | "DCEC **improves** coordination..." | 3.1 참조. Abstract L025와 같은 문장 |
| 9 | p.19 L974 | "pairing each ego policy with partner policies **that were not encountered** during its training" | "that **are not encountered** during its training" | 같은 부록 L984 "whose behavior **differs** from the self-play partners **encountered** during training", 본문 L346 "**are** also unseen during training", L348 "**are held out** from training" |
| 10 | p.9 L471–472 (Figure 9 캡션) | "Scores **were averaged** across five layouts per participant" | "Scores **are averaged** across five layouts per participant" | 다른 캡션은 현재형 — Figure 8 L449 "Batch scaling **benefits**", Figure 12 L1015 "Environment scaling **benefits**", Table 1 L897 "All layers **use** orthogonal initialization" |

## 5. 선행연구 시제 통일 (중간 우선순위)

현재완료 13곳, 단순과거 6곳이 섞여 있다. 문법 오류는 아니지만 같은 구문이 네 줄 간격으로 다르게 쓰인 곳이 있다.

**직접 충돌하는 쌍:**

- L126 "Prior work (Raileanu & Fergus, 2021) **has proposed** decoupling the actor and critic networks..."
- L130 "They **proposed** that controlling the value function's update frequency and learning schedule **improves** generalization performance."

- L067 "prior work... on ZSC **has primarily focused** on data diversity"
- L097 "Early ZSC research (Strouse et al., 2021) mainly **focused** on increasing training-partner diversity"

**단순과거 6곳 (현재완료 또는 현재형으로 통일 권장):**

| 위치 | 현재 원문 | 수정안 |
|---|---|---|
| p.1 L041 | "Early approaches in ZSC **improved** partner diversity through population-based training methods" | "**have improved** partner diversity" |
| p.1 L043–044 | "The research focus later **expanded** to environment diversity" | "**has since expanded** to environment diversity" |
| p.1 L045 | "This **motivated** approaches that diversify the environments" | "This **motivates** approaches that diversify..." |
| p.1 L048 | "a recent work **showed** that environment diversity alone **is** sufficient" | "a recent work **has shown** that..." (뒤의 `is`는 그대로) |
| p.2 L097 | "Early ZSC research mainly **focused** on increasing training-partner diversity" | "**has mainly focused** on..." |
| p.3 L130 | "They **proposed** that controlling the value function's update frequency... **improves** generalization" | "They **propose** that..." (뒤의 `improves`는 그대로) |

참고로 L048과 L130은 한 문장 안에 과거형과 현재형이 같이 있다("**showed** that ... **is** sufficient"). 종속절의 현재형은 일반적 사실이므로 맞다. 주절만 고치면 된다.

## 6. Appendix E (인간 실험) 처리 방침

L1136–1173은 과거형 동사가 약 20개로 논문에서 유일하게 과거형이 지배적인 구역이다. **이건 그대로 두는 것을 권한다.** 실제로 한 번 수행한 실험이고, HCI 논문에서 표준 관례다. 다만 세 가지만 손본다.

### 6.1 과거형을 유지할 곳 (수정 불필요)

L1136 `We adopted a within-subjects design` / `every participant played` · L1139 `Participants completed one episode` · L1140 `we randomly assigned` · L1142 `We also randomized the agent order` · L1144 `Participants completed a brief tutorial` / `the entire study lasted` · L1147 `The study was conducted through a web-based interface` · L1151 `We recruited 20 participants` / `Our institutional ethics review board approved` / `all participants provided informed consent` · L1153 `We provided compensation` · L1155 `we withheld information about the identities` · L1156 `Participants rated the agent after each interaction`

### 6.2 설문 문항은 절대 바꾸지 말 것

L1158 `The questions were:`와 L1159–1167의 Q1–Q7 ("The agent **adapted** to me...", "The agent **was** consistent...", "The agent's actions **were** human-like." 등)은 참가자에게 실제로 제시한 문구다. 현재형으로 고치면 설문 도구 자체를 잘못 보고하는 게 된다.

### 6.3 과거형 구역 안에서도 현재형이어야 하는 곳 (2군데)

시행 절차가 아니라 **환경·인터페이스의 속성**을 설명하는 문장이다.

| 위치 | 현재 원문 | 수정안 | 근거 |
|---|---|---|---|
| p.22 L1143–1144 | "Each episode **comprised** 200 environment steps and **awarded** 20 points per successful delivery" | "Each episode **comprises** 200 environment steps and **awards** 20 points..." | 같은 보상 설정이 Appendix B L855에서 현재형 — "The sparse task reward **is** +20 for each successfully delivered soup" |
| p.22 L1153–1155 | "Participants **controlled** their character using the keyboard, with the arrow keys for directional movement..." | "Participants **control** their character using the keyboard..." | 조작 방식은 인터페이스의 속성. 바로 앞 L1148이 이미 현재형 — "NiceWebRL **supports** server-side parallelization" |

L1153은 판단의 여지가 있다. "참가자가 그렇게 조작했다"는 시행 기술로 보면 과거형도 되므로, 통일만 되면 둘 다 허용 가능하다. 다만 L1143은 환경 스펙이므로 현재형을 권한다.

### 6.4 이미 맞는 현재형 (건드리지 말 것)

- L1148 "NiceWebRL **supports** server-side parallelization" — 도구의 일반적 성질
- L1141 "For layouts in which the starting positions **correspond** to different roles" — 레이아웃의 속성 (같은 문장의 `experienced`는 과거형 유지가 맞다)
- L1173 "Bar plots for each survey question **show** the Likert response distributions" — 그림 지시는 현재형

## 7. 시제 문제가 아닌 것

- **p.11 L540–577 (AI use / Ethics / Reproducibility):** L544 "we **used** generative AI tools"의 과거형은 disclosure 관례상 맞다. 문제는 시제가 아니라 **템플릿 지시문과 placeholder가 그대로 남아 있다는 것**이다. 별도 리뷰 파일의 해당 항목 참고.
- **L1169–1172 (Cronbach α, reverse-coding, t-test):** "We **evaluated** the internal consistency", "We **reverse-coded**", "We **compared** the aggregated preference scores" — 인간 실험 데이터 분석 절차이므로 규칙 2에 따라 과거형 유지가 맞다.
- **본문 전반의 현재완료** (`has improved`, `has been adopted`, `have been proposed`, `has demonstrated` 등 13곳) — 선행연구 서술로 적절하다.

## 8. 체크리스트

뒤에서 앞으로 적용.

- [ ] L1153 `controlled` → `control` (선택)
- [ ] L1143 `comprised`/`awarded` → `comprises`/`awards`
- [ ] L989 `interact` → `interacted` (L477을 과거형으로 유지하는 경우)
- [ ] L974 `were not encountered` → `are not encountered`
- [ ] L491 `improved` → `improves`
- [ ] L488 `addressed` → `address`
- [ ] L477–479 / L476–477 — 6.6절 인간 실험 시제 선택 확정 후 적용
- [ ] L471 `were averaged` → `are averaged`
- [ ] L377 `were trained` → `are trained`
- [ ] L323 `evaluated` → `evaluate`
- [ ] L130 `proposed` → `propose`
- [ ] L097 `focused` → `has mainly focused`
- [ ] L084–086 `applied`/`observed`/`increased`/`showed` → 현재형
- [ ] L072 `conducted` → `conduct`
- [ ] L048 `showed` → `has shown`
- [ ] L045 `motivated` → `motivates`
- [ ] L043 `expanded` → `has since expanded`
- [ ] L041 `improved` → `have improved`
- [ ] 수정 후 Abstract와 Conclusion을 나란히 놓고 대응 문장 시제 재확인
