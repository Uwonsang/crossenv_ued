# 논문 리뷰 중간 기록

대상: `overcooked_decoupled_scaling.pdf` (저장소 루트, 24쪽). 작성일: 2026-09-20.

사용자의 요청으로 검토 도중 저장한 기록이다. 아래 줄 번호는 PDF에 인쇄된 번호이며, 추출 텍스트의 줄 번호가 아니다. 논문 전체 텍스트를 읽었고 모든 주요 figure 페이지(p.2, p.5, p.6, p.7, p.8, p.9, p.16, p.21)를 렌더링하여 확인했다. Dual Destination은 실제 코드, 대표 학습 당시 Hydra 설정, 체크포인트 36개, 평가 CSV 및 평가 bank까지 확인했다. Overcooked 학습·평가 코드는 검토하지 않았다. GSNR 인용 원 논문(Liu et al., 2020)과 E3T baseline 원 논문(Yan et al., 2023)의 원문은 확인했다. 논문이나 실험 코드는 수정하지 않았다.

표시: **확인**은 현재 파일에서 직접 확인한 사실, **해석**은 근거를 바탕으로 한 판단, **추가 확인**은 아직 단정하면 안 되는 사항이다.

## 1. 전반적인 인상

논문의 중심 이야기는 이해하기 쉽다. 환경 다양성을 늘리는 CEC에 actor–critic 분리와 IDAAC 보조 목적함수를 결합하고, 큰 배치에서 성능이 더 좋아지는지를 살펴보는 구성이다. 여러 종류의 파트너 평가와 인간 실험까지 포함한 점은 장점이다. Dual Destination의 현재 CSV에서는 DCEC가 fixed와 procedural 모두 가장 높은 평균을 보이며, 논문의 주요 순위 설명과 일치한다.

가장 큰 약점은 관측 결과와 원인 설명 사이의 거리다. 낮은 training value loss와 낮은 evaluation return의 동반 변화만으로 overfitting이나 shared representation의 인과적 책임을 확정하기는 어렵다. DCEC는 표현 분리뿐 아니라 auxiliary loss, 파라미터 수, optimizer와 gradient clipping 구성도 달라진다. 따라서 성능 향상 자체는 보여주더라도 이를 전부 decoupling의 효과라고 설명하려면 추가 ablation이 필요하다.

현재 원고에는 이전 실험 설명이 최종 실험과 섞여 있다. 특히 Dual Destination의 Empty/Wall 혼합 학습 설명은 Random3 학습과 맞지 않으며, E3T의 네트워크 크기도 공통 architecture 설명과 다르다. 통계 단위, 평가 에피소드 수, checkpoint 선택 규칙도 더 명시해야 한다. 이것들은 문장만 예쁘게 고치는 문제가 아니라 재현성과 결과 해석에 영향을 주는 수정 사항이다.

이전 대화에서 내가 권했던 표현도 재검토해야 한다. Dual Destination을 실제 minibatch가 65K인 것처럼 부르는 표현, raw reward가 약 -100인 결과를 그냥 “almost no reward”라고 표현하는 것, task별 oracle 없이 “maximum attainable reward”에 근접했다고 하는 표현은 정정하는 편이 정확하다. 또한 seed별 평균을 만든다고 공유 partner로 인한 의존성이 자동으로 없어지는 것은 아니다.

## 2. 논문 위치별 코멘트

- **[높음 / 해석] p.1 L014–023, p.2 L070–086, p.4 L197–212: overfitting과 원인의 단정.** Training value loss 감소와 held-out return 감소는 일반화 문제와 양립하지만, overfitting의 직접 증거로 충분하지 않다. On-policy value target과 방문 상태 분포가 각 정책마다 달라져 loss끼리도 완전히 같은 과제를 측정하지 않는다. `indicating overfitting`, `a major source ... is representation sharing`를 가설 수준으로 낮추고, 고정 validation bank에서 train/test 성능·value prediction error, capacity-matched shared 모델, decoupling-only ablation을 제시하는 것이 좋다.

- **[높음 / 해석] p.4 L200–206, Figure 2: r=0.83의 표본 수.** 설명대로라면 상관계수는 여섯 seed를 평균 낸 네 개 배치 설정에 대한 값이다. 독립 실험 24개를 사용한 상관관계로 읽히지 않도록 `across four batch-size averages`를 명시해야 한다. 네 점의 상관을 강한 메커니즘 증거로 해석하지 말고 seed별 분산과 불확실성을 보여주는 것이 좋다.

- **[높음 / 해석] p.2 L082–086, p.9 L439–450: batch scaling의 교란요인.** 총 environment steps를 맞췄다는 것은 총 rollout 횟수와 optimizer update 횟수까지 같다는 뜻이 아니다. Rollout 길이와 minibatch 수·epoch 수가 고정이면 N_env 증가에 따라 수집 iteration 및 optimizer step 수가 줄어든다. 배치당 환경 다양성, 최적화 횟수, 스케줄 변화가 함께 변한다는 점을 기재해야 한다. `same steps`만으로 batch size의 순수 인과효과를 주장하지 않는 편이 좋다.

- **[높음 / 해석] p.5 L240–255: GSNR로 일반화 원인을 입증하는 연결.** Value-gradient GSNR이 높아졌다는 것은 policy의 환경 불변성 또는 generalization이 개선된 원인을 직접 측정한 것은 아니다. 측정 parameter 집합, shared trunk와 critic head 포함 여부, sample의 단위, recurrent timestep 의존성, zero variance 및 log(0) 처리, seed별 오차범위를 명시해야 한다. `supports our hypothesis`는 가능하지만 입증한 것처럼 확대하지 않는 것이 좋다. **인용 원본 대조(완료):** Liu et al. (2020, ICLR)의 GSNR 정리는 동일 분포 train/test split에서 전체 학습 loss의 파라미터별 gradient에 대한 일반화 gap 관계를 다룬다. 이 논문은 그 개념을 (a) 전체 objective가 아닌 value loss에만, (b) i.i.d. train/test가 아닌 서로 다른 task(MDP) 간 transfer에 적용한다. 대상 loss와 분포 이동의 성격이 원 정리의 설정과 다르므로, `[Liu et al., 2020]`을 인용하면서 곧바로 `related to better generalization`이라고 쓰면 원 정리를 그대로 이전한 것처럼 읽힌다. 적용 범위 차이(다른 loss, 다른 형태의 distribution shift)를 한 문장으로 명시하는 것이 좋다.

- **[추가 확인] p.5 Figure 3, p.18 L936: 진단 실험의 학습 길이.** Figure 3의 가로축은 300 million steps까지이고 Table 2의 Overcooked budget은 3e9이다. 진단만 3e8 동안 돌린 것인지, 축 단위가 다른지, 표가 잘못된 것인지 설명이 필요하다. 현재는 코드 미검토이므로 오류로 단정하지 않는다.

- **[높음 / 확인] p.7 L325–328, p.16 L836–839: Dual Destination 학습 분포가 코드와 다름.** 원고는 매 episode Empty/Wall 두 variant 중 하나를 균등 선택한다고 설명한다. 최종 CEC/DCEC는 5×5에서 벽 세 칸을 무작위로 배치하는 Random3이다. Empty/Wall A는 fixed 평가 구성으로 소개하고, 학습 분포는 Random3로 고쳐야 한다. 자세한 근거는 3절.

- **[높음 / 확인] p.6 L309–317, p.18 L939–945: batch 정의의 적용 범위.** `Throughout this paper`라고 minibatch 표본 수를 정의한 뒤 N_env=256을 65,536으로 대응시킨다. 이것은 rollout 256인 Overcooked에만 맞는다. Dual Destination은 rollout 100이므로 minibatch는 25,600개 agent-transition이다. 해당 대응을 Overcooked에 한정하고 Dual은 N_env=256으로 표기해야 한다. 기존 파일명의 `65k`는 실제 표본 수의 근거가 아니다.

- **[높음 / 확인] p.17 L907–910: E3T가 동일 base encoder라는 설명.** Dual의 실제 E3T checkpoint는 Conv 512/256, encoder FC 512/256이다. IPPO/CEC 및 DCEC 각 trunk는 Conv 64/32, FC 512/512이다. E3T를 별도 표로 기술해야 한다. 같은 PPO hyperparameter와 같은 network architecture는 다른 주장이다.

- **[높음 / 확인] p.6 L293–300 Eq.(3), p.15 L769–777 Eq.(6)–(7): 실제 loss와 수식의 차이.** DCEC 구현은 clipped value loss와 1/2 계수를 사용한다. Advantage auxiliary target은 raw GAE가 아니라 minibatch에서 표준화한 GAE이며 loss에 1/2도 있다. 수식에 반영하거나 본문의 식이 개념적 표현이고 구현은 PPO clipping/normalization을 사용한다고 명시해야 한다. Terminal masking도 정의해야 한다.

- **[중간 / 확인] p.15 L783–804: discriminator 입력의 정확한 위치.** 원고의 h가 Eq.(1)의 recurrent output을 뜻한다면 코드와 다르다. 코드는 actor head의 마지막 hidden representation을 다음 timestep의 같은 representation과 연결한다. Adjacent pair, episode boundary 제외, 최종 timestep 제외, discriminator/encoder gradient 분리는 구현에 있다. 표현을 `policy-head features` 등으로 맞추면 된다.

- **[높음 / 확인] p.8 L381–388: “almost no reward”와 정규화 의미.** Procedural raw mean은 IPPO -99.494, E3T -99.174이다. 0에 가까운 것은 정규화값이지 raw return이 아니다. `rarely achieve simultaneous goal occupancy`가 정확하고도 자연스럽다. 정규화 식 자체는 plotting과 일치한다.

- **[중간 / 확인] p.8 L385–388: maximum attainable reward 표현.** 200은 step별 상한을 100번 더한 공통 상한이다. 초기 위치에서 이동이 필요한 각 task의 실제 최댓값은 더 낮다. 현재 bank의 동역학을 반영한 독립 joint-state BFS 계산으로 fixed oracle은 각 191, procedural 평균 oracle은 192.29였다. DCEC 평균은 각각 160.55, 172.286이다. 현재 문장은 정확한 수치나 공통 상한과 oracle의 구분을 덧붙이는 것이 좋다.

- **[중간 / 확인] p.5 Figure 3, p.9 Figure 8, p.9 Figure 9(a): 오차 표시 정의 누락.** Figure 3의 shaded band, Figure 8의 error bar, Figure 9(a)의 error bar 모두 캡션에 무엇을 나타내는지(seed 수, SD/SEM, map-averaging 방식) 설명이 없다. Figure 5/9(b)와 같은 기준으로 각 figure 캡션에 오차 표시의 정의와 반복 단위를 명시해야 한다.

- **[높음 / 확인] p.7 Figure 5 L372–374, p.19 L979–981: error bar와 평가 횟수 누락.** 현재 업로드 PDF 캡션에는 error bar 설명이 없다. 실제 plot은 방향을 합친 unordered seed pair 15개에 대한 SEM이고 pair끼리는 seed를 공유한다. `independent runs`의 SEM이라고 쓰면 안 된다. 각 ordered pair와 맵별 평가가 한 episode라는 사실도 빠져 있다.

- **[중간 / 해석] p.7 L348–356: fixed 평가도 환경 일반화가 섞임.** IPPO/E3T에는 학습한 fixed task에서 partner generalization을 측정하지만, CEC/DCEC에는 gradient training에서 제외한 fixed configuration으로의 transfer도 함께 측정한다. Fixed 평가 전체를 순수 partner generalization이라고 단정하는 문장은 조정하는 것이 좋다.

- **[중간 / 확인] p.7 L335–342: FCP의 benchmark 범위.** Baseline 목록에는 FCP가 포함되지만 Dual Figure 5 및 실제 Random3 XP에는 FCP가 없다. `FCP is evaluated on Overcooked only`처럼 범위를 명시해야 한다. FCP를 제외한 이유도 짧게 설명하는 것이 좋다.

- **[높음 / 확인] p.8 L426–434, p.20 Figure 13: “모든 구성에서 DCEC로 향한다”와 payoff의 충돌.** Overall 표에서 E3T×E3T=136.2인데 DCEC65K(row)×E3T(col)=125.8, 반대 방향은 121.0이다. 방향 평균을 써도 123.4로 E3T self-pair 값보다 낮다. 표를 payoff로 쓰는 표준 replicator dynamics에서 E3T 비중이 거의 1일 때 희귀 DCEC는 증가하지 않는다. CEC의 E3T 상대 값도 낮으므로 E3T/CEC/DCEC 게임에는 E3T 근방에서 DCEC가 줄어드는 구간이 있다. `Across all settings ... different population compositions`는 축소해야 한다. Figure 7에서 실제 사용한 payoff 변환과 row/column 역할 처리를 공개해야 한다.

- **[중간 / 확인] p.8 L411–412: Figure 7 캡션.** 그림은 simplex 위 replicator dynamics인데 캡션은 `Average cross-play reward`라고만 적혀 있고 `Empirical game-theoretic` 뒤 문장도 미완성이다. Vertex의 전략, 화살표의 방향·크기, color bar의 의미, payoff averaging을 설명해야 한다.

- **[중간 / 추가 확인] p.8 L433–434, p.20 Appendix D.2.** 본문은 fixed 및 procedural meta-game 행렬을 부록에서 제공한다고 하지만 D.2에서 확인된 Figure 13은 fixed 다섯 맵과 overall이다. Procedural payoff matrix도 제공하거나 참조 문장을 수정해야 한다.

- **[높음 / 추가 확인] p.9 L469–473와 p.22 L1169–1173: 인간 실험 유의성의 대상.** Figure 9는 네 개 survey 항목을 보여주며 모든 baseline과 Holm 보정 후 p<.001이라고 적혀 있다. Appendix E는 일곱 항목을 합친 preference score에 one-sided paired t-test를 했다고 설명한다. 합산점수의 유의성이 개별 네 항목 각각의 유의성을 뜻하지는 않는다. 어느 outcome에 대한 어느 비교인지, Holm family가 무엇인지, n=20 participant 평균인지, effect size와 corrected p를 표로 밝혀야 한다. 실제 계산이 틀렸다고 단정한 것은 아니다.

- **[중간 / 확인] p.9 L482: “8K to 65K training partners”.** 8K/65K는 partner 수가 아니라 PPO minibatch size이다. `increasing the PPO minibatch size from 8K to 65K`로 고쳐야 한다.

- **[중간 / 추가 확인] p.9 L478–479와 p.22 L1140–1143: 인간 실험 순서.** 본문은 agent-layout 조합을 무작위 순서로 제시했다고 쓰고, 부록은 agent 순서를 무작위화했다고 쓴다. Layout 순서도 randomize했는지 실제 절차에 맞춰 통일해야 한다. Random assignment가 role별 balanced assignment와 같은 것은 아니다.

- **[중간 / 확인] p.22 L1157–1173, p.23 Figure 14: 설문 정의와 집계.** 본문은 모두 strongly disagree–strongly agree라고 하지만 Q7 그림은 very poor–very good이다. Q7의 다른 응답 anchor를 명시하고, Figure 14의 count는 participant×layout 응답인지 설명해야 한다. Cronbach alpha=.91만으로 일곱 문항이 단일 개념이라는 결론이 보장되는 것은 아니며, 반복응답을 독립 참여자로 취급하지 않아야 한다.

- **[중간 / 확인] p.8 L390, L415: 6.3과 6.4 제목 중복.** 두 절 모두 Performance on Overcooked이다. 성능 결과를 합치고 EGTA를 별도 subsection으로 분리하는 편이 자연스럽다.

- **[높음 / 확인] p.21 L1080–1133: Human-proxy 부록이 비어 있음.** 텍스트 추출뿐 아니라 페이지를 렌더링하여 확인했다. 해당 평가를 했다는 본문 주장에 대응하는 결과, proxy 출처·선택 방법·평가 조건을 넣어야 한다.

- **[높음 / 확인] p.11 L540–577: 제출용 statement가 template 상태.** AI use, Ethics, Reproducibility에 지시문과 placeholder가 남아 있다. 실제 수행한 내용을 적어야 한다. 인간 실험 승인·동의·보상은 Appendix E의 기술과 일치하게 작성해야 하며, 확인하지 않은 사실을 채우면 안 된다. 학회 정책의 최신 요건 자체는 이번 검토에서 확인하지 않았다.

- **[중간 / 해석] p.6 L301–307, p.8 L417–425, p.10 L488–494: DCEC 전체와 decoupling-only의 구분.** Full DCEC의 성능으로 auxiliary loss 없이 네트워크만 분리해도 같은 효과가 난다고 결론 낼 수 없다. Decoupled-only, DAAC, IDAAC, capacity-matched shared 비교를 제시하거나 표현을 `the IDAAC-based DCEC architecture`로 한정해야 한다.

- **[중간 / 해석] p.10 L495–501: limitation 범위가 좁음.** Partner diversity 결합을 하지 않았다는 점 외에도 독립 학습 seed 수, 실험 benchmark 범위, method-dependent architecture, 통계적 의존성, 인간 표본 수 등을 짧게 언급하면 주장과 한계가 균형을 이룬다.

- **[낮음 / 확인] p.3 L139–151: 형식 정의 정돈.** H-step return은 보통 t=0,...,H-1인데 여기서는 H까지 합산한다. Observation을 쓰면서 observation function 정의가 없고, task가 initial distribution만 바꾼다는 설명은 layout/object configuration이 state에 포함된다는 정의가 있어야 자연스럽다. Recurrent policy도 현재 observation만이 아니라 history 또는 recurrent state에 조건화된다는 표기를 정리하면 좋다.

- **[낮음 / 확인] p.3 L158, p.4 L162, p.8 L425, p.15 L765/L804, p.17 L900: 표현·문장부호.** `partners` 뒤 마침표, `Cooperation (CEC)` 띄어쓰기, L425/L804의 종결 마침표, `bonus to encourage exploration`, `DCEC uses` 등을 수정한다. p.1 L027의 `with human–AI`는 `in human–AI coordination`이 자연스럽다.

- **[낮음 / 확인] p.19 L1015 및 p.24 L1266–1267/L1290–1291: caption 재사용.** Figure 12는 맵별 cross-play 막대인데 environment scaling 캡션을 사용한다. Figure 15/16은 각각 CEC/DCEC value-loss vs return 그림인데 두 모델을 함께 다룬다는 동일 캡션이 붙었다. 실제 표시 대상과 맞춘다. Figure 13 Overall의 DCEC(32)/(256)도 다른 패널의 (8K)/(65K)와 통일한다.

## 3. Dual Destination 코드 대조

### 3.1 확인한 자료

- 학습 코드: [Random3 CEC](../../baselines/CEC_UED/random3_cec_dual_destination_with_xp.py), [Random3 DCEC](../../baselines/CEC_UED/random3_dcec_dual_destination_with_xp.py), [IPPO](../../baselines/CEC_UED/modified_wall_ippo_general_dual_destination_with_xp.py), [E3T](../../baselines/CEC_UED/modified_wall_e3t_dual_destination_with_xp.py).
- 환경: [Random3](../../jaxmarl/environments/toy_coop/toy_coop_no_pink_random3.py), [NoPink](../../jaxmarl/environments/toy_coop/toy_coop_no_pink.py), [기본 layout·step 정의](../../jaxmarl/environments/toy_coop/modified_wall_toy_coop.py).
- 평가: [Random3 XP](../../baselines/CEC_UED/random3_procedural_xp_eval.py), [평가 YAML](../../baselines/CEC_UED/xp_config/random3_procedural_xp.yaml), [plot 코드](../../baselines/CEC_UED/plot_random3_xp.py).
- 학습 당시 설정: [IPPO](../../outputs/2026-09-02/07-44-26/.hydra/config.yaml), [E3T](../../outputs/2026-09-02/03-36-07/.hydra/config.yaml), [CEC](../../outputs/2026-09-19/11-22-52/.hydra/config.yaml), [DCEC](../../outputs/2026-09-19/11-23-02/.hydra/config.yaml). 네 파일은 대표 실행 설정이며 모든 seed의 과거 설정을 각각 대조한 것은 아니다.
- 수치: [plot values](../../baselines/CEC_UED/results/random3_xp/plots/random3_65k_plot_values.csv), `results/random3_xp/*_episodes.csv`, `random3_heldout_100.npz`.

### 3.2 실험의 실제 구성

| 항목 | 확인된 내용 |
|---|---|
| 환경 | 5×5, agent 2명, 동일한 goal 2개, pink goal 없음 |
| 보상 | 다음 상태에서 두 agent가 서로 다른 goal에 동시에 있으면 각자 +2, 아니면 -1 |
| 종료 | 100 steps, 성공해도 즉시 종료하지 않고 계속 보상을 받음 |
| 관측 | agent 자신/상대/goal/wall의 4채널, full spatial observation |
| CEC/DCEC 학습 | Random3 벽 세 칸 + agent/goal 위치 무작위, 유효성 검사 및 held-out 제외 |
| IPPO/E3T 학습 | Empty와 Wall A 각각 고정 학습, random_reset=false |
| held-out | procedural 초기 구성 100개 + fixed 초기 구성 2개 |
| 학습 seed | 각 모델군 0–5. IPPO/E3T는 map마다 6개, CEC/DCEC는 각각 6개 |
| XP pair | 같은 알고리즘 내 서로 다른 seed 30개 ordered pair, 동일 seed 제외 |
| 방향 | i→agent_0, j→agent_1 및 반대 방향 모두 평가 |
| 반복 | 각 ordered pair×task마다 한 episode |
| 행동 | argmax가 아니라 categorical sampling, beta=1 |
| procedural 평가 | 모델군마다 동일한 100개 task, 3,000 episode |
| fixed 평가 | CEC/DCEC는 각 60 episode, IPPO/E3T는 map별 모델군마다 30 episode |
| plot 집계 | 방향 평균 후 15개 unordered pair, IPPO/E3T는 두 train-map 모델군 평균 |

근거: 평가 코드 L31–73, L382–456, L488–514, L591–655. 실제 CSV에서도 동일 seed 0건, ordered pair 30개, pair×task당 row 1개를 확인했다. Procedural 위치가 항상 좌우로 정렬된 것은 아니므로 일반적인 설명은 “agent slot 양방향”이 정확하다.

### 3.3 학습 설정과 checkpoint

공통 PPO 값은 대표 Hydra 설정에서 LR=3e-4, gamma=.99, GAE lambda=.95, clip=.2, entropy=.005, VF=1, max grad norm=.5, anneal=true, NUM_ENVS=256, NUM_STEPS=100, minibatches=2, epochs=32, budget=1e8이다. 현재 Table 2의 Dual 열은 이 값들과 맞는다. Epoch 32라는 숫자만으로 오류라고 볼 근거는 없다.

실제 업데이트는 floor(1e8/(256×100))=3,906회이며 총 environment transition은 99,993,600이다. Rollout당 agent-transition은 51,200개, minibatch당 25,600개이고 update당 optimizer step은 32×2=64회다. `65K`라는 내부 이름을 실제 batch size로 옮겨 적으면 안 된다.

현재 evaluator 패턴으로 6개 모델군×6 seed=36개 checkpoint를 읽었다. 각 seed의 matching file은 정확히 하나였고, 모두 update_steps=3906, 배열 값은 finite, 각 군 내 seed별 parameter hash는 서로 달랐다. 현재 동명이인 checkpoint 중 잘못 선택한 증거는 없다. 다만 평가 loader L183은 여러 파일이 있으면 mtime이 가장 최신인 것을 택하므로 향후 결과 고정용 manifest에는 파일 경로와 hash를 남기는 것이 좋다.

아래 경로의 `lr-*` 아래 `seed{0..5}_ckpt0_*_updates3906.pkl`이 평가 대상이다.

| 모델 | 경로 |
|---|---|
| IPPO | `ckpts/ippo/ToyCoopNoPink/modified_wall/{empty,wall_a}_with_xp_numenv256/ikFalse/reset_all/ippo_layout_eval/` |
| E3T | `ckpts/e3t/ToyCoopNoPink/modified_wall/{empty,wall_a}_with_xp_numenv256/ikFalse/reset_all/e3t/` |
| CEC | `ckpts/ippo/ToyCoopNoPink/modified_wall/random_3_walls_with_xp_numenv256/ikTrue/reset_all/cec_layout_eval/` |
| DCEC | `ckpts/idaac/ToyCoopNoPink/modified_wall/random_3_walls_with_xp_numenv256/ikTrue/reset_all/` |

CEC L1172–1178 및 DCEC L1796–1824의 저장 로직은 최종 모델이다. 파일명 `improved`가 best-validation checkpoint를 뜻하지 않는다. 논문에는 final checkpoint selection임을 명시하는 편이 좋다.

### 3.4 네트워크와 DCEC loss의 세부 불일치

| 모델 | Conv filter | Encoder FC | checkpoint 전체 parameter 수 |
|---|---|---|---:|
| IPPO/CEC | 64, 32 | 512, 512 | 1,873,190 |
| E3T | 512, 256 | 512, 256 | 4,901,387 |
| DCEC | 각 trunk 64, 32 | 각 trunk 512, 512 | 3,343,628 |

이 수치는 checkpoint의 실제 array shape/count로 확인했다. 모든 모델의 저장된 LSTM recurrent kernel은 hidden size 256에 해당한다. DCEC parameter 수에는 학습용 critic 및 auxiliary module이 포함되어 있으므로 inference actor만의 크기와 혼동하면 안 된다.

- **확인:** E3T 코드 L169–198의 layout_name 조건으로 empty/wall_a에서는 큰 Conv와 작은 FC2가 선택된다. “동일 base architecture” 문장은 수정이 필요하다.
- **확인:** DCEC L536–545에서 policy/value trunk가 분리된다. L828–850에서 policy/value/classifier별 optimizer와 gradient clipping도 분리된다. CEC는 shared 모델 전체에 clipping을 적용한다. 같은 .5라는 설정만으로 optimizer 동작이 동일한 것은 아니다.
- **확인:** DCEC L1140–1161의 value clipping, normalized GAE, 1/2 계수는 논문 Eq.(3)/(7)에 그대로 기술되어 있지 않다.
- **확인:** DCEC L603–642의 advantage/discriminator 입력은 actor head의 penultimate feature이다. L1109–1117은 episode boundary와 마지막 timestep을 제외하며, L1214–1248은 classifier gradient를 분리한다. 분리 자체가 깨져 있다는 증거는 발견하지 않았다.
- **확인:** 코드 default는 DAAC_ADV_COEF=.25, IDAAC_ORDER_COEF=.001, nonlinear classifier=false, classifier LR=policy/value LR=3e-4이다(L724–730, L1855–1861). Table 2에 auxiliary coefficient, classifier 구조, update 빈도도 추가해야 한다.
- **해석:** 시간 순서를 구분하기 어렵게 만드는 objective가 task 전체에 대한 invariance를 직접 보장하지는 않는다. `encourages` 정도의 표현이 적절하다. 원조 IDAAC와 동일 구현이라고 하기보다는 recurrent multi-agent adaptation이라고 설명하는 편이 정확하다.

### 3.5 Held-out과 맵 샘플러

CEC L215–234, DCEC L255–275에서 공통 생성 함수로 procedural 100개를 만들고 fixed 2개를 덧붙인다. Random3 환경 L81–103은 wall, agent 위치 집합, goal 위치 집합이 함께 일치하는지 확인한다. Agent/goal 순서를 바꾼 구성도 같은 것으로 취급한다. L119–132에서 일치하면 반복 재샘플링한다.

따라서 제외하는 것은 **102개의 초기 task configuration**이다. Wall A 벽 형태 자체를 모든 agent/goal 배치에 걸쳐 제외하는 것은 아니며, held-out 초기 상태와 같은 상태를 episode 도중 방문하는 것까지 금지하지도 않는다. 이는 초기 task를 hold out하는 설계로서 설명하면 된다. Empty는 0-wall이므로 Random3의 3-wall training distribution 밖이다. Procedural 100개는 동일 생성 규칙 내 미사용 구성에 해당하며, unseen wall-family 일반화라고 과장하지 않아야 한다.

현재 저장 bank의 배열 hash는 `d7a2f35e3d9bfa4637d2f38830f86120553582a416295a93a56553bc7da7db5c`이다. 현재 evaluator는 bank를 현재 코드로 재생성하여 비교한다(L272–285). 학습 checkpoint에는 당시 held-out hash가 저장되어 있지 않으므로 과거 실행까지 hash로 증명한 것은 아니다. 이것은 기록의 한계이지, 서로 다른 서버에서 다른 맵을 썼다는 증거가 아니다.

**새로 발견한 샘플러 문제: 좌표 flatten 순서가 다르다.** `modified_wall_toy_coop.py:66`의 all_pos는 x를 바깥 loop로 나열한다. 반면 `toy_coop_no_pink_random3.py:156–158`의 wall_map.reshape(-1)는 y를 바깥 축으로 펼친다. 따라서 free mask가 허용한 index를 all_pos로 변환할 때 x/y가 뒤바뀐다. 예를 들어 실제 wall[y=0,x=1]을 막았는데 후보 목록에서 실제 위치 (1,0)가 허용될 수 있다.

Random3에서는 이후 reachability 검사(L167–201)가 실제 벽 위의 agent/goal을 거부하므로 최종 102개 bank에 벽 위의 agent/goal은 없었다. 하지만 sampling candidate 단계에서 전치된 벽 위치를 이미 제외한 뒤 실제 벽 위치도 거부하므로, 최종 위치는 W와 W의 transpose 양쪽을 피하게 된다. 실제 bank에서도 두 종류의 위치 겹침은 모두 0이었다. **최종 맵이 불가능하다는 문제가 아니라, 의도한 free-cell sampling보다 분포가 제한되는 구현 문제**다. 학습/평가에 공통 적용되므로 이것만으로 DCEC 우위가 조작되었다거나 사라진다고 말할 근거는 없다. 수정하면 생성 bank가 달라질 수 있으므로 기존 모델에 새 bank를 덮어씌우지 말고 실험 버전을 분리해야 한다. 이번 리뷰에서는 수정하지 않았다.

### 3.6 보상·정규화·oracle

보상 근거는 NoPink L117–132, step horizon은 부모 환경의 step_env/is_terminal이다. 성공한 timestep 수를 K라고 하면 R=2K-(100-K)=3K-100이고, 현재 plot의 (R+100)/300은 정확히 K/100이다. 이는 episode success rate와 다르다. Evaluator CSV의 success는 R>-100, 즉 episode 중 한 번이라도 동시 goal 점유를 했는지다.

| 모델 | Fixed raw | Fixed normalized | Procedural raw | Procedural normalized |
|---|---:|---:|---:|---:|
| IPPO | 35.800 | .452667 | -99.494 | .001687 |
| E3T | 107.650 | .692167 | -99.174 | .002753 |
| CEC | -83.250 | .055833 | 36.492 | .454973 |
| DCEC | 160.550 | .868500 | 172.286 | .907620 |

Raw episode CSV에서 직접 계산한 평균은 plot_values와 일치했다. CEC fixed는 Empty -70.4, Wall A -96.1이며, DCEC는 Empty 173.4, Wall A 147.7이다. 따라서 fixed에서 CEC의 낮은 결과가 한 map의 averaging 착오 때문이라는 증거는 없다.

Oracle 진단은 checkpoint rollout을 다시 돌린 것이 아니다. 저장 bank에서 코드와 동일한 경계 clipping, wall 충돌, 두 agent의 동일 칸 이동 취소, 서로 위치 교환 허용을 적용하는 joint-state BFS를 별도로 계산했다. 최초 동시 goal 도달 이동 횟수가 d이면 도달 step부터 +2이므로 oracle return은 200-3(d-1)이다. 현재 두 fixed task는 d=4로 각 191이며, procedural 평균은 192.29, 같은 min-max normalization으로 .9743이다. DCEC procedural .90762는 oracle보다 .06668 낮다. 이 수치를 최종 원고에 새로 넣으려면 독립 BFS 구현과 실제 환경의 전이 일치 검사를 추가한 뒤 별도 oracle 산출물로 보존하는 것이 좋다.

주의: evaluator가 CSV에 기록하는 `normalized_return`은 아직 R/200이다(L637, L654). 최종 plot은 reward_mean에서 (R+100)/300을 다시 계산한다(plot L60). 현재 그림은 논문 식과 맞지만, CSV normalized 컬럼을 그대로 재사용하면 다른 수치가 나온다.

### 3.7 통계와 평가 해석

- **확인:** plot L62–84는 15개 unordered pair의 평균과 pandas SEM을 사용한다. 대략 SD(pair mean)/sqrt(15)이다. 각 pair가 같은 여섯 seed를 공유하므로 독립 run 15개의 SEM은 아니다. 상관에 따라 오차 추정이 왜곡될 수 있으며 반드시 항상 과소추정이라고 단정하지는 않는다.
- **보완:** Seed별 평균도 동일 상대 seed를 공유하므로 그 여섯 평균이 완전히 독립이라고 말하면 안 된다. Training seed를 재표집 단위로 하는 적절한 dyadic/seed-cluster 방법이나 leave-one-seed-out 민감도 분석을 고려해야 한다. Seed가 여섯 개뿐이라는 한계도 남는다.
- **확인:** Fixed map에서 pair·방향당 한 stochastic episode만 있으므로 정책 sampling noise를 반복 episode로 분리해 추정하지 못한다. Procedural에서도 task마다 한 episode다. 잘못된 평균 계산은 아니지만 논문에 평가 반복 수를 명시하고 fixed 반복 평가를 늘리는 것이 좋다.
- **확인:** 모든 pair는 같은 action seed=2026에서 task index로 만든 RNG를 쓴다. Reproducible common randomness이며 독립 action-repeat 여러 번을 돌린 것과 다르다.
- **확인:** IPPO/E3T procedural 막대는 Empty-trained 6개 모델군과 Wall-A-trained 6개 모델군 결과를 평균한 것이다. 두 모델군끼리 cross-play한 것은 아니다. 이 집계 규칙이 논문에 필요하다.
- **추가 확인:** 학습 중 기존 seed98과 fixed XP를 주기적으로 기록한다. 코드상 seed98은 평가용이고 gradient training partner로 들어가지 않는다. 그러나 그 결과를 보고 hyperparameter나 학습 분포를 선택했는지는 코드만으로 알 수 없다. 이 경우 fixed map은 optimizer에서 held out되었더라도 전혀 보지 않은 최종 test set이라고 부르기는 어렵다.

### 3.8 추가로 발견한 E3T 학습 쟁점

E3T L563–573은 환경마다 한 agent의 logits에 .55를 곱해 sampling하고 그 distribution의 log_prob를 저장한다. L668–689의 PPO loss에서는 .55를 재적용하지 않은 pi의 log_prob를 numerator로 사용하며, partner slot을 제외하는 mask가 없다. 따라서 tempered slot의 ratio는 parameter update 이전에도 일반적으로 1이 아니다.

이는 표준적인 동일 behavior-policy PPO update와 다른 구조다. 현재 결과 전체를 무효라고 단정하지 않는다. 다만 baseline 재현성 검토에서 우선 확인할 항목이며, 코드가 이 동작을 한다는 사실은 확인했다.

**원조 E3T와의 대조(완료):** Yan et al. (2023, NeurIPS, "An Efficient End-to-End Training Approach for Zero-Shot Human-AI Coordination") Eq.(4)와 Algorithm 1을 직접 확인했다. 두 가지가 다르다. (1) 원조 partner mixture는 로짓 온도가 아니라 별도 균일분포 π_r과의 명시적 convex combination이다: π_p = ε·π_r + (1-ε)·π_e. 로짓에 .55를 곱해 같은 categorical distribution을 완만하게 만드는 것은 균일분포와의 mixture와 수식이 다르다. (2) Algorithm 1 L7–11과 4.3절에 따르면 PPO는 tempering되지 않은 ego 슬롯의 행동 a_t^e에만 적용된다. Tempered partner 슬롯의 행동 a_t^p는 환경 전이에는 관여하지만 PPO 목적함수에 들어가지 않는다. 즉 원조 방법은 tempered 슬롯을 애초에 학습 대상에서 제외하므로 old/new log_prob 불일치 문제 자체가 생기지 않는다. 현재 코드는 tempered 슬롯의 transition도 마스킹 없이 PPO 업데이트에 포함하면서 old_log_prob만 tempered 분포로 기록하므로, 원조 알고리즘에는 없는 조합이다. 따라서 이 동작은 의도적인 off-policy correction이라기보다 원조 알고리즘과 어긋나는 구현일 가능성이 크다.

### 3.9 현재 주장에 대한 판단

“이번 cross-seed 평가에서 DCEC의 평균이 모든 비교 모델보다 높다”는 결론은 CSV가 지지한다. “Random3 procedural tasks에서 fixed-trained IPPO/E3T보다 CEC가 높고 DCEC가 더 높다”도 맞다. 하지만 이것이 특정 원인인 value interference 완화 때문이라는 결론, 임의 알고리즘·인간 partner까지 Dual에서 일반화한다는 결론, 실제 oracle와 거의 같다는 결론은 각각 별도 근거 또는 표현 조정이 필요하다.

수정 문장의 출발점으로 다음 표현을 권한다.

> For cross-environment training in Dual Destination, we generate 5-by-5 tasks with three randomly placed wall cells and randomized agent starting positions and goal locations. We exclude 100 procedural initial configurations and the two fixed evaluation configurations from training resets. We evaluate independently trained policies using all 30 ordered cross-seed pairings among six seeds, excluding same-seed pairs.

위 문장에 uniform free-cell sampling을 추가하려면 3.5절의 좌표 문제를 먼저 처리해야 한다. 평가 프로토콜에는 stochastic episode 반복 수와 map/model-group averaging 설명도 덧붙여야 한다.

## 4. 이어서 확인할 항목

- ~~전체 figure의 화살표, 축 범위, 오차막대, 글씨 크기를 원본 PDF 이미지로 최종 점검.~~ **완료.** p.2, p.8, p.21에 이어 p.5/6/7/9/16도 렌더링 이미지로 확인했다. 화살표·축 범위·글씨 크기에서 새로운 문제는 없었다. 다만 Figure 3(p.5)의 shaded band, Figure 8(p.9, batch scaling)과 Figure 9(a)(p.9, human coordination bar)의 error bar가 캡션에 정의 없이 등장한다. Figure 5(p.7)의 error bar 문제는 이미 위에 기록되어 있었고, Figure 8/9(a)는 이번에 새로 확인한 항목이다. 이 둘은 Overcooked 결과이므로 코드로 재검증하지 않고 원고상 캡션 누락으로만 지적한다.
- ~~GSNR의 개념적 주장을 뒷받침하는 cited primary source와 실제 진단 프로토콜의 대응.~~ **완료.** Liu et al. (2020, ICLR)을 확인했고 위 2절 해당 항목에 적용 범위 차이를 추가했다. Overcooked 진단 코드 자체는 이번 범위에서 읽지 않았다.
- ~~E3T tempered-policy update가 원조 방법의 의도인지 대조.~~ **완료.** Yan et al. (2023, NeurIPS) 원문을 확인했고 3.8절에 구체적 불일치를 기록했다.
- Fixed test monitoring을 이용한 모델·설정 선택 여부는 저자의 실험 이력 확인이 필요하다. (미해결 — 코드나 논문만으로 답할 수 없음)
- 논문용 유의성 검정을 새로 실행하지 않았다. Human p-value, alpha, IRB 관련 사실은 원고의 주장으로만 취급했다. (미해결 — 새 통계 검정을 요청받지 않았음)
- 임시 진단 스크립트는 `/tmp/audit_paper_results.py`와 `/tmp/audit_paper_checkpoints.py`에 있다. 전자는 bank/CSV/oracle, 후자는 JAX를 로드하지 않고 checkpoint array shape·finite·hash를 검사한다. 학습 및 GPU 평가 재실행은 하지 않았다.

참고한 외부 1차 자료: [IDAAC 원논문](https://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf)은 분리된 policy/value와 auxiliary objectives의 출처 및 adaptation 범위 확인에 사용했다. [Dyadic data variance 논문](https://arxiv.org/abs/1312.3398)은 구성원을 공유하는 pair 사이의 의존성을 고려해야 한다는 통계적 근거다. 이 자료가 현재 XP에 특정 estimator를 그대로 적용하면 충분하다는 뜻은 아니다.

## 5. 리뷰어가 공격할 수 있는 지점

- **“CEC에 IDAAC를 붙인 것 이상의 기여는 무엇인가?”** 새 알고리즘 발명보다 cross-environment scaling에 대한 실증 발견과 recurrent multi-agent adaptation을 분명히 해야 한다.
- **“Overfitting이라고 하지만 value target과 optimizer update 횟수가 모두 다른 것 아닌가?”** 현재 training-loss 상관과 총 environment steps 통제만으로는 반박이 충분하지 않다.
- **“효과가 decoupling 때문인가, 더 큰 모델·auxiliary loss·optimizer 분리 때문인가?”** Capacity 및 component ablation이 필요하다.
- **“최종 학습 맵 분포를 원고에 다르게 적은 이유는 무엇인가?”** Empty/Wall 혼합 설명을 Random3와 맞추고 sampler의 실제 제약까지 정리해야 한다.
- **“E3T도 같은 architecture라고 했는데 왜 checkpoint 차원이 다른가?”** 실제 architecture를 공개하고 공정성 주장을 좁혀야 한다.
- **“E3T baseline이 원조 논문의 mixture policy 및 PPO 적용 범위와 다른데, 이 baseline이 공정하게 재현되었다고 할 수 있는가?”** 원조는 tempered partner 슬롯을 PPO 대상에서 제외하지만 현재 구현은 포함한다. Baseline 재현 방법을 원문과 대조해 명시해야 한다.
- **“Fixed에서 한 episode씩이고 pair들이 seed를 공유하는데 error bar를 신뢰할 수 있는가?”** 반복 평가 수와 uncertainty estimator를 명시해야 한다.
- **“Test task를 학습 중 평가해서 방법을 선택한 것 아닌가?”** Gradient exclusion과 model-selection holdout을 구분하여 실험 이력을 설명해야 한다.
- **“배치가 실제로 25,600인데 왜 65K라고 부르는가?”** Dual에는 N_env=256을 쓰고 Overcooked batch-size 명명과 구분해야 한다.
- **“높은 cross-seed XP가 정말 낯선 partner 적응인가, 공통 convention 수렴인가?”** Dual 결과의 범위를 same-method unseen seeds로 한정하고, 적응 메커니즘을 별도로 입증하지 않았다면 단정하지 않아야 한다.
- **“EGTA가 모든 population에서 DCEC 우세라고 하지만 payoff 표는 반례를 주지 않는가?”** Universal dominance 주장을 수정하고 basin 및 게임 구성 의존성을 보여야 한다.
- **“인간 실험의 p<.001은 합산점수인가 각 문항인가?”** Participant 단위 비교와 multiple-comparison family를 공개해야 한다.
- **“원래 방법보다 유리한 환경·seed·설정만 선택해서 보여주는 것 아닌가?”** 실험 선택 기준, 제외한 변형, 최종 평가 bank와 checkpoint provenance를 투명하게 기록해야 한다. 현재 자료만으로 선택 편향이 있었다고 단정하는 것은 아니다.
