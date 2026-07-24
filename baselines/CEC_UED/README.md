# CEC-UED training and diagnostic metrics

이 문서는 `ippo_general_gradient.py`와
`ippo_general_gradient_pop.py`가 Weights & Biases(W&B)에 기록하는 지표를
정리한다. 별도 언급이 없다면 actor와 critic은 shared trunk를 포함한다.

## 기록 시점

```text
LOG_INTERVAL = max(1, NUM_UPDATES // 100)
```

W&B의 global step은 0부터 시작하는 PPO `update_step`이다. 현재 코드의
`env_step`은 다음과 같이 기록된다.

$$
\text{env\_step}
= \text{update\_step}\times\text{NUM\_ENVS}\times\text{NUM\_STEPS}.
$$

지표는 기록 방식에 따라 세 종류로 나뉜다.

| 종류 | 기록 방식 |
|---|---|
| PPO loss, entropy, ratio, optimizer gradient norm, parameter weight norm, `returns` | interval 동안 finite 값의 평균 |
| target, critic RMSE, TD-error RMSE | 해당 logging update의 rollout snapshot |
| gradient conflict, representation feature/rank | 해당 logging update에서만 계산한 pre-update snapshot |
| evaluation | 가장 최근 diagnostic evaluation 결과 |
| layout별 training return | interval에서 종료된 episode들의 평균 |

마지막 PPO update는 interval 경계와 일치하지 않더라도 기록된다. Gradient
conflict와 representation feature/rank는 PPO parameter update 직전의 parameter
및 rollout로 계산된다. Weight norm은 각 PPO minibatch optimizer step 직전
parameter에서 계산하지만 W&B 값은 interval 동안 평균한다.

## 표기법

- $N$: environment sample 수. 기본 설정에서는 `NUM_ENVS=256`.
- $T$: `NUM_STEPS`.
- $A$: `NUM_ACTORS`. 기본 Overcooked 설정에서는 environment당 두 actor이므로
  $A=2N$이다.
- $A_m=A/\texttt{NUM\_MINIBATCHES}$: 한 PPO minibatch의 actor 수.
- $g_i$: environment sample $i$ 또는 layout $i$의 gradient.
- $\theta_k$: parameter tree의 $k$번째 array leaf.
- $n_k$: $\theta_k$의 원소 수.
- $\Phi\in\mathbb{R}^{M\times D}$: rollout에서 수집한 penultimate feature
  matrix. 시간과 actor 축을 $M$개의 sample 축으로 합친다.
- $\sigma_1\ge\cdots\ge\sigma_r$: $\Phi$의 singular values.
- $\epsilon=10^{-8}$: 수치 안정성을 위한 작은 상수.
- `<layout>`: `cramped_room_9`, `asymm_advantages_9`, `coord_ring_9`,
  `counter_circuit_9`, `forced_coord_9` 중 하나.

## 실제 PPO gradient norm

이 지표들은 gradient clipping과 Adam 적용 전의 total PPO gradient를 각
minibatch에서 측정한 뒤 epoch/minibatch 및 logging interval에 걸쳐 평균한다.
Bias와 1D parameter를 포함한 모든 array leaf가 참여한다.

Global L2 norm은 다음과 같다.

$$
\|g\|_2=\sqrt{\sum_k\sum_j g_{k,j}^2}.
$$

현재 코드의 leaf-count weighted RMS는 다음과 같다.

$$
\operatorname{WeightedRMS}(g)=
\sqrt{
\frac{\sum_k n_k\|g_k\|_2^2}
{\sum_k n_k}
}.
$$

이는 일반적인 elementwise RMS
$\sqrt{\sum_k\|g_k\|_2^2/\sum_k n_k}$와 다르며, 큰 leaf에 더 큰 가중치를
준다.

| W&B key | 포함 범위 |
|---|---|
| `gradient_norm/global_norm` | 전체 parameter tree |
| `gradient_norm/weighted_rms_norm` | 전체 parameter tree |
| `gradient_norm/actor_weighted_rms_norm` | shared trunk와 actor branch/output |
| `gradient_norm/critic_weighted_rms_norm` | shared trunk와 critic branch/output |
| `gradient_norm/shared_weighted_rms_norm` | shared trunk |

Actor와 critic 값에는 같은 shared trunk가 각각 포함된다. 또한 total PPO
gradient에서 module만 선택하므로 순수 actor-loss gradient와 순수 value-loss
gradient를 분리한 값은 아니다.

## Target, critic 및 TD error

GAE의 TD residual과 advantage는 다음과 같다.

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
$$

$$
\hat A_t=\delta_t+
\gamma\lambda(1-d_t)\hat A_{t+1},
\qquad
y_t=\hat A_t+V(s_t).
$$

일반 CEC에서는 모두 raw reward scale이다.

| W&B key | 수식 및 의미 |
|---|---|
| `target_raw/mean` | $\operatorname{mean}(y_t)$ |
| `target_raw/<layout>/mean` | 해당 layout의 $\operatorname{mean}(y_t)$ |
| `critic/rmse` | $\sqrt{\operatorname{mean}((y_t-V(s_t))^2)}$ |
| `critic/<layout>/rmse` | 해당 layout의 critic RMSE |
| `td_error/rmse` | $\sqrt{\operatorname{mean}(\delta_t^2)}$ |
| `td_error/<layout>/rmse` | 해당 layout의 TD-error RMSE |

### PopArt scale

PopArt는 raw target $y_t$를 다음과 같이 정규화한다.

$$
\tilde y_t=\frac{y_t-\mu}{\sigma}.
$$

| W&B key | scale |
|---|---|
| `target_raw/*` | 역정규화된 실제 reward scale |
| `target_popart/mean` | normalized target scale |
| `target_popart/<layout>/mean` | normalized target scale |
| `target_popart/<layout>/std` | normalized target scale |
| `critic/*/rmse` | normalized target/value scale |
| `td_error/*/rmse` | raw reward scale |

따라서 일반 CEC와 PopArt CEC의 `critic/rmse`는 같은 이름이어도 scale이
다르므로 직접적인 수치 비교에 주의해야 한다.

## Layout-family gradient

Gradient conflict 진단에는 rollout 전체가 아니라 앞쪽
`GRAD_CONFLICT_WINDOW_STEPS` 구간을 사용한다.

Layout $l$에서 관측된 sample 수를 $n_l$, 전체 sample 수를
$n=\sum_l n_l$, layout 평균 gradient를 $g_l$이라 한다. 두 agent는 같은
environment layout에 속하며 gradient 계산에는 두 agent가 모두 포함된다.

### Sample share와 norm

$$
p_l=\frac{n_l}{n},
\qquad
G=\sum_l n_lg_l.
$$

| W&B key | 수식 |
|---|---|
| `sample_share/<layout>` | $p_l$ |
| `grad_norm_actor/<layout>` | $\|g_l^{\mathrm{actor}}\|_2$ |
| `grad_norm_critic/<layout>` | $\|g_l^{\mathrm{value}}\|_2$ |
| `grad_norm_actor/total` | $\|G^{\mathrm{actor}}\|_2/n$ |
| `grad_norm_critic/total` | $\|G^{\mathrm{value}}\|_2/n$ |

### Layout contribution

방향을 무시한 sample-weighted gradient 크기는 다음과 같다.

$$
C_l^{\mathrm{magnitude}}=p_l\|g_l\|_2.
$$

전체 combined gradient 방향으로 투영한 signed contribution은 다음과 같다.

$$
C_l^{\mathrm{signed}}=
p_l\|g_l\|_2\cos(g_l,G).
$$

| W&B key | 의미 |
|---|---|
| `grad_contribution_magnitude_actor/<layout>` | actor의 $C_l^{\mathrm{magnitude}}$ |
| `grad_contribution_magnitude_critic/<layout>` | value의 $C_l^{\mathrm{magnitude}}$ |
| `grad_contribution_signed_actor/<layout>` | actor의 $C_l^{\mathrm{signed}}$ |
| `grad_contribution_signed_critic/<layout>` | value의 $C_l^{\mathrm{signed}}$ |

`magnitude`는 항상 0 이상이다. `signed`가 음수이면 해당 layout gradient가
전체 combined gradient 방향을 방해한다.

### Layout 간 conflict

$$
\cos(g_i,g_j)=
\frac{g_i^\top g_j}{\|g_i\|_2\|g_j\|_2+\epsilon}.
$$

| W&B key | 의미 |
|---|---|
| `grad_conflict_actor/<layout_i>_vs_<layout_j>` | actor layout gradient cosine |
| `grad_conflict_value/<layout_i>_vs_<layout_j>` | value layout gradient cosine |

Layout이 해당 rollout에 존재하지 않으면 관련 layout metric은
`NaN`이다.

## Environment-sample gradient conflict

하나의 sample은 한 environment slot의 앞쪽
`GRAD_CONFLICT_WINDOW_STEPS` trajectory이며 두 actor를 함께 포함한다. Episode
reset을 통과하면 `done`이 recurrent state를 reset한다.
Actor와 value gradient는 별도로 측정한다.

정규화된 gradient를 다음과 같이 둔다.

$$
u_i=\frac{g_i}{\|g_i\|_2+\epsilon}.
$$

### 전체 pair 평균 cosine

```text
grad_conflict_sample_actor/avg_pairwise_cosine
grad_conflict_sample_value/avg_pairwise_cosine
```

$$
\overline{C}=
\frac{1}{N(N-1)}\sum_{i\ne j}u_i^\top u_j.
$$

다음 항등식을 사용하므로 full gradient matrix를 저장하지 않고도 모든 pair를
정확히 반영한다.

$$
\sum_{i\ne j}u_i^\top u_j
=\left\|\sum_i u_i\right\|_2^2-\sum_i\|u_i\|_2^2.
$$

### Conflict rate와 negative cosine

```text
grad_conflict_sample_actor/conflict_rate
grad_conflict_sample_value/conflict_rate
grad_conflict_sample_actor/avg_negative_cosine
grad_conflict_sample_value/avg_negative_cosine
```

매 diagnostic update마다 environment를 무작위로 섞고 인접한 두 environment를
묶어 random perfect matching $\mathcal M$을 만든다. `NUM_ENVS=256`이면
128개의 disjoint pair를 사용한다.

$$
\operatorname{conflict\_rate}=
\frac{1}{|\mathcal M|}\sum_{(i,j)\in\mathcal M}
\mathbf 1(u_i^\top u_j<0).
$$

$$
\operatorname{avg\_negative\_cosine}=
\frac{1}{|\mathcal M|}\sum_{(i,j)\in\mathcal M}
\max(0,-u_i^\top u_j).
$$

`avg_negative_cosine`은 conflict pair만을 조건으로 한 평균이 아니다. 전체
matched pair에 대한 negative part의 평균이므로 conflict 빈도와 강도를 모두
반영한다. 두 지표는 전체 $\binom{N}{2}$ pair의 정확한 값이 아니라 매
update의 random matching estimator이다.

### Sample alignment

```text
grad_conflict_sample_actor/alignment
grad_conflict_sample_value/alignment
```

$$
\operatorname{alignment}=
\frac{\|\sum_i g_i\|_2^2}
{N\sum_i\|g_i\|_2^2+\epsilon}.
$$

값이 1에 가까우면 environment gradient들이 집단적으로 같은 방향이고, 0에
가까우면 상쇄되거나 서로 다른 방향이다. 이 값은 random matching이 아니라
전체 $N$개 gradient를 반영한다.

## Representation weight metrics

현재 weight metric은 kernel뿐 아니라 bias와 learned 1D scale/shift를 포함한
모든 parameter leaf를 사용한다. SimBaV2의 update-level parameter norm과
맞추기 위해 각 PPO minibatch optimizer step 직전 parameter에서 계산한다.
먼저 한 PPO update의 epoch/minibatch 전체를 평균하고, W&B에는 다시 logging
interval 동안의 update 평균을 기록한다. 따라서 이 지표는 diagnostic snapshot이
아니다.

$$
\|\theta\|_2=
\sqrt{\sum_k\sum_j\theta_{k,j}^2}.
$$

$$
\operatorname{WeightedRMS}(\theta)=
\sqrt{
\frac{\sum_k n_k\|\theta_k\|_2^2}
{\sum_k n_k}
}.
$$

| W&B key | 포함 범위 |
|---|---|
| `representation_weight/weight_norm` | 전체 parameter tree의 global L2 norm |
| `representation_weight/actor_weight_norm` | shared와 actor branch/output의 global L2 norm |
| `representation_weight/critic_weight_norm` | shared와 critic branch/output의 global L2 norm |
| `representation_weight/shared_weight_norm` | shared encoder/RNN parameter의 global L2 norm |
| `representation_weight/weighted_rms_norm` | 전체 parameter tree의 weighted RMS |
| `representation_weight/actor_weighted_rms_norm` | shared와 actor branch/output |
| `representation_weight/critic_weighted_rms_norm` | shared와 critic branch/output |
| `representation_weight/shared_weighted_rms_norm` | shared encoder/RNN parameter의 weighted RMS |

PopArt는 output-preserving rescaling을 critic output layer에 적용하므로 일반
CEC와 PopArt CEC의 `critic_weight_norm`을 직접 비교할 때 주의해야 한다.

## Gradient kurtosis

각 PPO minibatch의 gradient clipping 및 Adam 적용 전 raw gradient 원소
$G_i$를 다음과 같이 변환한다.

$$
L_i=\log(|G_i|+\epsilon),\qquad \epsilon=10^{-8}.
$$

기록하는 값은 excess kurtosis가 아닌 Pearson kurtosis다.

$$
K=
\frac{\mathbb{E}[(L_i-\mu_L)^4]}
{\left(\mathbb{E}[(L_i-\mu_L)^2]\right)^2+10^{-12}}.
$$

```text
gradient_kurtosis/global
gradient_kurtosis/actor
gradient_kurtosis/critic
gradient_kurtosis/shared
```

Actor와 critic은 각각 shared trunk와 해당 branch/output을 포함한다. Shared는
`Conv_0`, `Conv_1`, `Dense_0`, `Dense_1`, `ScannedRNN_0`만 포함한다.
값이 클수록 log-absolute gradient 분포의 tail이 무겁거나 극단적인 gradient
원소가 존재한다는 의미다. 다른 optimizer-update 지표와 마찬가지로 interval
동안 수행된 PPO minibatch들의 평균을 W&B에 기록한다.

## Penultimate feature metrics

Shared recurrent trunk와 actor/critic의 마지막 output layer 직전 activation으로
$\Phi$를 각각 구성한다.
Feature를 mean-center하지 않으므로 아래 spectrum은 centered covariance가 아니라
uncentered feature matrix의 spectrum이다. SimBaV2의 별도 metric replay batch에
대응하여, CEC에서는 diagnostic logging update에서 첫 PPO epoch의 첫 minibatch가
사용할 actor permutation을 동일한 RNG로 재현한다. 해당 minibatch의 전체
`NUM_STEPS` trajectory와 actor별 초기 LSTM state를 사용하므로 episode reset과
recurrent history가 실제 PPO update와 일치한다. Representation은 첫 minibatch
update 직전 parameter에서 별도 diagnostic forward로 한 번만 계산하며 학습
gradient에는 포함되지 않는다.

같은 environment의 두 actor를 평균하거나 concatenate하지 않고 각 actor-time
feature를 독립적인 matrix row로 사용한다. Feature matrix의 sample 수는

$$
M=TA_m
$$

이다. 기본값에서는 $M=400\times256=102{,}400$다. 이 값들은 interval 평균이
아니라 해당 첫 epoch·첫 minibatch의 pre-update snapshot이다.

### Feature norm

$$
\operatorname{feature\_norm}=
\frac{1}{M}\sum_{m=1}^{M}\|\phi_m\|_2.
$$

```text
representation_feature/shared_feature_norm
representation_feature/actor_feature_norm
representation_feature/critic_feature_norm
```

### Singular-value scale와 concentration

$$
\texttt{normalized\_sigma\_1}=\frac{\sigma_1(\Phi)}{\sqrt M},
\qquad
\texttt{sigma\_1\_ratio}=
\frac{\sigma_1}{\sum_j\sigma_j}.
$$

```text
representation_feature/shared_normalized_sigma_1
representation_feature/actor_normalized_sigma_1
representation_feature/critic_normalized_sigma_1
representation_feature/shared_sigma_1_ratio
representation_feature/actor_sigma_1_ratio
representation_feature/critic_sigma_1_ratio
```

`normalized_sigma_1`의 제곱 $\sigma_1^2/M$이
$\Phi^\top\Phi/M$의 최대 eigenvalue다. 현재 feature는 centering하지 않으므로
이는 covariance가 아니라 uncentered second-moment matrix의 eigenvalue다.

### Intermediate path-total feature norm

각 hidden layer $q$에서 sample별 feature L2 norm의 평균을 먼저 계산한다.

$$
F_q=\frac{1}{M}\sum_m\|h_m^{(q)}\|_2.
$$

Shared total은 shared path의 layer norm들을 단순 합산한다.

$$
F_{\mathrm{shared,total}}=
\sum_{q\in\mathrm{shared}}F_q.
$$

Actor와 critic total은 이 shared total과 해당 branch의 layer norm들을
단순 합산한다.

$$
F_{\mathrm{actor,total}}=
\sum_{q\in\mathrm{shared}}F_q+
\sum_{q\in\mathrm{actor}}F_q,
$$

$$
F_{\mathrm{critic,total}}=
\sum_{q\in\mathrm{shared}}F_q+
\sum_{q\in\mathrm{critic}}F_q.
$$

```text
representation_feature_total/shared_feature_norm
representation_feature_total/actor_feature_norm
representation_feature_total/critic_feature_norm
```

이 값은 하나의 penultimate feature norm이 아니라 여러 layer norm의 합이므로
network depth와 architecture가 다른 모델 사이의 직접 비교에는 적합하지 않다.

## Representation rank metrics

기본 cutoff는 $c=0.01$, threshold는 $1-c=0.99$다.

### Lyle feature rank

$$
\operatorname{feature\_rank}=
\sum_j\mathbf 1\left(\frac{\sigma_j}{\sqrt M}>c\right).
$$

```text
representation_rank/shared_feature_rank
representation_rank/actor_feature_rank
representation_rank/critic_feature_rank
```

이 rank는 absolute feature scale에 영향을 받는다.

### Roy--Vetterli effective rank

$$
p_j=\frac{\sigma_j}{\sum_k\sigma_k},
\qquad
\operatorname{effective\_rank}=
\exp\left(-\sum_jp_j\log p_j\right).
$$

```text
representation_rank/shared_effective_rank_vetterli
representation_rank/actor_effective_rank_vetterli
representation_rank/critic_effective_rank_vetterli
```

Spectrum이 여러 방향에 균등하게 퍼질수록 커지고 한두 방향에 집중될수록
작아진다. Uniform feature rescaling에는 불변이다.

### Kumar srank

$$
\operatorname{srank}_{c}=
\min\left\{k:
\frac{\sum_{j=1}^{k}\sigma_j}{\sum_j\sigma_j}
\ge 1-c
\right\}.
$$

```text
representation_rank/shared_srank_kumar
representation_rank/actor_srank_kumar
representation_rank/critic_srank_kumar
```

### PCA approximate rank

$$
\operatorname{rank}_{\mathrm{PCA},c}=
\min\left\{k:
\frac{\sum_{j=1}^{k}\sigma_j^2}{\sum_j\sigma_j^2}
\ge 1-c
\right\}.
$$

```text
representation_rank/shared_approximate_rank_pca
representation_rank/actor_approximate_rank_pca
representation_rank/critic_approximate_rank_pca
```

### Numerical matrix rank

$$
\tau=\max(M,D)\,\epsilon_{\mathrm{machine}}\,\sigma_1,
\qquad
\operatorname{matrix\_rank}=
\sum_j\mathbf 1(\sigma_j>\tau).
$$

```text
representation_rank/shared_matrix_rank
representation_rank/actor_matrix_rank
representation_rank/critic_matrix_rank
```

## Evaluation과 return

| W&B key | 의미 |
|---|---|
| `eval/mean` | 고정 evaluation layout 평균 return |
| `eval/<layout>` | layout별 evaluation return |
| `eval_xp/mean` | human-proxy cross-play 평균 return |
| `eval_xp/<layout>` | layout별 human-proxy cross-play return |
| `train_returns/<layout>` | logging interval에 종료된 해당 layout episode return 평균 |

## 참고 문헌

- [Roy & Vetterli, *The Effective Rank: A Measure of Effective
  Dimensionality*](https://zenodo.org/records/40328) (2007): entropy-based
  effective rank.
- [Kumar et al., Implicit Under-Parameterization Inhibits Data-Efficient Deep
  Reinforcement Learning](https://arxiv.org/abs/2010.14498): threshold srank.
- [Yang et al., Harnessing Structures for Value-Based Planning and
  Reinforcement Learning](https://arxiv.org/abs/1909.12255): PCA approximate
  rank.
- [Lyle et al., Understanding and Preventing Capacity Loss in Reinforcement
  Learning](https://arxiv.org/abs/2204.09560): threshold feature rank.
- [Lyle et al., Understanding Plasticity in Neural
  Networks](https://arxiv.org/abs/2303.01486): per-transition gradient
  covariance/cosine analysis.
