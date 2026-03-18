# CrossEnv UED - 프로젝트 구조

> **주제**: Cross-Environment Cooperation Enables Zero-shot Multi-agent Coordination  
> **기반**: [JaxMARL](https://github.com/FLAIROx/JaxMARL) (JAX 기반 다중에이전트 강화학습)

---

## 📁 최상위 디렉터리 구조

```
crossenv_ued/
├── baselines/          ← 다양한 강화학습 알고리즘 구현
├── jaxmarl/            ← 핵심 환경(Environment) 코드
├── requirements/       ← 의존성 패키지 목록
├── runpod/             ← 클라우드 배포 설정 (Docker, 배포 스크립트)
├── tutorials/          ← 학습용 노트북 및 예제
├── README.md           ← 프로젝트 개요
└── pyproject.toml      ← Python 패키지 설정
```

---

## 🤖 baselines/ - 강화학습 알고리즘들

### 1️⃣ **IPPO/** - Independent PPO (기본 단일환경 학습)

**목적**: 각 에이전트가 독립적으로 PPO 알고리즘 실행

**특징**:
- 각 에이전트의 정책 파라미터가 공유되지 않음
- 에이전트 간 협력이 명시적이지 않음

**주요 파일**:
| 파일 | 환경 | 설명 |
|------|------|------|
| `ippo_ff_hanabi.py` | Hanabi | Feed-Forward 네트워크 기반 |
| `ippo_rnn_hanabi.py` | Hanabi | LSTM RNN 기반 |
| `ippo_cnn_overcooked.py` | Overcooked | CNN 기반 (시각 입력) |
| `ippo_ff_mpe.py` | MPE | Multi-agent Particle Environment |
| `ippo_ff_switch_riddle.py` | Switch Riddle | 간단한 협력 게임 |

---

### 2️⃣ **MAPPO/** - Multi-Agent PPO (공유 정책)

**목적**: 모든 에이전트가 **하나의 공유 네트워크**로 학습

**특징**:
- 정책 파라미터 공유 → 에이전트 간 협력 유도
- 중앙 집중식 학습 접근
- Global "world_state" 관측 사용

**주요 파일**:
- `mappo_rnn_hanabi.py`
- `mappo_rnn_mpe.py`
- `mappo_rnn_smax.py`

---

### 3️⃣ **QLearning/** - Q-Learning 계열 (가치 기반)

**목적**: 에이전트들이 Q값을 통해 최적 정책 학습

**알고리즘들**:
| 알고리즘 | 설명 |
|---------|------|
| **IQL** | Independent Q-Learners (개별 학습) |
| **VDN** | Value Decomposition Network (가치 분해) |
| **QMIX** | Q-값 선형결합을 통한 협력 |
| **TransfQMix** | Transformer 활용 QMIX |
| **SHAQ** | Shapley 가치 이론 적용 |

**특징**:
- RNN 기반 에이전트
- 파라미터 공유 선택 가능
- Experience Replay 사용
- Double Q-Learning (안정성 향상)

---

### 4️⃣ **CEC/** - Cross-Environment Cooperation (논문 핵심)

**목적**: 論文의 핵심 주장 증명
- **단일 에이전트** + **다양한 절차 생성 환경**에서 훈련
- **vs 다중 에이전트** + **단일 고정 환경** (IPPO, MAPPO 등)
- 결과: CEC가 훨씬 더 우수한 일반화 성능 달성

**두 가지 핵심 개념**:

#### 🎓 훈련 (Training)
- **절차 생성(Procedurally Generated) 환경**: 수백~수천 가지의 **임의 생성 맵**에서 훈련
- `random_reset_fn` 사용하여 매 에피소드마다 새로운 맵 생성
- 에이전트 **2명과 함께** 학습 (하나의 에이전트 정책으로 학습)

#### ✅ 평가 (Evaluation - 5가지 Hand-crafted 맵)
- 다른 알고리즘과의 공정한 비교를 위해 동일한 5가지 고정 맵에서 평가
- 5가지 맵:
  ```
  make_asymm_advantages_9x9()      # 비대칭 이점
  make_coord_ring_9x9()            # 환형 조정
  make_counter_circuit_9x9()       # 카운터 회로
  make_forced_coord_9x9()          # 강제 조정
  make_cramped_room_9x9()          # 비좁은 방
  ```

**주요 파일 및 역할**:

| 파일 | 역할 | 논문 표기 |
|------|------|---------|
| `ippo_general.py` | CEC 기본 IPPO | **CEC** |
| `ippo_general_population.py` | CEC + Population-based fine-tuning | **CEC-FT** |
| `cross_algo.py` | 다중환경 평가 로직 | 평가 유틸 |
| `e3t.py` | 앙상블 환경 전략 | **E3T** |
| `fcp_general.py` | 초점 크로스훈련 전략 | **FCP** |
| `actor_networks.py` | 신경망 아키텍처 (LSTM, GAT, GCN) | 공통 |

**테스트 파일**:
- `test_general.py` (24KB) - CEC 성능 평가
- `test_e3t.py` (28KB) - E3T 성능 평가
- `test_oracle.py` (26KB) - 오라클(최적) 모델 벤치마크
- `test_all_models_cross.py` (17KB) - 모든 모델 비교 (IPPO, MAPPO vs CEC, E3T, FCP)

---

### 5️⃣ **CEC_UED/** - CEC + UED (자동 환경 생성)

**목적**: 
- CEC의 개념을 **Unsupervised Environment Design (UED)**로 확장
- **절차 생성 프로그래밍 대신 VAE로 자동 생성**
- Hand-crafted 환경 완전 제거 + CEC와 유사 성능

**구조**:
```
CEC_UED/
├── ippo_general.py         ← UED 프레임워크 내 훈련 루프
├── ippo_general_vae.py     ← VAE 기반 환경 생성기
├── config/                 ← 설정 파일
├── shell/                  ← 실행 스크립트
└── VAE/                    ← VAE 신경망 구현
```

**학습 흐름**:
```
1. VAE로부터 새로운 환경 샘플링
   ↓
2. 에이전트 훈련 (그 환경에서)
   ↓
3. 성능 평가 (에이전트 개선 정도 측정)
   ↓
4. 우수한 환경만 보관 (고난이도 선별)
   ↓
5. VAE 재훈련 (우수 환경 분포 학습)
   ↓
6. 1번으로 돌아가서 반복
```

**논문의 결론**: 
- 🎯 **CEC_UED ≈ CEC** (손공작 환경 vs 자동생성 환경)
- ✅ 둘 다 기준선(IPPO, MAPPO)보다 **현저히 우수**
- 🚀 **Hand-crafted 환경의 한계 극복**

---

## 🌍 jaxmarl/ - 환경 및 기반 코드

**역할**: 다중에이전트 환경 정의 및 고성능 JAX 시뮬레이션

**주요 구조**:
```
jaxmarl/
├── environments/          ← 게임 환경들
│   ├── overcooked/       ← 요리사 협력 게임 (주요 환경)
│   ├── mpe/              ← Multi-agent Particle Environment
│   ├── hanabi/           ← 카드 게임
│   ├── smax/             ← 전략 게임
│   ├── coin_game/        ← 동전 수집
│   └── ...
├── wrappers/             ← 환경 래퍼 (로깅, 변환 등)
├── gridworld/            ← 그리드 월드
├── viz/                  ← 시각화 및 렌더링
└── tutorials/            ← 학습 가이드
```

**핵심 환경**:

| 환경 | 설명 | 특징 |
|------|------|------|
| **Overcooked** | 🍳 요리사 2명의 협력 | CEC의 주요 환경, 복잡한 레이아웃 |
| **ToyCoop** | 🎮 간단한 협력 게임 | CEC_UED 테스트용 |
| **MPE** | 🚀 입자 이동 환경 | 연속 제어, 통신 가능 |
| **Hanabi** | 🎴 카드 게임 | 부분관측, 협력 필수 |
| **SMAX** | ⚔️ 전략 게임 | 격자 환경, 전술 게임 |

---

## 📊 비교: 알고리즘별 학습 방식

```
┌────────────┬──────────┬────────────┬──────────────────┬────────────────┐
│ 알고리즘   │ 정책공유 │ 협력방식   │ 환경 수           │ 특징           │
├────────────┼──────────┼────────────┼──────────────────┼────────────────┤
│ IPPO       │ ✗ 없음   │ 암시적     │ 1 (고정 맵)      │ 기준선 (약함)   │
│ MAPPO      │ ✓ 있음   │ 공유       │ 1 (고정 맵)      │ 기준선 (약함)   │
│ Q-Learn    │ 선택     │ 가치분해   │ 1 (고정 맵)      │ 기준선 (약함)   │
│ CEC        │ ✗ 없음   │ 암시적     │ ~100-1000s       │ 절차생성 환경   │
│ CEC (E3T)  │ ✗ 없음   │ 앙상블     │ ~100-1000s       │ 앙상블 전략     │
│ CEC (FCP)  │ ✗ 없음   │ 초점       │ ~100-1000s       │ 초점 전략       │
│ CEC_UED    │ ✗ 없음   │ 암시적     │ VAE 생성 (자동)  │ 환경 진화 학습  │
└────────────┴──────────┴────────────┴──────────────────┴────────────────┘
```

**핵심 차이**: 
- 기준선: **1개 맵** (고정) + **2명 에이전트**
- **CEC**: **수백~수천 맵** (절차생성) + **1명 에이전트** = 우수한 일반화

---

## 🚀 runpod/ - 클라우드 배포

**목적**: GPU 클라우드 플랫폼(Runpod)에서 훈련

**구조**:
```
runpod/
├── Dockerfile          ← 컨테이너 이미지 정의
├── deploy.py           ← 배포 스크립트
├── README.md           ← 배포 가이드
└── config/
    └── train_ep20_bs512.yaml  ← 클라우드 훈련 설정
```

**사용 목적**:
- 대규모 병렬 훈련
- GPU 무제한 사용
- 데이터 센터 환경

---

## 📚 tutorials/ - 학습 자료

| 파일 | 설명 |
|------|------|
| `JaxMARL_Walkthrough.ipynb` | JaxMARL 기초 튜토리얼 |
| `mpe_introduction.py` | MPE 환경 사용법 |
| `overcooked_introduction.py` | Overcooked 환경 사용법 |
| `smax_introduction.py` | SMAX 환경 사용법 |
| `storm_introduction.py` | Storm 환경 사용법 |

---

## 🔍 논문 핵심 실험 흐름 & 주요 발견

```
1️⃣ 기준선 수립 (Baselines)
   ├─ IPPO (IPPO/) - 2에이전트, 단일 맵 - 약한 일반화
   ├─ MAPPO (MAPPO/) - 2에이전트, 단일 맵 - 약한 일반화
   └─ Q-Learning (QLearning/) - 2에이전트, 단일 맵

2️⃣ CEC 핵심 아이디어
   ├─ IPPO (단일 에이전트)
   ├─ 절차 생성 환경 (수백~수천 개 맵)  ← 훈련
   └─ 5가지 hand-crafted 맵에서 평가 ← 테스트
   
   결과: 단일 에이전트 + 다양한 환경
        > 다중 에이전트 + 단일 환경 ✅

3️⃣ CEC 개선 기법들 (CEC/)
   ├─ E3T (e3t.py) - 앙상블 환경 전략
   ├─ FCP (fcp_general.py) - 초점 크로스훈련
   └─ CEC-FT (ippo_general_population.py) - Population-based fine-tuning

4️⃣ CEC_UED 확장
   ├─ CEC + UED 결합
   ├─ VAE로 자동 생성된 환경 사용
   └─ 수동 설계 불필요 + CEC와 유사한 성능
```

**주요 발견**: 
- 🎯 **절차 생성 다양한 환경** = 충분한 도메인 무작위화
- 👥 **에이전트 수 << 환경 다양성** (중요도 역전)
- 🚀 Hand-crafted 한계를 벗어남

---

## 📂 실행 방법

### 기본 훈련 (CEC)
```bash
python baselines/CEC/ippo_general.py
```

### CEC_UED 훈련
```bash
python baselines/CEC_UED/ippo_general.py
```

### 테스트
```bash
python baselines/CEC/test_general.py
python baselines/CEC/test_e3t.py
```

---

## 🎯 폴더별 역할 요약

| 폴더 | 역할 | 훈련 방식 | 핵심 |
|------|------|---------|------|
| **IPPO** | 기준선 (약함) | 2 에이전트, 1 맵 | 기본 구현 |
| **MAPPO** | 기준선 (약함) | 2 에이전트, 1 맵 | 정책 공유 |
| **QLearning** | 기준선 (약함) | 2 에이전트, 1 맵 | 가치 기반 |
| **CEC** | 🏆 논문 핵심 | 1 에이전트, **절차생성 수백~수천 맵** | 환경 다양성 > 에이전트 수 |
| **CEC_UED** | 논문 확장 | 1 에이전트, **VAE 생성 환경** | 자동화 + 진화 선별 |

---

## 💾 저장 위치 및 구성

- **코드**: `baselines/{ALGORITHM}/{implementation}.py`
- **설정**: `baselines/{ALGORITHM}/config/*.yaml`
- **체크포인트**: `baselines/{ALGORITHM}/checkpoints/` (훈련 후 생성)
- **환경 코드**: `jaxmarl/environments/`
- **시각화**: `jaxmarl/viz/`

---

**마지막 업데이트**: 2026-03-11  
**프로젝트**: CrossEnv UED (Cross-Environment Cooperation + Unsupervised Environment Design)
