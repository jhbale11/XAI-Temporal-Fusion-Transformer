# XAI-Temporal-Fusion-Transformer
@jhbale11
- Interpretable Time Series Forecasting에서 SOTA(State Of Art)인 TFT에 대한 논문 리뷰와 코드 구현입니다.
- 질문 사항은 본 저장소의 Issues 탭에 남겨주시길 바랍니다.

#### Paper : [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting(Google Research,2020)](https://arxiv.org/pdf/1912.09363.pdf)
#### Review : [jhbale11's Velog](https://velog.io/@jhbale11/Temporal-Fusion-Transformer2020%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)

# 0. Abstract

여러 날을 예측하는 Multi-Horizon Forecasting 시계열 데이터는 복잡한 입력들을 포함하는데 그러한 것들에는 시간에 따라 변하지 않는 변수(이 논문에서는 Static Covariates라 표현), 시간에 따라 변하는 관측가능한 변수(Time Varying Observable Input), 타겟과 어떠한 상호작용을 하는지 사전 정보가 없는 외인성 변수(Exogenous Variable) 등 이 존재한다.

현재까지 여러 딥러닝 모델들이 제안되었지만, 그들은 대표적인 Black Box 모델들로 실제 상황에서 전체 길이의 입력들이 어떻게 사용되는지, 모델은 어떠한 Time Step을 중요하게 고려하고 있는지에 대한 통찰을 제공하지는 못한다. 따라서 논문에서는 Attention 기반 구조와 해석 가능한 인사이트를 제공할 수 있는 TFT(Temporal Fusion Transformer)을 제안한다.

# 1. Introduction

미래의 여러 Time Step에 대해서 예측을 진행하는 Multi-Horizon Forecasting은 시계열 예측 분야에서 매우 중요한 문제이다. 한 스텝 후만 예측하는 것과 달리 Multi-Horizon은 전체 시계열 데이터에 대한 접근을 가능하게 하고, 미래의 다양한 스텝들에 대한 예측 결과를 통해 활용할 수 있는 범위도 넓어진다. 미래에 대한 예측은 유통업, 헬스 케어, 금융 분야에서 중요하게 사용되고 있으며 시계열 예측의 성능 향상은 엄청난 잠재력을 가지고 있다.

![source : https://arxiv.org/pdf/1912.09363.pdf](https://media.vlpt.us/images/jhbale11/post/ac2b49ac-13a8-4a2b-b076-5051910c7dbf/TFT1.png)

실제 Multi-Horizon Forecasting은 위 그림과 같이 다양한 데이터 Input이 필요하다.

(1) 현재에는 관측을 통해서 그 값을 알 수 있지만 미래의 값은 알 수 없는 Observed Input
(2) 시간에 따라 달라지지만, 현재에도 그 값을 알 수 있으며 미래에도 그 값을 알 수 있는 Time Vary Known Input(week, weekofyear, holiday, quarter)
(3) 시간과 관계 없이 변하지 않는 정적 공변량 Static Covariates(ex 상점의 위치)

과거에 Autoregressive Model이 사용되었으나, 이는 모든 외인성 입력들이 미래에도 알 수 있다는 가정을 하고 있다는 문제가 있었으며, 많은 모델이 Time-Dependent Feature 들과 단순하게 결합하는 방식으로 동적인 Covariates를 무시했던 등의 이유로 Multi-Horizon의 다양한 종류의 입력들을 고려하는데 실패하였다. 최근 많은 개선된 모델들이 데이터의 고유한 특성을 구조적으로 정렬함으로써 좋은 성과를 얻어냈다. 현재 사용되는 대부분은 모델들은 'Black Box'모델들로 예측이 많은 파라미터 사이에 복잡한 비선형적인 상호작용에 의해 결정된다.

모델이 한 예측에 대해서 왜 그런 결과가 나왔는지 이유를 제공해줄 수 없는 이러한 Black Box 모델은 유저들이 모델의 출력을 신뢰하거나 개발자가 모델의 구조를 디버깅하기 힘들게 한다. 그리고 대부분은 DNN 모델들은 시계열에 적합하지 않다.

(1) Convolution 기반의 Post-Hoc 기법인 LIME과 SHAP은 Input Feature의 시간 순서를 고려하지 않느다. LIME은 매 데이터 포인트마다 독립적으로 모델ㅇ들이 만들어지기에 매우 비효율적이며, SHAP의 경우 Feature들이 근접한 Time Step과 독립적으로 고려된다는 점에서 시계열에 적합하지 않다. 이러한 Post-Hoc 접근들은 시계열 데이터에서 중요한 시간대에 따른 상관관계를 설명하는 능력이 매우 부족하다.

(2) 반면 Language나 Speech에서 주로 사용되는 Attention 기반의 모델들은 Sequence 데이터에 대한 해석력이 매우 뛰어나다. Language나 Speech와 달리 시계열 데이터는 다양한 Feature를 가지고 있다는 것이 특징인데, 이러한 방법론을 시계열 데이터에 적용했을 때 중요한 Time Step을 발견할 수는 있지만 Feature 간의 관계에서 어떤 Feature가 중요하게 고려되고 있는지, 주어진 Time Step에 따른 Feature들의 각각의 중요성을 구분하기는 힘들다. 따라서 새로운 기법은 이러한 예측이 해석가능하게 하는 것이 필수적이다.

Attention 기반의 DNN 구조인 TFT는 새로운 해석력을 제공한다. Attention Score를 통해 Time Step에서의 중요성을 해석할 수 있음은 물론, Static Variable에 대하여, Encoder Variable에 대하여, Decoder Variable에 대해서 Interpretation을 제공한다는 점에서 큰 장점을 가진다.

# 2. Related Work

Autoregressive 모델들과 Seq2Seq 모델로 나뉘며 DeepAR과 같은 LSTM Network나 DSSM(Deep State Space Model) 등이 존재한다. 최근에는 Transformer 기반의 구조들이 예측 시 receptive 영역을 늘리기 위해 지역적인 처리를 하는 convolution이나 sparse attention을 같이 사용하고 있다. (Conv-Transforemr, Informer)

과거 사용하던 대부분의 모델은 여러 Horizon에 대한 값을 한번에 출력하도록 훈련된다. 이러한 모델들은 일반적으로 Seq2Seq 모델을 기반으로 하며, LSTM 인코더를 이용하여 과거의 입력을 요약하고 미래의 예측을 다양한 기법을 사용하여 만들어냈다. MQRNN(Multi-Horizon Quantile Recurrent Forecaster)는 LSTM이나 Conv 인코더를 이요하여 Context Vector를 만들고 이를 각각의 Horizon에 대하여 MLP에 feed하였다. Multi-Model Attention 기법은 LSTM 인코더를 이용하여 Context Vector를 만들고 Bi-directional LSTM을 디코더로 사용하여 이전 기법들에 비해서 좋은 성능을 가지고 있지만 해석력은 여전히 문제가 되었다.

TFT는 이러한 문제를 정적 Feature들에 대하여 개별적인 Encoder-Decoder Attention을 사용함으로 해결하고자 하였다.

# 3. Multi-Horizon Forecasting

주어진 시계열 데이터셋에 대하여 고유한 객체 I가 존재한다고 하자. 각 객체 i는 정적 공변량의 집합과 연관되어 있다. (상점의 경우 I는 고객, 정적 공변량은 상점의 위치가 될 것이다) 시간에 의존적인 입력 Feature들은 두 개로 나뉘는데 하나는 Observed Input이고 나머지는 Known Input이다. (위의 핵심 아이디어 부분에서 설명한 바와 같다) 여러 상황에서 TFT는 Quantile Regression을 사용하여 예측을 진행한다. 각 Quantile 예측은 다음과 같은 형태를 지닌다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FThwhw%2Fbtq4Sl6dy9P%2FKTNMQnGZIs54FisGE5PM50%2Fimg.png)

이 때, y는 q번째 Sample Quantile에 대하여 t 시점에서 타우 번째 뒤에 해당하는 예측에 해당하며, f는 예측 모델이다. k의 Look-back Window를 이요하여 시작하는 지점인 t로부터 k개 이전까지의 값들을 이용하는 식이라고 해석할 수 있다. x는 Known Input, s는 Static Covariate의 집합을 의미한다.

# 4. Model Architecture

![source : https://arxiv.org/pdf/1912.09363.pdf](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbX08Yv%2Fbtq4OsSUmLc%2FwfrgYLSkHYMlQQjvrhSeWK%2Fimg.png)

TFT의 주요 구성은 아래와 같다.

(1) Gating Mechanism을 통해 불필요한 성분을 스킵하여 광범위한 데이터셋에 대하여 adaptive depth와 network complexity 감소를 통해 가능하게 한다.

(2) Variable Selection Network를 통해 관련 있는 Input Variable만 선택한다.

(3) Static Covariate Encoder를 통해 정적 공변량들을 Context Vector에 인코딩하고 네트워크에 결합한다.

(4) Temporal Processing을 통해 Observed Input과 Known Input 모두에 대해 장기 단기 시간 관계를 학습한다. Local Processing을 위해 seq2seq layer를 사용하며, Interpretable Multi-Head Attention을 통해 장기 의존성을 알아낸다.

(5) Prediction Intervals을 통해 Quantile을 이용하여 매 Prediction Horizon에 대하여 Target이 존재할 수 있는 범위를 제공한다.

### 4.1 Gating Mechanisms
### 4.2 Variable Selection Networks
### 4.3 Static Covariate Encoder
### 4.4 Interpretable Multi-Head Attention
### 4.5 Temporal Fusion Decoder
### 4.6 Quantile Outputs

# Experiments
#### (1) Volatility
- Purpose : 31가지 Daily Index 데이터를 사용, 지난 252일(Intradays)의 데이터를 통해 다음주(5 Business Day)의 Volatility에 대한 예측 진행
- Duration : 2000-01-03 ~ 2019-06-28까지 데이터 사용
- Data Split : 2016년까지의 데이터를 Train, 2016-2017년 데이터를 Validation, 2018년 이후의 데이터를 Test Set으로 사용
- Regime Detection : S&P 500 Index의 Attention Vector에 Distance Metrics를 적용하여 Significant Regime Detection 이후, 해당 Regime에서의 Attention Value 값이 높음을 통해 Interpretability 검증

#### (2) Features
- Time Index : Dates
- Target : log of volatility(log of 5min sub-sampled realized volatility[rv5_ss])
- Observed Input : open to close(daily open-to-close returns)
- Time Varying Known Input
  - days_From_Start
  - day_of_week
  - day_of_month
  - week_of_year
  - month

#### (3) Result
![source : https://arxiv.org/pdf/1912.09363.pdf](https://media.vlpt.us/images/jhbale11/post/8f319bd0-8efa-48a4-af24-2ba07ba4fc64/1-s2.0-S0169207021000637-gr5.jpg)
Static Covairates : Region
