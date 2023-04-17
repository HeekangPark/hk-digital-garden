---
{"title":"Likelihood","date_created":"2023-01-26","date_modified":"2023-01-29","dg-publish":true,"dg-path":"/nlp/likelihood.md","permalink":"//nlp/likelihood/","dgPassFrontmatter":true,"created":"","updated":""}
---


## 모델링 문제

- 고등과정까지 우린 "식 → 값" 형태의 문제를 많이 풂
    - 변수들 간의 관계식이 주어졌을 때 특정 변수의 값을 구하는 형태
    - ex) 두 변수 $x$, $y$ 사이의 관계식이 $y = 3x + 5$라 주어졌을 때, $x=3$에서 $y$의 값은?

- 하지만 우린 때때로 "값 → 식" 형태의 문제를 풀어야 함 : 모델링(modeling) 문제
    - 변수들의 값(데이터)이 주어졌을 때 이로부터 변수들 간의 관계식을 구하는 형태
    - ex) 매 시간 $t$마다 물체의 위치 $d$를 관측했더니 다음과 같은 데이터를 얻었을 때, $t$와 $d$ 사이의 관계식은?

        | $t$ | $d$ |
        |:---:|:---:|
        |  0  |  0  |
        |  1  |  1  |
        |  2  |  3  |
        |  3  |  6  |
        |  4  | 10  |

- 일반적으로 모델링 문제는 다음 순서로 풂
    1. 관계식의 대략적인 형태를 가정하고
        - ex) 선형 함수, 지수 함수, 정규 분포 등
        - 이 가정이 잘못되면 안 좋은 model이 된다. 
    2. 데이터를 가장 잘 설명할 수 있는 관계식의 파라미터들을 찾는다.
        - ex) 최소제곱법(MSE) : 오차를 최소화하는(정확히는 오차의 제곱값을 최소화하는) 파라미터를 찾는 방법

## 확률변수, 확률분포

- 표본공간(sample space) : 확률 실험(random experiment)에서의 모든 가능한 결과들의 집합
- 표본(sample) : 확률 실험의 결과 하나. 표본 공간의 원소.
- 사건(event) : 표본들의 모임. 표본 공간의 부분집합.
- 사건공간(event space) : 가능한 모든 사건들의 집합
- 확률변수(random variable) : 표본을 입력으로 받아 실수(real number)를 출력으로 내는 함수

    - ex) 동전을 두 번 던지는 상황을 생각해 보자.
        - 발생할 수 있는 모든 결과는 $\textrm{HH}$, $\textrm{HT}$, $\textrm{TH}$, $\textrm{TT}$, 이렇게 4가지이다. 즉, 표본공간 $\Omega$는 다음과 같다.

            $$\Omega = \{\textrm{HH},\, \textrm{HT},\, \textrm{TH},\, \textrm{TT}\}$$

        - "확률변수 $X$를 앞면이 나온 횟수로 정의한다"는 말은, $X$를 다음과 같이 mapping하는 함수로 정의하겠다는 말이다.

            |                        sample                        | $X$ |
            |:----------------------------------------------------:|:---:|
            |        "앞면이 0번 나온 표본"($\textrm{TT}$)         |  0  |
            | "앞면이 1번 나온 표본"($\textrm{HT}$, $\textrm{TH}$) |  1  |
            |        "앞면이 2번 나온 표본"($\textrm{HH}$)         |  2  |

        - $X = 1$이라는 말은, 함수 $X$의 값을 1로 만드는 표본들의 집합(사건)을 뜻하는 말로서, 집합 $\{\textrm{HT},\, \textrm{TH}\}$를 의미한다.

    - 확률변수의 정의역은 표본공간이고, 공역은 실수 집합이다. 확률변수의 치역을 상태공간(state space)라 한다.
    - 표본공간이 가산집합인 경우 이산확률변수(discrete random variable), 불가산집합인 경우 연속확률변수(continuous random variable)이라 한다.

- 확률함수(probability function) 또는 확률분포(probability distribution) : 사건을 입력으로 받아 확률값(probability)를 출력으로 내는 함수
    - ex) 동전을 두 번 던지는 상황에서, 확률변수 $X$를 앞면이 나온 횟수로 정의하자.
        - 각 확률변수 값에 따른 확률값은 다음과 같다.

            |                        sample                        | $X$ | $P(X)$ |
            |:----------------------------------------------------:|:---:|:------:|
            |        "앞면이 0번 나온 표본"($\textrm{TT}$)         |  0  |  1/4   |
            | "앞면이 1번 나온 표본"($\textrm{HT}$, $\textrm{TH}$) |  1  |  1/2   |
            |        "앞면이 2번 나온 표본"($\textrm{HH}$)         |  2  |  1/4   |

    - 확률분포를 알면 특정 사건에 대한 확률값을 (당연히) 바로 알 수 있다.
        - ex) 위 예시에서, $X=1$인 사건의 확률은 $P(X=1) = P(\{ \textrm{HT},\, \textrm{TH}\}) = \frac{1}{2}$이다.

    - 이산확률변수에 대한 확률분포를 이산확률분포(discrete probability distribution), 연속확률변수에 대한 확률분포를 연속확률분포(continuous probability distribution)이라 한다.
        - 이산확률분포는 확률질량함수(PMF, probability mass function)를 통해 표현 가능
        - 연속확률분포는 확률밀도함수(PDF, probability density function)를 통해 표현 가능

    - 유명한 확률분포 : 균등분포(uniform distribution), 푸아송 분포(Poisson distribution), 정규분포(normal distribution, Gaussian distribution) 등


## 확률분포 모델링

- 확률분포에 대해서도 "식 → 값", "값 → 식" 두 가지 종류의 문제를 생각해 볼 수 있음

- "식 → 값"
    - ex) 확률변수 $X$를 동전을 던졌을 때 앞면이 나오는 경우 1, 뒷면이 나오는 경우 0이라 정의할 때, $X$가 베르누이분포(Bernoulli distribution) $\mathrm{Bern}(x | \mu=0.5)$를 따른다는 것이 알려져 있다고 하자. 이때 앞면이 나올 확률은?
    - 확률분포로부터 쉽게 확률(probability)을 계산할 수 있음

- "값 → 식"
    - ex) 찌그러진 동전을 100번 던져 앞면을 60번, 뒷면을 40번 얻었다고 하자. 확률변수 $X$를 동전을 던졌을 때 앞면이 나오는 경우 1, 뒷면이 나오는 경우 0이라 정의할 때, $X$는 어떤 확률분포를 따르는가? 더 나아가, 101번째 동전을 던진다면, 앞면, 뒷면이 나올 확률은 각각 얼마로 추정할 수 있는가?
        - 직관적으로, 이때까지의 데이터로 미루어보아, 앞면이 나올 확률을 0.6, 뒷면이 나올 확률을 0.4로 추정하는 것이 합리적
        - 이 직관을 어떻게 수학적으로 표현할까? → likelihood

- likelihood(가능도, 우도)
    - "주어진 데이터"가 "추정한 확률분포"로부터 나왔을 가능성
    - 추정한 확률분포가 얼마나 "그럴듯한지(likelyhood)"를 수치화
    - "좋은 추정"을 했다면 likelihood가 높음 → 즉, likelihood를 최대로 만드는 추정(MLE, maximum likelihood estimation)을 찾는 것이 모델링의 목표
    - 어떻게 계산하나?
        - 추정한 확률분포에 데이터를 대입. 즉, 추정한 확률분포의 출력값.
        - 이산확률분포의 경우, PMF의 출력값
        - 연속확률분포의 경우, PDF의 출력값

- 확률 vs. likelihood
    - 확률 : 확률분포의 파라미터 $\theta$가 고정되어 있을 때, 확률변수의 값 또는 범위가 나올 확률을 계산

        $$P(X|\theta)$$

    - likelihood : 데이터(확률변수의 값 또는 확률변수의 범위가 나올 확률) $X$가 고정되어 있을 때, 확률분포의 파라미터 $\theta$가 데이터를 얼마나 잘 설명하는지를 계산

        $$L(\theta | X)$$

    - 이산확률분포의 경우 PMF의 출력값은 우리가 직관적으로 이해하는 확률값과 같다. 즉, 이산확률분포에서는 확률 = likelihood
        - 단, 값이 같다는 것이지, 그 의미(뉘앙스)가 같다는 건 아니다.
        - 확률 계산 시에는 PMF가 참임을 알고 있는 상태
        - likelihood 계산 시에는 PMF가 아직 참임을 모르는 상태(추정)

    - 연속확률분포의 경우 PDF의 출력값은 우리가 직관적으로 이해하는 확률값과 다르다. 즉, 연속확률분포에서는 확률 ≠ likelihood
        - PDF의 $y$값은 likelihood를 

- MLE(maximum likelihood estimation)

    - 모든 데이터에 대해 likelihood를 최대화하는 확률분포를 찾기 위한 방법

    - 각 데이터를 독립(iid, independent and identical distributed)이라 가정하면, 데이터 전체에 대한 likelihood는 각 데이터의 likelihood의 곱이 됨

    - 일반적으론 likelihood를 직접 최대화하기보단, negative log likelihood(NLL loss)를 최소화하는 방식으로 사용
        - NLL(negative log likelihood) : likelihood의 계산값에 log를 씌우고 부호를 음수로 바꾼 것

        - 왜 log를 씌우나?
            - 역전파(backpropagation) 시 미분을 용이하게 하기 위해
            - 데이터 전체에 대한 log likelihood는 각 데이터의 log likelihood의 합이 됨

        - 왜 negative를 씌우나?
            - 다른 대부분의 loss(ex. MSE loss)는 최소화하는 방향으로 학습됨
            - 이를 똑같이 맞춰주기 위해, likelihood를 최대화하는게 아니라 negative likelihood를 최소화하는 방향으로 학습함

- 모델링을 바라보는 세 가지 관점
    - "데이터를 가장 잘 설명할 수 있는 관계식의 파라미터들을 찾는다" : 데이터를 가장 잘 설명할 수 있는 관계식은 어떤 식을 의미할까?
    - 관점 1. 오차의 관점 : MSE(men square error)
        - 각 관측값에 대해 추정한 관계식의 예상값과 참값 사이의 오차를 계산하고, 그 오차가 작을수록 좋은 모델이라 생각
    - 관점 2. likelihood의 관점 : MLE(maximum likelihood estimation)
        - $N$개의 데이터가 (모집단으로부터) $N$번의 실험/추출로 얻어진 것이라 보고, likelihood를 최대화하는 확률분포가 좋은 모델이라 생각
    - 관점 3. 엔트로피의 관점 : CE(cross entropy)
        - 주어진 데이터를 일종의 정보(information)로 보고, 평균 정보량(엔트로피)을 가장 줄이는 모델이 좋은 모델이라 생각
        - 평균 정보량을 줄인다 = 주어진 데이터를 얻은 일이 별로 놀랍지 않은(흔한) 일이다 = 모델이 더 그럴듯하다