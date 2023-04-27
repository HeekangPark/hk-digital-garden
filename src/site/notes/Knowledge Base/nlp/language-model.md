---
{"title":"Language Model","date_created":"2023-01-29","date_modified":"2023-02-17","dg-publish":true,"alias":"Language Model","dg-path":"nlp/language-model.md","permalink":"/nlp/language-model/","dgPassFrontmatter":true,"created":"2023-01-29","updated":"2023-02-17"}
---

## Language Model(LM)이란?

- 언어의 확률 분포를 나타낸 모델
- 문장의 출현 확률을 계산하는 모델 = 이전 단어(token)들이 주어졌을 때 다음 단어의 확률을 계산하는 모델
    - 위 두 정의는 본질적으로 같은 것
    - 예를 들어 문장 "$\textrm{<BOS> I love you <EOS>}$"를 생각해 보자.
    - 문장의 출현 확률 $P(\textrm{<BOS>}, \textrm{I}, \textrm{love}, \textrm{you}, \textrm{<EOS>})$는 chain rule에 의해 다음과 같이 전개됨

        $$\begin{align}
        &P(\textrm{<BOS>}, \textrm{I}, \textrm{love}, \textrm{you}, \textrm{<EOS>})\\
        &= P(\textrm{<EOS>} | \textrm{<BOS>}, \textrm{I}, \textrm{love}, \textrm{you}) \,\,P(\textrm{<BOS>}, \textrm{I}, \textrm{love}, \textrm{you})\\
        &= P(\textrm{<EOS>} | \textrm{<BOS>}, \textrm{I}, \textrm{love}, \textrm{you}) \,\,P(\textrm{you} | \textrm{<BOS>}, \textrm{I}, \textrm{love}) \,\,P(\textrm{<BOS>}, \textrm{I}, \textrm{love})\\
        &= P(\textrm{<EOS>} | \textrm{<BOS>}, \textrm{I}, \textrm{love}, \textrm{you}) \,\,P(\textrm{you} | \textrm{<BOS>}, \textrm{I}, \textrm{love}) \,\,P(\textrm{love} | \textrm{<BOS>}, \textrm{I}) \,\,P(\textrm{<BOS>}, \textrm{I})\\
        &= P(\textrm{<EOS>} | \textrm{<BOS>}, \textrm{I}, \textrm{love}, \textrm{you}) \,\,P(\textrm{you} | \textrm{<BOS>}, \textrm{I}, \textrm{love}) \,\,P(\textrm{love} | \textrm{<BOS>}, \textrm{I}) \,\,P(\textrm{I} | \textrm{<BOS>}) \,\,P(\textrm{<BOS>})\\
        \end{align}$$

     - 이때 각 항은 다음과 같은 의미를 가짐

        - $P(\textrm{<EOS>} | \textrm{<BOS>}, \textrm{I}, \textrm{love}, \textrm{you})$ : 이전 단어 "$\textrm{<BOS> I love you}$"가 주어졌을 때 다음 단어로 "$\textrm{<EOS>}$"가 등장할 확률
        - $P(\textrm{you} | \textrm{<BOS>}, \textrm{I}, \textrm{love})$ : 이전 단어 "$\textrm{<BOS> I love}$"가 주어졌을 때 다음 단어로 "$\textrm{you}$"가 등장할 확률
        - $P(\textrm{love} | \textrm{<BOS>}, \textrm{I})$ : 이전 단어 "$\textrm{<BOS> I}$"가 주어졌을 때 다음 단어로 "$\textrm{love}$"가 등장할 확률
        - $P(\textrm{I} | \textrm{<BOS>})$ : 이전 단어 "$\textrm{<BOS>}$"가 주어졌을 때 다음 단어로 "$\textrm{I}$"가 등장할 확률
        - $P(\textrm{<BOS>})$ : 단어 "$\textrm{<BOS>}$"가 등장할 확률 (상수)
        
    - 이처럼 문장의 출현 확률을 계산하는 모델은, 이전 단어들이 주어졌을 때 다음 단어의 확률을 계산하는 모델과 본질적으로 같다.

- 수식적으로 다음과 같이 나타낼 수 있음
    - $P(x_1, x_2, \cdots, x_n)$ 또는 $P(x_{1:n})$ : 문장 $x_1, x_2, \cdots, x_n$의 출현 확률

    - $P(x_n | x_1, x_2, \cdots, x_{n-1})$ 또는 $P(x_n | x_{<n})$ : 이전 단어 $x_1, x_2, \cdots, x_{n-1}$이 주어졌을 때, 다음 단어 $x_n$의 출현 확률

    - 두 표기법 사이에는 다음 식이 성립

        $$P(x_{1:n}) = \prod_{i=1}
{ #n}
 P(x_i | x_{<i})$$

        $$\log P(x_{1:n}) = \sum_{i=1}
{ #n}
 \log P(x_i | x_{<i})$$

- 한국어의 경우, 단어의 어순이 중요하지 않고, 또 생략 가능하기 때문에, 단어와 단어 사이의 확률을 계산하는 것이 어려움 → 한국어 LM을 만드는 것이 어려움
    - ex) "나는 너를 사랑해" = "너를 나는 사랑해" = "사랑해. 나는 너를."
    - "나는" 뒤에는 "너를", "사랑해"가 모두 올 수 있다 : 확률이 퍼짐

## Language Model의 활용

- 더 자연스러운 문장을 선택할 수 있다.
    - 더 자연스러운 문장일수록 등장 확률이 높다.
- 더 자연스러운 다음 단어를 샘플링할 수 있다. → 이걸 반복하면 문장을 생성 가능(Natural Language Generation)
    - 더 자연스러운 다음 단어일수록 등장 확률이 높다.

## Language Model의 분류

- Autoregressive Model(= Causal Language Model)
    - 이전 단어들이 주어졌을 때 다음 단어를 예측하는 방식(Causal Language Modeling Task)으로 학습된 LM
    - Transformer의 decoder로 구현
    - Natural Language Generation(NLG) Task에서 강점
    - ex) GPT
- Autoencoding Model
    - 주변 단어들이 주어졌을 때 가운데 단어를 예측하는 방식(Masked Language Modeling Task)으로 학습된 LM
    - Transformer의 encoder로 구현
    - Natural Language Understanding(NLU) Task에서 강점
    - ex) BERT
- Encoder-Decoder Model(= Seq2Seq Model)
    - Transformer의 encoder와 decoder를 모두 사용해 구현
    - NLU, NLG Task 모두 가능
    - ex) BART

## Language Model의 구조

### n-gram



[[Knowledge Base/nlp/n-gram\|n-gram]] 참조

### RNN



### Transformer



## Interpolation

- language model들을 linear하게 합친 것

    $$\tilde{P}(w_t | w_{t-k}, \cdots, w_{t-1}) = \lambda_1 P_1 (w_t | w_{t-k}, \cdots, w_{t-1}) + \lambda_2 P_2 (w_t | w_{t-k}, \cdots, w_{t-1}) + \cdots + \lambda_n P_n (w_t | w_{t-k}, \cdots, w_{t-1})$$

    - $\lambda_i$ : interpolation ratio ($\sum \lambda_i = 1$). $\lambda_i$의 값을 조정해서 특정 LM의 중요도(weight)를 조절할 수 있음

- 사용처
    - domain adapted LM : general LM + domain specific LM
        - general LM : general한 corpus에 대해 학습된 LM
        - domain specific LM : domain specific한 corpus에 대해 학습된 LM
        - domain specific한 corpus는 일반적으로 양이 적음 
            - domain specific LM은 unseen word sequence 문제가 발생할 확률 높음
            - general corpus와 domain specific corpus를 합쳐서 training하면 domain specific corpus의 정보가 중요하게 반영되지 않을 확률 높음
    - [[Knowledge Base/nlp/n-gram#back-off\|back-off]]

## Language Model의 평가

[[Knowledge Base/nlp/perplexity\|perplexity]] 참고

### GPT

- transformer의 decoder만을 활용해 만든 모델
- training objective
    - Causal Language Modeling(CLM) Task : masked token 앞에 있는 token들만 주어졌을 때(unidirectional), masked token을 예측하는 task

### BERT

- transformer의 encoder만을 활용해 만든 모델
- training objective
    - Masked Language Modeling(MLM) Task : masked token 앞뒤의 token들이 주어졌을 때(bidirectional), masked token을 예측하는 task
    - Next Sentence Prediction(NSP) Task : 한 문장 뒤에 오는 다음 문장을 예측하는 task
