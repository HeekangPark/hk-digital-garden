---
{"title":"n-gram Language Model","date_created":"2023-01-21","date_modified":"2023-01-29","dg-publish":true,"alias":"n-gram Language Model","dg-path":"nlp/n-gram.md","permalink":"/nlp/n-gram/","dgPassFrontmatter":true,"created":"2023-01-21","updated":"2023-01-29"}
---


## simple count method

- 이전 단어들이 주어졌을 때 다음 단어의 등장 확률은 다음과 같이 단어들의 등장 빈도를 이용해 근사할 수 있음

    $$P(x_t | x_{<t}) \approx \frac{\mathrm{COUNT}(x_1, x_2, \cdots, x_{t-1}, x_t)}{\mathrm{COUNT}(x_1, x_2, \cdots, x_{t-1})}$$

- 즉 corpus의 모든 단어 순열들을 lookup table에 넣고, 확률을 계산하고자 하는 단어가 query되었을 때 lookup table을 scan하는 것으로 확률 계산 가능

- simple count method의 문제점
    - unseen word sequence : corpus에 문장 "$x_1$, $x_2$, …, $x_{t-1}$", "$x_1$, $x_2$, …, $x_{t-1}$, $x_{t}$"이 없을 수도 있음 → 분모 또는 분자가 0이 되어 계산 불가

## n-gram method

- n-gram method : Markov assumption 적용, 앞 $k$개의 단어만 봄
    - corpus에 $t$개의 단어로 이루어진 문장 "$x_1$, $x_2$, …, $x_{t-1}$, $x_{t}$"이 정확히 존재할 때만 쓸 수 있는 [[Knowledge Base/nlp/n-gram#simple count method\|#simple count method]]의 한계를 극복

    $$P(x_t | x_{<t}) \approx P(x_t | x_{t-k}, \cdots, x_{t-1}) \approx \frac{\mathrm{COUNT}(x_{t-k}, \cdots, x_{t-1}, x_t)}{\mathrm{COUNT}(x_{t-k}, \cdots, x_{t-1})}$$

    $$\log P(x_{1:t}) \approx \sum_{i=1}
{ #t}
 \log P(x_i | x_{i-k}, \cdots, x_{i-1})$$

- $k=0$($n=1$) : unigram(1-gram)
    - (이전 단어에 대한 고려 없이) 전체 corpus에서 각 단어의 출현 빈도만 생각하는 LM

        $$P(x_t | x_{<t}) \approx P(x_t)$$
    
        $$\log P(x_{1:t}) \approx \sum_{i=1}
{ #t}
 \log P(x_i)$$

- $k=1$($n=2$) : bigram(2-gram)
    - 앞 1개의 단어만 보고 출현 빈도 count

        $$P(x_t | x_{<t}) \approx P(x_t | x_{t-1}) \approx \frac{\mathrm{COUNT}(x_{t-1}, \cdots, x_t)}{\mathrm{COUNT}(x_{t-1})}$$
    
        $$\log P(x_{1:t}) \approx \sum_{i=1}
{ #t}
 \log P(x_i | x_{t-1})$$

- $k=2$($n=3$) : trigram(3-gram)
    - 앞 2개의 단어만 보고 출현 빈도 count

        $$P(x_t | x_{<t}) \approx P(x_t | x_{t-2}, x_{t-1}) \approx \frac{\mathrm{COUNT}(x_{t-2}, x_{t-1}, x_t)}{\mathrm{COUNT}(x_{t-2}, x_{t-1})}$$
    
        $$\log P(x_{1:t}) \approx \sum_{i=1}
{ #t}
 \log P(x_i | x_{i-2}, x_{i-1})$$

- $n$이 커질수록 Markov assumption이 더 약하게 적용되므로 corpus에 더 정확히 fit 시킬 수 있음. 하지만 generality 떨어짐 → 적절한 $n$ 선택하는 것이 중요
    - 보통은 3-gram 많이 사용
    - corpus의 크기가 크다면 4-gram을 사용

- n-gram method의 문제점
    - unseen word sequence : [[Knowledge Base/nlp/n-gram#simple count method\|#simple count method]]의 문제점과 동일. [[Knowledge Base/nlp/n-gram#simple count method\|#simple count method]]보단 확률이 낮겠지만, corpus에 문장 "$x_{t-k}$, …, $x_{t-1}$", "$x_{t-k}$, …, $x_{t}$"이 없을 수도 있음 → 분모 또는 분자가 0이 되어 계산 불가

## smoothing

- add one smoothing : 단어 빈도수 counting할 때 1을 더해줌 (단, $|V|$는 전체 vocabulary의 크기)
    - 확률을 계산하고자 하는 단어 $x_t$의 앞 전체 단어($t$개) 또는 일부 단어($k$개)의 순열이 corpus에 정확히 존재해야 하는 [[Knowledge Base/nlp/n-gram#simple count method\|#simple count method]], [[Knowledge Base/nlp/n-gram#n-gram method\|#n-gram method]]의 문제점 해결

    $$P(x_t|x_{<t}) \approx \frac{\mathrm{COUNT}(x_{t-k}, \cdots, x_t) + 1}{\mathrm{COUNT}(x_{t-k}, \cdots, x_{t-1}) + |V|}$$

- 일반화
    - 다음과 같이 $a$을 더하는 형태로 일반화 가능

        $$P(x_t|x_{<t}) \approx \frac{\mathrm{COUNT}(x_{t-k}, \cdots, x_t) + a}{\mathrm{COUNT}(x_{t-k}, \cdots, x_{t-1}) + |V| \times a}$$

    - 다음과 같이 쓸 수도 있음

        $$P(x_t|x_{<t}) \approx \frac{\mathrm{COUNT}(x_{t-k}, \cdots, x_t) + b \times \frac{1}{|V|}}{\mathrm{COUNT}(x_{t-k}, \cdots, x_{t-1}) + b}$$

    - 이때 $\frac{1}{|V|}$는 vocabulary에서의 단어의 등장 빈도라 이해할 수 있음. 이를 다음과 같이 실제 corpus에서의 단어의 등장 빈도(즉, unigram) $P(x_t)$로 쓰면 조금 더 data-driven한 근사가 가능

        $$P(x_t|x_{<t}) \approx \frac{\mathrm{COUNT}(x_{t-k}, \cdots, x_t) + b \times P(x_{t})}{\mathrm{COUNT}(x_{t-k}, \cdots, x_{t-1}) + b}$$

- smoothing의 문제점
    - out of vocabulary(OoV) : 찾고자 하는 단어가 corpus에 아예 존재하지 않는다면(ex. 신조어) 여전히 확률 계산 불가

## back-off

- back-off : n-gram들을 interpolation

    $$\begin{align}
    \tilde{P}(w_t | w_{t-k}, \cdots, w_{t-1}) 
    &= \lambda_1 \,\,P(w_t | w_{t-k}, \cdots, w_{t-1}) \\
    &+ \lambda_2 \,\,P(w_t | w_{t-k + 1}, \cdots, w_{t-1}) \\
    &+ \cdots \\
    &+ \lambda_{k-1} \,\,P(w_t | w_{t-1}) \\
    &+ \lambda_{k} \,\,P(w_t)
    \end{align}$$

    - unseen word sequence에 대처할 수 있음
    - 문제점
        - out of vocabulary(OoV)