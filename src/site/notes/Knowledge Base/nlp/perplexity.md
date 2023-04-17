---
{"title":"Perplexity","date_created":"2023-01-29","date_modified":"2023-01-29","dg-publish":true,"alias":"Perplexity","dg-path":"nlp/perplexity.md","permalink":"/nlp/perplexity/","dgPassFrontmatter":true,"created":"2023-01-29","updated":"2023-01-29"}
---


- 테스트 문장 $x_1$, $x_2$, …, $x_n$에 대해, LM $P(x ; \theta)$의 perplexity는 다음과 같이 계산

    $$\begin{align}
    \mathrm{PPL}(x_1, x_2, \cdots, x_n ;\, \theta) 
    &= P(x_1, x_2, \cdots, x_n ;\, \theta)^{-\frac{1}{n}} \\
    &= \sqrt[n]{\frac{1}{P(x_1, x_2, \cdots, x_n ;\, \theta)}} \\
    &= \sqrt[n]{\frac{1}{\prod^n_{i=1}P(x_i | x_{<i} ;\, \theta)}} \\
    \end{align}$$

    - LM에서 테스트 문장이 나올 확률(likelihood)을 계산하여 그 역수에 기하평균($n$제곱근)을 계산 ($n$은 문장 길이)

    - perplexity가 작을수록 좋은 LM
        - 좋은 LM은 테스트 문장을 잘 만들어낼 것. 즉 테스트 문장의 likelihood는 커질 것. 따라서, 좋은 LM일수록 perplexity는 작아짐.

    - 기하평균을 구하는 이유?
        - 문장이 길어지면 chain rule에 의해 1보다 작은 확률값을 계속 곱해가는 likelihood는 당연히 작아짐
        - 문장 길이에 대한 영향을 없애기 위해 기하평균을 계산하여 normalization

- perplexity의 의미
    - 테스트 문장이 나올 확률(likelihood)의 역수의 기하평균($n$제곱근)
        - 테스트 문장의 likelihood가 클 수록 좋은 LM
    - 각 time-step 별 다음 단어를 선택할 때 헷갈리는 선택지 수의 평균
        - 선택지가 적을수록(= 다음 단어에 대해 강하게 확신할수록 = 다음 단어를 덜 헷갈릴수록) 좋은 LM
        - perplexity가 높다 = 다음 단어들에 대한 확률 분포가 flat하다
        - perplexity가 낮다 = 다음 단어들에 대한 확률 분포가 sharp하다
    - 테스트 문장(ground truth)과 LM의 확률분포 사이의 cross entropy의 exponential
        - 즉, PPL에 $\log$를 씌운 것이 cross entropy와 같음
        - 테스트 문장이 LM 입장에서 덜 놀라운(= 흔한 = 자연스러운 = 평균 정보량이 적은) 문장일수록 좋은 LM
        - perplexity를 낮추는 것은 cross entropy를 낮추는 것과 같다.