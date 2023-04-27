---
{"title":"Text Generation with CLM","date_created":"2023-04-25","date_modified":"2023-04-25","dg-publish":true,"alias":"Text Generation with CLM","dg-path":"nlp/huggingface-clm-generation.md","permalink":"/nlp/huggingface-clm-generation/","dgPassFrontmatter":true,"created":"2023-04-25","updated":"2023-04-25"}
---


본 문서에서는 CLM(Causal Language Model)에서 텍스트를 생성하는 방법에 대해 알아본다.

## CLM(Causal Language Model)

- Causal Language Model(CLM) : Causal Language Modeling task에 대해 학습된 language model. ex) GPT
    - Causal Language Modeling task : 앞 token들로부터, 그 다음 token을 맞추는 task
        - 텍스트를 단방향(uni-direction)으로만 볼 수 있다.
        - auto-regressive
    - 일반적으로 자연어 생성(NLG, natural language generation) task에 강하다고 알려져 있음
    - 오늘날에는 transformer의 decoder 구조를 이용해서 주로 만듦 → decoder 모델이라 부르기도 함

- cf) Masked Language Model(MLM) : Masked Language Modeling task에 대해 학습된 language model. ex) BERT
    - Masked Language Modeling task : 앞뒤 token들로부터, 가려진(masked) token을 맞추는 task.
        - 텍스트를 양방향(bidirectional)으로 볼 수 있다.
        - auto-encoding
    - 일반적으로 자연어 이해(NLU, natural language understanding) task에 강하다고 알려져 있음
    - 오늘날에는 transformer의 encoder 구조를 이용해서 주로 만듦 → encoder 모델이라 부르기도 함

## Text Generation

### generation task == search task

- objective : '가장 그럴듯한' sequence를 찾는 것 = 가장 확률이 높은 sequence를 찾는 것
- chain rule에 의해, sequence의 확률은 sequence를 구성하고 있는 각 token들의 (이전 token이 주어졌을 때) 등장 확률의 곱으로 생각할 수 있다.
    - 일반적으론 계산 안정성을 위해서 확률의 곱 연산을 수행하기보단 log 확률의 합 연산을 수행
    - 자세한 설명은 [[Knowledge Base/nlp/language-model\|Language Model]] 문서 참조
- 즉 생성 문제 = 최적 경로 탐색 문제

### CLM에서의 text generation

- 기본 idea : `t`개의 token으로 구성된 sequence가 CLM에 입력되었을 때, 마지막 token의 마지막 layer에서의 hidden state에는 다음 `t + 1`번째 token을 예측하기 위한 정보가 들어 있다.

- 마지막 token의 마지막 decoder layer에서의 hidden state를 LM Head에 입력하면, vocabulary에 있는 각 token들이 다음 token으로 얼마나 적절한지 정보를 담고 있는 logit을 얻을 수 있다.
    - LM Head : fully-connected(= linear) layer (`Linear(hidden_size, vocab_size)`)

- LM head의 출력값(즉, logit)에 softmax를 씌우면, vocabulary에 있는 각 token들이 다음 token으로 올 수 있는 확률(probability)을 얻을 수 있다.

### 기타

- (길이가 다른) batch 입력을 처리하려면, padding을 왼쪽에 주어야 한다.
    - 일반적으로 하듯이 오른쪽에 padding을 주게 되면, batch 내 몇몇 example은 마지막 token에 도달했지만(그래서 이제 generation을 시작해야 하지만) 나머지 example은 아직 도달하지 않은(generation을 시작할 수 없는) 상태가 될 수 있음.
    - 왼쪽 padding을 사용해야 batch 내 모든 example들의 마지막 token들을 같은 위치(마지막 열)에 모을 수 있다.
        - 어차피 padding이 들어가는 곳에는 attention mask가 씌워지기에 큰 문제가 없다.
    - 🤗huggingface transformers 라이브러리의 tokenizer에서는 `padding_side="left"`를 주어 손쉽게 왼쪽 padding을 줄 수 있다.
        ```python title:"left padding" linenos highlight:"3"
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id # gpt2의 tokenizer에는 pad_token이 정의되어 있지 않기 때문에, eos_token을 pad_token으로 사용하라고 알려줘야 한다.

        texts = [
            "Hello",
            "I love",
            "I want to"
        ]

        tokenized = tokenizer(texts, padding=True, return_tensors="pt")

        print(tokenized.input_ids)
        # tensor([[50256, 50256, 15496],
        #         [50256,    40,  1842],
        #         [   40,   765,   284]])

        print(tokenized.attention_mask)
        # tensor([[0, 0, 1],
        #         [0, 1, 1],
        #         [1, 1, 1]])
        ```

## Text Generation Strategy

- 크게 deterministic한 방법과 stochastic한 방법으로 분류 가능
    - deterministic : greedy search, beam search
    - stochastic : top-k sampling, top-p sampling(nucleus sampling)
    - 기타 : contrastive search

### Greedy Search

#### Idea

![greedy search](https://huggingface.co/blog/assets/02_how-to-generate/greedy_search.png)

- 매 시점마다 가장 확률이 높은 token들만을 계속해서 선택하는 방식
    - 즉 model에 이전까지의 입력/생성 token을 입력하고 그 값으로 다음 token을 예측하는 연산을 `max_new_token`번 수행해야 한다.
    - 예를 들어 위 그림에선 ("The", "nice", "woman")이 선택된다.

```python title:"greedy search : skeleton code" linenos highlight:"27"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer, model 불러오기
tokenizer =  AutoTokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id # gpt2의 tokenizer에는 pad_token이 정의되어 있지 않기 때문에, eos_token을 pad_token으로 사용하라고 알려줘야 한다.

model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(device) # 마찬가지

# tokenization
texts = ["Hello", "I love", "I want to"]

tokenized = tokenizer(texts, padding=True, return_tensors="pt").to(device)
input_ids, attention_mask = tokenized.input_ids, tokenized.attention_mask # tensor[batch_size, seq_len]

# generation
max_new_tokens = 5
for _ in range(max_new_tokens):
    with torch.no_grad():
        model_inputs = model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask) # 텍스트 생성에 필요한 입력값을 가공해 주는 method
        model_output = model(**model_inputs)
    
    # next token 계산 : argmax 연산을 이용, logit 값이 가장 큰 token을 선택(greedy search)
    next_token_logits = model_output.logits[:, -1, :] # tensor[batch_size, vocab_size]
    next_tokens = next_token_logits.argmax(dim=1).reshape(-1, 1) # tensor[batch_size, 1]
    
    # prepare for next iteration
    input_ids = torch.cat([input_ids, next_tokens], dim=1) 
    attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=1)

print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
# ["Hello, I'm sorry,", "I love the way you're doing", 'I want to be able to do that']
```

- 코드 해설
    - line 20 - 35 : `max_new_tokens`번 iteration을 돌며 next token을 생성
    - line 22 : `model.prepare_inputs_for_generation()`
        - model이 텍스트 생성을 진행할 수 있도록 (model-specific하게) 입력값을 가공해 주는 method
        - 후술할 `.generate()` 메소드를 사용하려면 모델은 반드시 이 메소드를 구현해야 한다.
    - line 26 :  마지막 token에 대한 logit 값
        - 마지막 token의 마지막 decoder layer에서의 hidden state 값을 LM head에 넣은 값
    - line 27 : argmax 연산을 이용, logit 값이 가장 큰 token을 선택 (greedy search)
    - line 30, 31 : 다음 iteration을 위해 입력값 뒤에 새로 만들어진 token을 붙임

> [!warning] 
> 이 구현은 greedy search의 뼈대만 간단히 구현한 것으로, 실제로 사용하기엔 무리가 있다.
> - 예를 들어 `<eos>` token 등장 시 early stop을 수행한다던지 하는 기능들은 모두 빠져 있다.
> - greedy search를 사용하고 싶다면 후술할 `.generate()` 메소드를 사용하자.

#### 장/단점

- 장점 : 
    - 빠르다.
    - 언제나 같은 값을 내놓는다(deterministic).
    - 메모리 소비량 매우 적음
- 단점 :
    - 최선의 정답이 아닐 수 있다.
        - 매 순간 가장 높은 확률의 token을 선택하는 것이 가장 높은 확률의 sequence를 선택한다는 것을 보장해주지 않는다.
        - 지금의 최선이 나중에는 나쁜 선택일 수 있다.
            - 앞 token들의 확률은 낮아도, 뒷 token들로 가면 확률이 매우 높아 최종 sequence의 확률이 더 높아질 수도 있다.
    - 별로 참신한(surprising) 텍스트를 생성하지 못한다.

#### 사용법

- 🤗huggingface transformers 라이브러리의 `generate()` 함수를 이용하면 손쉽게 greedy search를 수행할 수 있다.
    - `.generate()` 메소드 호출 시 `num_beams=1`, `do_sample=False`로 주면 사용할 수 있다.

        ```python title:".generate() 메소드 이용해 greedy search 수행하기" linenos highlight:"27,28"
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # tokenizer, model 불러오기
        tokenizer =  AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id # gpt2의 tokenizer에는 pad_token이 정의되어 있지 않기 때문에, eos_token을 pad_token으로 사용하라고 알려줘야 한다.
        
        model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(device) # 마찬가지
        
        # tokenization
        texts = [
            "Hello",
            "I love",
            "I want to"
        ]
        
        tokenized = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        
        # generation
        output = model.generate(
            **tokenized,
            max_new_tokens=5, # (최대) 5개의 token만 생성
            return_dict_in_generate=True, # 출력값을 tuple 형태가 아닌 dict 형태로 받기 위한 옵션
            output_scores=True, # score 값을 받기 위한 옵션
            num_beams=1,
            do_sample=False
        )
    
        # decoding
        print(tokenizer.batch_decode(output.sequences, skip_special_tokens=True))
        # ["Hello, I'm sorry,", "I love the way you're doing", 'I want to be able to do that']
        ```

#### 출력값

> [!info] Reference
>  https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation.GreedySearchDecoderOnlyOutput

##### `sequences`

- shape : `torch.LongTensor[batch_size, sequence_length`

- 생성 결과 만들어진 sequence
    - 입력으로 준 token들까지 모두 포함되어 있다는 점에 유의
        - 새로 생성된 token들만 보고 싶다면 다음과 같이 입력 token들을 잘라내고 보면 된다.

            ```python title:"새로 생성된 token들만 보기" linenos highlight:"3"
            input_ids_len = tokenized.input_ids.shape[1]

            generated_token_ids = output.sequences[:, input_ids_len:]

            print(tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True))
            # [", I'm sorry,", " the way you're doing", ' be able to do that']
            ```

    - 두 번째 dimension(`sequence_length`)의 크기는 최대 `max_length`까지 갈 수 있다.
        - 만약 batch 내 모든 example들이 `<eos>` token을 생성해 생성이 일찍 종료되었다면 그보다 짧아질 수 있다.

##### `scores`

- shape : `tuple(torch.FloatTensor[batch_size, vocab_size])`
- `.generate()` 메소드 호출 시 `output_scores=True` 옵션을 주어야 확인할 수 있다.

- 각 시점에서의 logit(LM head의 출력)을 모은 tuple
    - tuple의 크기는 최대 `max_new_token`까지 갈 수 있다.
        - 만약 batch 내 모든 example들이 `<eos>` token을 생성해 생성이 일찍 종료되었다면 그보다 짧아질 수 있다.
    - 위 skeleton 코드에서 매 시점마다 `next_token_logits`을 모은 값

> [!tip] Advanced
> 만약 logits processor를 사용한다면, logits processor가 적용된 값이 담긴다.

##### `attentions`

- shape : `tuple(tuple(torch.FloatTensor[batch_size, num_heads, generated_lengths, sequence_length]))`
- `.generate()` 메소드 호출 시 `output_attentions=True` 옵션을 주어야 확인할 수 있다.

- 각 시점에서의, 각 layer마다의, attention 값들을 모은 tuple들의 tuple

##### `hidden_states`

- shape : `tuple(tuple(torch.FloatTensor[batch_size, generated_length, hidden_size]))`
- `.generate()` 메소드 호출 시 `output_hidden_states=True` 옵션을 주어야 확인할 수 있다.

- 각 시점에서의, 각 layer마다의, hidden state를 모은 tuple들의 tuple

#### 여담 : transition score

- `model.compute_transition_scores()` 메소드를 이용하면 매 시점 model이 얼마만큼의 확률로 다음 token을 추정했는지를 알 수 있다.
    - 이 값을 transition score라 한다.
    - transition score가 0이라면 해당 시점에선 생성이 진행되지 않았음을(즉, `<eos>` token의 등장 등으로 인한 early stop이 있었음을) 나타낸다.

    ```python title:"compute_transition_scores() 메소드" linenos highlight:"31-36"
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # tokenizer, model 불러오기
    tokenizer =  AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id # gpt2의 tokenizer에는 pad_token이 정의되어 있지 않기 때문에, eos_token을 pad_token으로 사용하라고 알려줘야 한다.
    
    model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(device) # 마찬가지
    
    # tokenization
    texts = [
        "Hello",
        "I love",
        "I want to"
    ]
    
    tokenized = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    
    # generation
    output = model.generate(
        **tokenized,
        max_new_tokens=5, # (최대) 5개의 token만 생성
        return_dict_in_generate=True, # 출력값을 tuple 형태가 아닌 dict 형태로 받기 위한 옵션
        output_scores=True, # score 값을 받기 위한 옵션
        num_beams=1,
        do_sample=False
    )
    
    # transition score 계산
    transition_scores = model.compute_transition_scores(
        sequences=output.sequences,
        scores=output.scores,
        normalize_logits=True
    ) # tensor[batch_size, sequence_length]
    
    # 첫 번째 example의 transition score 확인
    input_ids_len = tokenized.input_ids.shape[1]
    generated_token_ids = output.sequences[:, input_ids_len:]
    for token, score in zip(generated_token_ids[0], transition_scores[0]):
        print(f"{token:5d} | {tokenizer.decode(token):8s} | {score.item():.3f} | {score.exp().item():.3%}")
        #    11 | ,        | -2.343 | 9.602%
        #   314 |  I       | -2.298 | 10.045%
        #  1101 | 'm       | -1.652 | 19.164%
        #  7926 |  sorry   | -2.471 | 8.453%
        #    11 | ,        | -1.528 | 21.707%
    ```
    
    - 코드 설명
        - line 35 : `normalize_logits=True`를 주면 각 token의 logit 값에 log softmax를 취한 값을 반환한다. `normalize_logits=False`를 주면 logit 값이 그대로 출력된다.
        - line 42 : `normalize_logits=True`를 주었기 때문에, 위 예제에서 `transition_scores`에 담긴 값은 log softmax가 취해진 logit 값이다. 따라서 이를 확률 값으로 변환하려면 `.exp()` 연산을 수행해야 한다.

### Beam Search

#### Idea

![beam search](https://huggingface.co/blog/assets/02_how-to-generate/beam_search.png)

- 탐색 시 `num_beams`개의 sequence 후보를 두고, 그 중 최고를 찾는 방법
    - 각 sequence 후보를 "beam"이라 한다.
    - 매 시점, `num_beams`개의 beam에 대해, vocabulary의 각 token들을 이어붙인 `vocab_size`개의 sequence의 확률을 구한 뒤(즉, `num_beams * vocab_size`개의 sequence 확률을 구한 뒤), 이 중 가장 높은 확률을 가진 sequence `num_beams`개를 추리는 과정을 반복
    - 예를 들어 `num_beams=2`인 위 그림에선 ("The", "dog", "has")가 선택된다.
        - 점선은 greedy search가 탐색한 path를 나타낸다.

- greedy search의 한계를 극복
    - 미래에 더 나은 sequence를 찾을 수 있도록 `num_beams`개의 후보를 유지
    - top1이 아니라 topk(k=`num_beams`) 연산을 수행
        - 실제 구현 시에는 k=`2 * num_beams`를 사용하여 여분의 beam을 만들어 놓음
    - greedy search는 `num_beams=1`인 beam search라 이해할 수 있음

- batch 처리
    - `num_beams`개의 경우의 수를 고려한다는 것은, 결국 `num_beams`번의 inference를 해야 한다는 것
    - 따라서 `batch_size` 크기의 batch 입력에 대해 `num_beams`만큼의 beam search를 수행하는 문제는, `batch_size * num_beams`크기의 batch 입력에 대해 inference를 수행하는 문제로 볼 수 있다. → 병렬화 가능
    - beam search를 시작할 때, 각 example을 `num_beams`개 복제하여 `batch_size * num_beams` 크기의 batch를 만들고 inference 진행
        - 다만 이렇게 하면 각 beam에서 나오는 결과값이 항상 같아지므로, 모든 beam이 같은 값으로 채워진다는 문제가 있음
        - 첫 번째 beam의 초기 sequence 확률을 0으로, 나머지 beam들의 초기 sequence score를 `-inf`로 채워놓으면 이 문제를 피할 수 있음
        - 그 결과 `beam_indices`의 첫 번째 column(즉, 생성을 시작하는 최초 시점에서 선택된 beam의 index)는 항상 0이 된다.

#### 장/단점

- 장점 : 
    - (greedy search 대비) 생성되는 sequence의 품질이 좋다.
    - 언제나 같은 값을 내놓는다(deterministic).
- 단점 :
    - 최선의 정답이 아닐 수 있다.
        - `num_beams`가 작으면 더더욱.
        - 물론 greedy search보다는 '최선에 가까운' 정답을 찾긴 한다.
    - 메모리 소비
        - (greedy search 대비) 메모리 소비가 (최소) `num_beams`배 커진다.

#### 사용법

- 🤗huggingface transformers 라이브러리의 `generate()` 함수를 이용하면 손쉽게 beam search를 수행할 수 있다.
    - `.generate()` 메소드 호출 시 `num_beams > 1`, `do_sample=False`로 주면 사용할 수 있다.
    - `num_return_sequences` 인자를 이용해 (각 example 당) 반환되는 sequence들의 수를 설정할 수 있다. 이 값은 (당연히) `num_beams`보다 클 수 없다.

        ```python title:".generate() 메소드 이용해 beam search 수행하기" linenos highlight:"27,28"
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # tokenizer, model 불러오기
        tokenizer =  AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id # gpt2의 tokenizer에는 pad_token이 정의되어 있지 않기 때문에, eos_token을 pad_token으로 사용하라고 알려줘야 한다.
        
        model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(device) # 마찬가지
        
        # tokenization
        texts = [
            "Hello",
            "I love",
            "I want to"
        ]
        
        tokenized = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        
        # generation
        output = model.generate(
            **tokenized,
            max_new_tokens=5, # (최대) 5개의 token만 생성
            return_dict_in_generate=True, # 출력값을 tuple 형태가 아닌 dict 형태로 받기 위한 옵션
            output_scores=True, # score 값을 받기 위한 옵션
            num_beams=5,
            do_sample=False,
            num_return_sequences=4
        )
    
        # decoding
        print(tokenizer.batch_decode(output.sequences, skip_special_tokens=True))
        # ["Hello.\n\nI'm", 'I love it. I love it', 'I want to thank you for your support']
        ```

#### 출력값

> [!info] Reference
>  https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation.BeamSearchDecoderOnlyOutput

##### `sequences`

- shape : `torch.LongTensor[batch_size * num_return_sequences, sequence_length`

- 생성 결과 만들어진 sequence
    - 입력으로 준 token들까지 모두 포함되어 있다는 점에 유의
    - 첫 번째 dimension(`batch_size * num_return_sequences`)이 단순히 `batch_size`가 아니라는 점에 유의
        - 다만 순서는 보장된다. 예를 들어 `batch_size=3`, `num_return_sequence=4`라면, `output.sequences[0:4]`는 첫 번째 example, `output.sequences[4:8]`은 두 번째 example, `output.sequences[8:12]`는 세 번째 example에 대한 생성이다.
    - 두 번째 dimension(`sequence_length`)의 크기는 최대 `max_length`까지 갈 수 있다.
        - 만약 batch 내 모든 example들이 `<eos>` token을 생성해 생성이 일찍 종료되었다면 그보다 짧아질 수 있다.

##### `sequences_scores`

- shape : `torch.FloatTensor[batch_size * num_return_sequences]`
- `.generate()` 메소드 호출 시 `output_scores=True` 옵션을 주어야 확인할 수 있다.

- 생성된 sequence의 최종 score
    - 각 token들 사이의 transition score의 합에 length penalty를 적용한 값
        
        $$\textrm{score} = \frac{1}{t}\sum_i
{ #t}
 \log p_{\theta} (x_i | x_{<i})$$
        
        - $\log p_{\theta} (x_i | x_{<i})$ : transition score, $t$ : sequence의 길이
        - transition score는 logit 값에 log softmax를 취한 값이므로, 이들을 더하면 sequence의 확률을 계산할 수 있다(정확히는, sequence 확률에 log를 씌운 값(= sequence score)을 얻을 수 있다).
        - 다만 단순히 더하기만 하면 길이가 긴 sequence일수록 확률이 낮게 나오므로, transition score들의 합을 sequence의 길이로 나눠 준다(length penalty)
            - sequence의 길이는 입력까지 포함한 길이라는 점에 유의

        > [!tip] Advanced
        > 만약 `length_penalty` $l$을 설정한 경우 sequence score는 다음과 같이 계산된다.
        > 
        > $$\textrm{score} = \frac{1}{t \cdot l}\sum_i
{ #t}
 \log p_{\theta} (x_i | x_{<i})$$
        
        ```python title:"sequence score 직접 계산하기" linenos highlight:"44"
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # tokenizer, model 불러오기
        tokenizer =  AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id # gpt2의 tokenizer에는 pad_token이 정의되어 있지 않기 때문에, eos_token을 pad_token으로 사용하라고 알려줘야 한다.
        
        model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(device) # 마찬가지
        
        # tokenization
        texts = [
            "Hello",
            "I love",
            "I want to"
        ]
        
        tokenized = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        
        # generation
        output = model.generate(
            **tokenized,
            max_new_tokens=5, # (최대) 5개의 token만 생성
            return_dict_in_generate=True, # 출력값을 tuple 형태가 아닌 dict 형태로 받기 위한 옵션
            output_scores=True, # score 값을 받기 위한 옵션
            num_beams=5,
            do_sample=False,
            num_return_sequences=4
        )
        
        # transition score 계산
        transition_scores = model.compute_transition_scores(
            sequences=output.sequences,
            scores=output.scores,
            beam_indices=output.beam_indices,
            normalize_logits=True
        ) # tensor[batch_size * num_return_sequences, sequence_length]

        # sequence score 계산
        input_ids_len = tokenized.input_ids.shape[1]
        output_len = input_ids_len + (transition_scores < 0).sum(axis=1)
        
        calculated_sequences_scores = transition_scores.sum(axis=1) / output_len
        
        for i, (calculated_sequence_score, sequence_score) in enumerate(zip(calculated_sequences_scores, output.sequences_scores)):
            print(f"example #{i:02d} | {calculated_sequence_score:.4f} | {sequence_score:.4f} | {torch.isclose(calculated_sequence_score, sequence_score)}")
            # example #00 | -1.1756 | -1.1756 | True
            # example #01 | -1.1891 | -1.1891 | True
            # example #02 | -1.2796 | -1.2796 | True
            # example #03 | -1.2798 | -1.2798 | True
            # example #04 | -1.0159 | -1.0159 | True
            # example #05 | -1.0794 | -1.0794 | True
            # example #06 | -1.0991 | -1.0991 | True
            # example #07 | -1.1187 | -1.1187 | True
            # example #08 | -1.0093 | -1.0093 | True
            # example #09 | -1.0591 | -1.0591 | True
            # example #10 | -1.0797 | -1.0797 | True
            # example #11 | -1.0971 | -1.0971 | True
        ```

        - 코드 설명
            - line 36 : (greedy search 때와는 다르게) beam search에서는 `beam_indices` 항목도 넘겨줘야 transition score를 계산할 수 있다.
            - line 42 : `<eos>` token 이후의 token들의 transition score는 0이므로, 0보다 작은 항목들의 개수를 세면 생성된 sequence의 실제 길이를 알 수 있다.
            - line 47 : 부동소수점 문제로 인해 float 간의 비교는 단순히 `==`로 할 수 없다. `torch.isclose()` 메소드를 사용해 두 값이 충분히 가까운지 비교했다.

##### `scores`

- shape : `tuple(torch.FloatTensor[batch_size * num_beams * num_return_sequences, vocab_size])`
- `.generate()` 메소드 호출 시 `output_scores=True` 옵션을 주어야 확인할 수 있다.

- 각 시점에서의 logit(LM head의 출력)을 모은 tuple
    - tuple의 크기는 최대 `max_new_token`까지 갈 수 있다.
        - 만약 batch 내 모든 example들이 `<eos>` token을 생성해 생성이 일찍 종료되었다면 그보다 짧아질 수 있다.

##### `beam_indices`

- shape : `torch.LongTensor[batch_size * num_return_sequences, sequence_length]`
- `.generate()` 메소드 호출 시 `output_scores=True` 옵션을 주어야 확인할 수 있다.

- 각 시점에서 선택된 `num_beams`개의 sequence가, 이전 시점에서 만들어진 beam들 중 몇 번째 beam에 새 token을 붙여 만들어 진 것인지를 나타내는 tensor
    - 예를 들어 `beam_indices[i, j] = k`라는 말은, 시점 `j`에 만들어진 `i`번째 sequence는, 이전 시점의 `k`번째 sequence에 새로운 token을 붙여 만들어진 것임을 나타냄

##### `attentions`

- shape : `tuple(tuple(torch.FloatTensor[batch_size * num_beams, num_heads, generated_lengths, sequence_length]))`
- `.generate()` 메소드 호출 시 `output_attentions=True` 옵션을 주어야 확인할 수 있다.

- 각 시점에서의, 각 layer마다의, attention 값들을 모은 tuple들의 tuple

##### `hidden_states`

- shape : `tuple(tuple(torch.FloatTensor[batch_size * num_beams * num_return_sequences, generated_length, hidden_size]))`
- `.generate()` 메소드 호출 시 `output_hidden_states=True` 옵션을 주어야 확인할 수 있다.

- 각 시점에서의, 각 layer마다의, hidden state를 모은 tuple들의 tuple