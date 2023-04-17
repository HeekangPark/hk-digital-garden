---
{"title":"huggingface GPT 모델 종류","date_created":"2023-03-21","date_modified":"2023-02-21","dg-publish":true,"dg-path":"/nlp/huggingface-gpts.md","permalink":"//nlp/huggingface-gpts/","dgPassFrontmatter":true,"created":"","updated":""}
---


## English

### [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)

|                             models                            | language | # of params | max seq len | # of layers | # of heads | hidden state dim | vocab size |
|:--------------------------------------------------------------:|:--------:|:-----------:|:-----------:|:-----------:|:----------:|:----------------:|:----------:|
|        #huggingface [`gpt2`](https://huggingface.co/gpt2)        | English  |    124M     |    1024     |     12      |     12     |       768        |   50257    |
| #huggingface [`gpt2-medium`](https://huggingface.co/gpt2-medium) | English  |    355M     |    1024     |     24      |     16     |       1024       |   50257    |
|  #huggingface [`gpt2-large`](https://huggingface.co/gpt2-large)  | English  |    774M     |    1024     |     36      |     20     |       1280       |   50257    |
|     #huggingface [`gpt2-xl`](https://huggingface.co/gpt2-xl)     | English  |    1.5B     |    1024     |     48      |     25     |       1600       |   50257    |

- paper : [[2018] Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- dataset : [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/) 40GB

### [Distilgpt2](https://huggingface.co/docs/transformers/model_doc/gpt2)

|                             models                             | language | # of params | max seq len | # of layers | # of heads | hidden state dim | vocab size |
|:--------------------------------------------------------------:|:--------:|:-----------:|:-----------:|:-----------:|:----------:|:----------------:|:----------:|
| #huggingface [`distilgpt2`](https://huggingface.co/distilgpt2) | English  |     82M     |    1024     |      6      |     12     |       768        |   50257    |

distilled version of `gpt2`

- dataset : [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/) 40GB

### [GPT-Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo)

|                                          models                                          | language | # of params | max seq len | # of layers | # of heads | hidden state dim | vocab size |
|:----------------------------------------------------------------------------------------:|:--------:|:-----------:|:-----------:|:-----------:|:----------:|:----------------:|:----------:|
| #huggingface [`EleutherAI/gpt-neo-125M`](https://huggingface.co/EleutherAI/gpt-neo-125M) | English  |    125M     |    2048     |     12      |     12     |       768        |   50257    |
| #huggingface [`EleutherAI/gpt-neo-1.3B`](https://huggingface.co/EleutherAI/gpt-neo-1.3B) | English  |    1.3B     |    2048     |     24      |     16     |       2048       |   50257    |
| #huggingface [`EleutherAI/gpt-neo-2.7B`](https://huggingface.co/EleutherAI/gpt-neo-2.7B) | English  |    2.7B     |    2048     |     32      |     20     |       2560       |   50257    |

EleutherAI's replication of the GPT-3 architecture

- dataset : [Pile dataset](https://pile.eleuther.ai/) 82GB

### [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)

|                                      models                                      | language | # of params | max seq len | # of layers | # of heads | hidden state dim | vocab size |
|:--------------------------------------------------------------------------------:|:--------:|:-----------:|:-----------:|:-----------:|:----------:|:----------------:|:----------:|
| #huggingface [`EleutherAI/gpt-j-6B`](https://huggingface.co/EleutherAI/gpt-j-6B) | English  |     6B      |    2048     |     28      |     16     |       4096       |   50400    |

EleutherAI's replication of the GPT-3 architecture
OpenAI의 GPT-3 `curie`의 alternative

- dataset : [Pile dataset](https://pile.eleuther.ai/) 82GB

### [GPT-NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox)

|                                          models                                          | language | # of params | max seq len | # of layers | # of heads | hidden state dim | vocab size |
|:----------------------------------------------------------------------------------------:|:--------:|:-----------:|:-----------:|:-----------:|:----------:|:----------------:|:----------:|
| #huggingface [`EleutherAI/gpt-neox-20b`](https://huggingface.co/EleutherAI/gpt-neox-20b) | English  |     20B     |    2048     |     44      |     64     |       6144       |   50432    |

EleutherAI's replication of the GPT-3 architecture

- dataset : [Pile dataset](https://pile.eleuther.ai/) 82GB

### [OPT](https://huggingface.co/docs/transformers/model_doc/opt)

|                                    models                                    | language | # of params | max seq len | # of layers | # of heads | hidden state dim | vocab size |
|:----------------------------------------------------------------------------:|:--------:|:-----------:|:-----------:|:-----------:|:----------:|:----------------:|:----------:|
| #huggingface [`facebook/opt-125m`](https://huggingface.co/facebook/opt-125m) | English  |    125M     |    2048     |     12      |     12     |       768        |   50272    |
| #huggingface [`facebook/opt-350m`](https://huggingface.co/facebook/opt-350m) | English  |    350M     |    2048     |     24      |     16     |       1024       |   50272    |
| #huggingface [`facebook/opt-1.3b`](https://huggingface.co/facebook/opt-1.3b) | English  |    1.3B     |    2048     |     24      |     32     |       2048       |   50272    |
| #huggingface [`facebook/opt-2.7b`](https://huggingface.co/facebook/opt-2.7b) | English  |    2.7B     |    2048     |     32      |     32     |       2560       |   50272    |
| #huggingface [`facebook/opt-6.7b`](https://huggingface.co/facebook/opt-6.7b) | English  |    6.7B     |    2048     |     32      |     32     |       4096       |   50272    |
|  #huggingface [`facebook/opt-13b`](https://huggingface.co/facebook/opt-13b)  | English  |     13B     |    2048     |     40      |     40     |       5120       |   50272    |
|  #huggingface [`facebook/opt-30b`](https://huggingface.co/facebook/opt-30b)  | English  |     30B     |    2048     |     48      |     56     |       7168       |   50272    |
|  #huggingface [`facebook/opt-66b`](https://huggingface.co/facebook/opt-66b)  | English  |     66B     |    2048     |     64      |     72     |       9216       |   50272    |

Meta AI에서 만든 오픈소스 GPT-3

- paper : [[2022]OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)

## Korean

### KoGPT

|                                   models                                   | language | # of params | max seq len | # of layers | # of heads | hidden state dim | vocab size |
|:--------------------------------------------------------------------------:|:--------:|:-----------:|:-----------:|:-----------:|:----------:|:----------------:|:----------:|
| #huggingface [`kakaobrain/kogpt`](https://huggingface.co/kakaobrain/kogpt) |  Korean  |     6B      |    2048     |     28      |     16     |       4096       |   64512    |

- dataset : ryan dataset 1.5B

kakao brain에서 만든 한국어 오픈소스 GPT-3

### KoGPT2

|                                models                                 | language | # of params | max seq len | # of layers | # of heads | hidden state dim | vocab size |
|:---------------------------------------------------------------------:|:--------:|:-----------:|:-----------:|:-----------:|:----------:|:----------------:|:----------:|
| #huggingface [`skt/kogpt2-base-v2`](https://github.com/SKT-AI/KoGPT2) |  Korean  |    125M     |    1024     |     12      |     12     |       768        |   51200    |

skt에서 만든 한국어 오픈소스 GPT-2

## Multilingual

### BLOOM

|                                        models                                        |                       language                        | # of params | max seq len | # of layers | # of heads | hidden state dim | vocab size |
|:------------------------------------------------------------------------------------:|:-----------------------------------------------------:|:-----------:|:-----------:|:-----------:|:----------:|:----------------:|:----------:|
| #huggingface [`bigscience/bloom-560m`](https://huggingface.co/bigscience/bloom-560m) | multilingual(46 languages + 13 programming languages) |    560M     |    2048     |     24      |     16     |       1024       |   250680   |
|  #huggingface [`bigscience/bloom-1b1`](https://huggingface.co/bigscience/bloom-1b1)  | multilingual(46 languages + 13 programming languages) |    1.1B     |    2048     |     24      |     16     |       1536       |   250680   |
|  #huggingface [`bigscience/bloom-1b7`](https://huggingface.co/bigscience/bloom-1b7)  | multilingual(46 languages + 13 programming languages) |    1.7B     |    2048     |     24      |     16     |       2048       |   250680   |
|   #huggingface [`bigscience/bloom-3b`](https://huggingface.co/bigscience/bloom-3b)   | multilingual(46 languages + 13 programming languages) |     3B      |    2048     |     30      |     32     |       2560       |   250680   |
|  #huggingface [`bigscience/bloom-7b1`](https://huggingface.co/bigscience/bloom-7b1)  | multilingual(46 languages + 13 programming languages) |    7.1B     |    2048     |     30      |     32     |       4096       |   250680   |
|      #huggingface [`bigscience/bloom`](https://huggingface.co/bigscience/bloom)      | multilingual(46 languages + 13 programming languages) |    176B     |    2048     |     70      |    112     |      14336       |   250680   |

OpenAI GPT-3 `davinci`의 alternative