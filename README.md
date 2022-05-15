# Framework Professor-Aluno

[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![Python](https://img.shields.io/badge/python-3.8.3-blue)](https://www.python.org/)
[![Tensorflow](https://img.shields.io/badge/tensorflow-2.2.0-orange)](https://www.tensorflow.org/)

> ⚠️ Implementação do **Framework Professor-Aluno** aplicado ao desafio **Cart-Pole**, usando **python**, **tensorflow** e **gym**. Adicionalmente, os agentes DQN foram adaptados da implementação disponibilizada por [**_Keon_**](https://github.com/keon/deep-q-learning). Todos os detalhes do ambiente, do framework e dos resultados obtidos estão descritos no [**_relatório_**](Professor_Aluno.pdf) do projeto.

## Overview

Este projeto aborda a implementação do **Framework Professor-Aluno** e investiga seus efeitos sobre o treinamento de um agente para resolver o jogo Cart-Pole. Ambos os agentes, professor e aluno, utilizam-se do algoritmo de aprendizado por reforço conhecido como Q-Learning, combinado a uma rede neural profunda, uma abordagem conhecida como DQN. Dos algoritmos presentes no framework, três foram investigados, sendo eles: **early advising**, **importance advising** e **mistake correcting**. Esses métodos foram utilizados para treinar um novo agente, e comparados ao treinamento de um agente sem a utilização do framework. Por fim, algumas variações de parâmetros do framework também foram investigadas.

No framework para aprendizado por reforço, o agente referido como **_aluno_**, aprende a realização de uma tarefa, enquanto outro agente, conhecido como **_professor_**, tem a possibilidade de sugerir uma ação para cada estado encontrado pelo aluno, como representado na figura:

![treinamento](/images/treinamento.jpeg)

O **CartPole-v1** é um ambiente simples que consiste em um mastro localizado sobre um carrinho móvel. A cada turno, o sistema aplica uma força sobre o mastro com o intuito de desequilibrá-lo. O agente deve então tentar reequilibrar o mastro movendo o carrinho para direita ou para esquerda.

![animacao](/images/animacao.gif)

A cada turno que o mastro passa em pé, o jogador recebe 1 ponto, e a recompensa máxima para vencer uma rodada é de 500 pontos. No entanto, o episódio acaba caso o mastro se mova mais que 15 graus, para qualquer lado, ou caso o carrinho se mova mais do que 2,4 unidades do centro do mapa.

## Instalação

Crie um ambiente e ative-o, em seguida clone o repositório e instale os requerimentos utilizando os seguintes comandos:

```Bash
git clone https://github.com/eduardoatr/Professor_Aluno.git
cd Professor_Aluno
pip install -r requirements.txt
```

## Execução

Execute o script **_cartpole.py_**, fornecendo como parâmetro o método de recomendação utilizado no framework, como mostrado no exemplo a seguir:

```Bash
python cartpole.py "mistake"
```

As opções de recomendação são: **early**, **importance** e **mistake**. Adicionalmente, a flag **_--show_** pode ser passada como parâmetro para que uma animação mostrando cada execução do ambiente seja renderizada. Por fim, outras configurações do método podem ser modificadas passando-as por parâmetro. Use o comando de ajuda para consultar as alterações possíveis da seguinte maneira:

```Bash
python cartpole.py --help
```

Para mais informações, consulte o [**_relatório_**](Professor_Aluno.pdf).
