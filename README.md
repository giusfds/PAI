# PAI - Segmentacao e classificacao mamografica

Projeto inicial para o trabalho pratico de Processamento e Analise de Imagens.

## Datasets

As imagens estão disponíveis em:

- [LCC](https://www.dropbox.com/scl/fi/sn225aaabb3k8dwmr368v/LCC.zip?rlkey=ldjmuou1bivxhqo7crt4wls8j&st=5udijp0k&dl=0)
- [LMLO](https://www.dropbox.com/scl/fi/yrd803iq7c2mfyt9x28tq/LMLO.zip?rlkey=dhtw3qvac492s6idye3r5b69u&st=2f3x8fa7&dl=0)
- [RCC](https://www.dropbox.com/scl/fi/lnt4g69bz9iedi7b43uxg/RCC.zip?rlkey=mg7wmpucj9yz3kg084keal702&st=jgqevx3u&dl=0)
- [RMLO](https://www.dropbox.com/scl/fi/yu5ntcjcis2qbwhw)

## Pré-Requisitos e Configuração de Ambiente

- Recomendado: Python 3.8 a 3.11 (64-bit). Verifique a versão com `python --version`.
- `pip` atualizado: `python -m pip install --upgrade pip`.
- Espaço em disco suficiente para datasets e modelos (vários GB).
- GPU (opcional): para treino mais rápido instale uma versão do PyTorch compatível com sua CUDA (veja https://pytorch.org/).

> Recomenda-se usar um ambiente virtual (`venv` ou `conda`) para isolar dependências:

- Windows (PowerShell):

    ```
    python -m venv venv
    venv\Scripts\Activate.ps1
    ```

- Windows (cmd):

    ```
    python -m venv venv
    venv\Scripts\activate
    ```

- Linux / macOS:

    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

- Após ativar o ambiente virtual, instale as dependências:

    ```
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

> Se pretende usar GPU, instale o PyTorch com suporte CUDA conforme as instruções oficiais em https://pytorch.org/ antes de instalar o restante das dependências.

## Arquitetura

- `main.py`: aplicacao grafica Tkinter em arquivo unico, conforme o enunciado.
- `Dataset/RCC`: base local com classes `D`, `E`, `F`, `G` na vista `right + CC`.
- `models/`: pesos treinados gerados localmente, ignorados pelo Git.
- `outputs/`: espaco para resultados, figuras e logs gerados localmente.

Dentro de `main.py`, o codigo esta separado em blocos:

- descoberta do dataset e regra de treino/teste por numeracao multipla de 4;
- leitura PNG/TIFF e normalizacao de imagens 8/16 bits;
- segmentacao automatica por limiar de Otsu, morfologia e maior componente;
- data augmentation por rotacoes `-20`, `-10`, `0`, `10`, `20`;
- classificacao binaria e de 4 classes com ResNet-18 e EfficientNet-B0;
- metricas exigidas no enunciado;
- Grad-CAM;
- interface grafica com visualizacao, zoom, treino, teste e classificacao.

## Como Executar

1. Ative o ambiente virtual criado anteriormente.

2. Instale as dependências, se ainda não o fez:

    ```
    pip install -r requirements.txt
    ```

3. Prepare os datasets: 
    - Baixe e extraia os arquivos nas pastas indicadas no projeto (veja seção "Datasets"). Garanta que a estrutura esteja em `Dataset/` conforme esperado pelo `main.py`.

4. Execute a aplicação:

    ```
    python main.py
    ```

- Na primeira execução o PyTorch pode baixar pesos pré-treinados (internet necessária).
- Os modelos treinados serão salvos em `models/` e os resultados em `outputs/`.

Dica: se encontrar erros relacionados a versões do PyTorch/CUDA, verifique a versão do CUDA instalada e reinstale o PyTorch com a combinação correta (veja https://pytorch.org/get-started/locally/).

## Observacoes de entrega

Antes de entregar, preencha no topo de `main.py` os nomes, matriculas, curso e campus dos
integrantes. Nao inclua `Dataset/`, `models/` ou `outputs/` no ZIP final.
