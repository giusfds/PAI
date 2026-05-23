# PAI - Segmentação e Classificação Mamográfica

Projeto inicial para o trabalho pratico de Processamento e Analise de Imagens.

## Datasets

As imagens estão disponíveis em:

- [RCC](https://www.dropbox.com/scl/fi/lnt4g69bz9iedi7b43uxg/RCC.zip?rlkey=mg7wmpucj9yz3kg084keal702&st=jgqevx3u&dl=0)

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

- `src/main.py`: aplicação gráfica Tkinter em arquivo unico, conforme o enunciado.
- `dataset/RCC`: base local com classes `D`, `E`, `F`, `G` na vista `right + CC`.
- `models/`: pesos treinados gerados localmente, ignorados pelo Git.
- `outputs/`: espaco para resultados, figuras e logs gerados localmente.

Dentro de `src/main.py`, o codigo esta separado em blocos:

- descoberta do dataset e regra de treino/teste por numeração multipla de 4;
- leitura PNG/TIFF e normalização de imagens 8/16 bits;
- segmentação automatica por limiar de Otsu, morfologia e maior componente;
- data augmentation por rotacoes `-20`, `-10`, `0`, `10`, `20`;
- classificação binaria e de 4 classes com ResNet-18 e EfficientNet-B0;
- métricas exigidas no enunciado;
- Grad-CAM;
- interface gráfica com visualização, zoom, treino, teste e classificação.

## Como Executar

1. Ative o ambiente virtual criado anteriormente.

2. Instale as dependências, se ainda não o fez:

    ```
    pip install -r requirements.txt
    ```

3. Prepare os datasets: 
    - Baixe e extraia os arquivos nas pastas indicadas no projeto (veja seção "Datasets"). Garanta que a estrutura esteja em `dataset/` conforme esperado pelo `src/main.py`.

4. Execute a aplicação:

    ```
    python src/main.py
    ```

- Na primeira execução o PyTorch pode baixar pesos pré-treinados (internet necessária).
- Os modelos treinados serão salvos em `models/` e os resultados em `outputs/`.

Dica: se encontrar erros relacionados a versões do PyTorch/CUDA, verifique a versão do CUDA instalada e reinstale o PyTorch com a combinação correta (veja https://pytorch.org/get-started/locally/).

## Observacoes de entrega

Antes de entregar, preencha no topo de `src/main.py` os nomes, matriculas, curso e campus dos
integrantes. Nao inclua `Dataset/`, `models/` ou `outputs/` no ZIP final.
