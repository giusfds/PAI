# PAI - Segmentacao e classificacao mamografica

Projeto inicial para o trabalho pratico de Processamento e Analise de Imagens.

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

## Como executar

Instale as dependencias em um ambiente Python:

```bash
pip install -r requirements.txt
```

Execute:

```bash
python main.py
```

Na primeira execucao, o PyTorch pode baixar pesos pre-treinados da ResNet/EfficientNet.
Depois do treino, os modelos ficam salvos em `models/` para evitar retreinamento durante
a apresentacao.

## Observacoes de entrega

Antes de entregar, preencha no topo de `main.py` os nomes, matriculas, curso e campus dos
integrantes. Nao inclua `Dataset/`, `models/` ou `outputs/` no ZIP final.
