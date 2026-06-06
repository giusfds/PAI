# Relatório Técnico — PAI: Segmentação e Classificação Mamográfica

## 1. Identificação do Projeto

**Nome:** PAI — Processamento e Análise de Imagens  
**Descrição:** Sistema de segmentação automática e classificação BIRADS de imagens mamográficas com interface gráfica desktop.  
**Disciplina:** Processamento e Análise de Imagens  
**Arquivo principal:** `src/main.py` (arquivo único, ~2.166 linhas)

---

## 2. Objetivo

Desenvolver uma aplicação desktop capaz de:

1. Carregar um dataset de mamografias organizadas por classes BIRADS.
2. Segmentar automaticamente a região da mama em cada imagem usando limiarização de Otsu e operações morfológicas.
3. Treinar redes neurais convolucionais profundas (ResNet-18, EfficientNet) via Transfer Learning para classificar as imagens em modo **binário** (BIRADS I+II vs. III+IV) ou **4 classes** (BIRADS I, II, III, IV).
4. Avaliar o modelo com métricas padronizadas (acurácia, precisão, sensibilidade, especificidade, F1, matriz de confusão).
5. Visualizar a ativação do modelo com **Grad-CAM**.
6. Expor todas as funcionalidades em uma interface gráfica intuitiva (Tkinter).

---

## 3. Dataset

### 3.1 Origem e Estrutura

O dataset utilizado é denominado **RCC** (Right CC view — projeção crânio-caudal direita). As imagens estão organizadas em quatro subpastas, cada uma correspondendo a uma classe BIRADS:

| Pasta              | Classe BIRADS | Índice numérico |
|--------------------|---------------|-----------------|
| `D + right + CC`   | BIRADS I      | 0               |
| `E + right + CC`   | BIRADS II     | 1               |
| `F + right + CC`   | BIRADS III    | 2               |
| `G + right + CC`   | BIRADS IV     | 3               |

### 3.2 Volume de Imagens

- **Total de imagens:** 1.256
- **Imagens por classe:** 314 (dataset balanceado)
- **Formatos suportados:** `.png`, `.tif`, `.tiff`
- **Profundidade de bits:** imagens de 8 bits e 16 bits (ambas suportadas)

### 3.3 Divisão Treino/Teste

A divisão é determinada pelo número extraído do nome do arquivo (ex.: `d_right_cc (12).png` → número 12):

- **Teste:** número divisível por 4 (`número % 4 == 0`)
- **Treino:** todos os demais

Isso resulta em aproximadamente **75% treino / 25% teste** por classe.

Com **Data Augmentation** (5 ângulos de rotação por imagem), o conjunto de treino efetivo alcança ~`4 × 5 × 235 = 4.700` amostras.

---

## 4. Pré-processamento e Segmentação

### 4.1 Leitura e Normalização

- As imagens são abertas em modo escala de cinza (`PIL.Image` convertida para `"L"`).
- Normalizadas para o intervalo `[0.0, 1.0]` por min-max: `(img - min) / (max - min)`.
- Redimensionadas para `224 × 224` pixels (tamanho padrão de entrada das redes ImageNet).
- Convertidas de 2D (grayscale) para 3 canais RGB por replicação (`np.stack([img]*3, axis=-1)`).

### 4.2 Pipeline de Segmentação Mamária

O `SegmentationProcessor` executa os seguintes passos:

1. **Downscale para 50%** — reduz o custo computacional do Otsu.
2. **Limiarização de Otsu** (`cv2.THRESH_OTSU`) com possibilidade de ajuste manual por `threshold_offset` (parâmetro configurável, padrão = 10).
3. **Maior componente conectada** (`cv2.connectedComponentsWithStats`, conectividade 8) — elimina artefatos e ruídos menores.
4. **Fechamento morfológico** (`cv2.MORPH_CLOSE`, kernel elíptico) — fecha buracos internos na máscara.
5. **Abertura morfológica** (`cv2.MORPH_OPEN`) — remove pequenas protuberâncias externas.
6. **Upsample da máscara** (`cv2.INTER_NEAREST`) de volta à resolução original.
7. **Aplicação da máscara** — pixels fora da mama são zerados na imagem original.
8. **Crop opcional** (`cv2.boundingRect`) — recorta a imagem ao bounding box da mama (ROI).

**Parâmetros configuráveis pelo usuário:**

| Parâmetro              | Padrão | Descrição                                           |
|------------------------|--------|-----------------------------------------------------|
| `threshold_offset`     | 10     | Ajuste sobre o limiar de Otsu                       |
| `closing_iterations`   | 1      | Iterações de fechamento morfológico                 |
| `kernel_size`          | 25     | Tamanho do elemento estruturante elíptico           |
| `crop`                 | False  | Recortar imagem ao bounding box da região segmentada|

### 4.3 Data Augmentation

Aplicado somente no conjunto de treino. Para cada imagem, são geradas 5 versões com rotações:

```
ROTATIONS = [-20°, -10°, 0°, 10°, 20°]
```

A rotação é feita via `scipy.ndimage.rotate` (modo `constant`, cval=0) e também via `torchvision.transforms.functional.affine` sobre tensores.

---

## 5. Modelos de Classificação

### 5.1 Abordagem: Transfer Learning (Feature Extraction)

Todos os modelos são carregados com **pesos pré-treinados no ImageNet** (`torchvision.models`). Todas as camadas são **congeladas** (`requires_grad = False`). Apenas a **última camada classificadora** é substituída e treinada.

### 5.2 Modelos Disponíveis

| ID              | Arquitetura      | Camada de saída substituída       |
|-----------------|------------------|-----------------------------------|
| `resnet18`      | ResNet-18        | `model.fc`                        |
| `efficientnet_b0` | EfficientNet-B0 | `model.classifier`               |
| `efficientnet_b1` | EfficientNet-B1 | `model.classifier`               |
| `efficientnet_b2` | EfficientNet-B2 | `model.classifier`               |
| `efficientnet_b3` | EfficientNet-B3 | `model.classifier`               |

A nova cabeça classificadora para todos os modelos é:

```python
nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(in_features, num_classes)
)
```

### 5.3 Modo Binário vs. 4 Classes

| Modo             | Classes de saída | Mapeamento                            |
|------------------|------------------|---------------------------------------|
| Binário          | 2                | BIRADS I+II → 0, BIRADS III+IV → 1   |
| 4 classes        | 4                | BIRADS I→0, II→1, III→2, IV→3        |

### 5.4 Configurações de Treinamento

| Parâmetro           | Padrão   | Faixa disponível na GUI                        |
|---------------------|----------|------------------------------------------------|
| Épocas              | 8        | 1 a 200                                        |
| Batch size          | 32       | 1 a 256                                        |
| Taxa de aprendizado | 0.001    | 0.0001 a 10 (11 opções pré-definidas)          |
| Dropout             | 0.3      | 0.0 a 10.0 (incremento 0.05)                  |
| Otimizador          | Adam     | —                                              |
| Função de perda     | CrossEntropyLoss | —                                      |

### 5.5 Infraestrutura de Treinamento

- **Dispositivo:** CUDA (GPU) se disponível, senão CPU.
- **DataLoader:** `pin_memory=True` (com CUDA), `num_workers` ajustado automaticamente (até 4), `prefetch_factor=3`.
- **Salvamento:** checkpoint `.pth` com `state_dict`, `model_type`, `binary_classification`, `dropout_rate`.
- **Cancelamento:** suportado por flag `cancel_requested` verificada a cada batch e época.
- **Execução:** em thread separada (`threading.Thread`) para não bloquear a GUI.

---

## 6. Grad-CAM

O `GradCAMProcessor` utiliza a biblioteca `pytorch-grad-cam` para gerar mapas de ativação de gradiente.

- **Camada alvo:**
  - ResNet-18: `model.layer4[-1]`
  - EfficientNet: `model.features[-1]`
- **Saída:** sobreposição colorida (heatmap) sobre a imagem processada, usando `show_cam_on_image`.
- Exibido no painel "Grad-CAM" da interface após cada classificação.

---

## 7. Métricas de Avaliação

### 7.1 Classificação Binária

Calculadas via `sklearn.metrics`:

| Métrica       | Fórmula                         |
|---------------|---------------------------------|
| Acurácia      | `(TP+TN)/(TP+TN+FP+FN)`        |
| Precisão      | `TP/(TP+FP)`                    |
| Sensibilidade | `TP/(TP+FN)` (recall)           |
| Especificidade| `TN/(TN+FP)`                    |
| F1            | `2·(P·R)/(P+R)`                |
| Matriz de Confusão | 2×2                       |
| Relatório     | `classification_report`         |

### 7.2 Classificação Multiclasse (4 classes)

Adicionalmente às métricas acima (com `average='macro'`):

| Métrica              | Descrição                                        |
|----------------------|--------------------------------------------------|
| Sensibilidade por classe | `recall_score(average=None)`               |
| Sensibilidade média  | Média das sensibilidades por classe              |
| Especificidade média | Calculada manualmente por OvR (One vs. Rest)     |
| Matriz de Confusão   | 4×4                                              |

---

## 8. Interface Gráfica (GUI)

Implementada com **Tkinter** e **ttk** (Python padrão). Layout de duas colunas: painel lateral (sidebar) + área de conteúdo principal com abas.

### 8.1 Aba "Treinamento"

- **Gráficos em tempo real:** Loss e Acurácia (treino e validação), atualizados ao fim de cada época via `matplotlib` + `FigureCanvasTkAgg`.
- **Barra de progresso:** por batch e por época com percentual.
- **Matriz de confusão:** renderizada com `ConfusionMatrixDisplay` após treinamento.
- **Painel de métricas finais:** todos os valores acima exibidos ao fim do treino.
- **Área de logs:** texto rolável com mensagens de log integrado via handler customizado (`TkinterLogHandler` + fila `queue.Queue`).

### 8.2 Aba "Classificação"

- **4 painéis de imagem:** Imagem original, Máscara, Imagem segmentada, Grad-CAM.
- **Barras de probabilidade:** uma barra de progresso por classe BIRADS com percentual.
- **Zoom:** controle deslizante de 0.2× a 3.0× com botão "Zoom 100%".
- **Importar modelo:** carrega arquivo `.pth` salvo anteriormente.

### 8.3 Sidebar (contexto-dependente)

- **Treinamento:** seleção de dataset, modelo, batch size, learning rate, épocas, dropout, segmentação, parâmetros morfológicos, botões treinar/cancelar/exportar.
- **Classificação:** importar modelo, selecionar imagem, botão classificar.

---

## 9. Arquitetura de Software

O projeto é estruturado em **classes com responsabilidades bem definidas** dentro de um único arquivo `src/main.py`:

| Classe / Componente      | Responsabilidade                                              |
|--------------------------|---------------------------------------------------------------|
| `DatasetManager`         | Descoberta recursiva de arquivos, parsing de nomes, split     |
| `ImageManager`           | Leitura, normalização, redimensionamento, conversão PIL↔numpy |
| `SegmentationProcessor`  | Pipeline de segmentação Otsu + morfologia + maior componente  |
| `DataAugmentationProcessor` | Rotações para aumento de dados                            |
| `MammographyDataset`     | `torch.utils.data.Dataset`: encapsula amostras + transforms  |
| `MetricsCalculator`      | Cálculo de métricas binárias e multiclasse com sklearn        |
| `TrainingManager`        | Treinamento, validação, avaliação, salvar/carregar modelo      |
| `GradCAMProcessor`       | Geração e sobreposição de mapas Grad-CAM                      |
| `ApplicationService`     | Serviço centralizado: orquestra todos os componentes acima    |
| `GUI` (`tk.Tk`)          | Interface gráfica completa (sidebar + notebook + painéis)     |
| `TkinterLogHandler`      | Redireciona logs Python para o widget de texto da GUI         |

### 9.1 Dataclasses e Enums

```python
class SampleBiradsClass(StrEnum): D, E, F, G
class ModelType(StrEnum): RESNET18, EFFICIENTNET_B0..B3
class SetType(Enum): TRAIN, TEST
class SampleSide(Enum): LEFT, RIGHT
class SampleView(Enum): CC, MLO

@dataclass Sample          # imutável (frozen=True, slots=True)
@dataclass SegmentationConfig
@dataclass DatasetConfig
@dataclass TrainingConfig
@dataclass ClassificationResult
@dataclass SegmentationResult
@dataclass GradCAMResult
```

---

## 10. Tecnologias e Dependências

### 10.1 Linguagem

- **Python** (recomendado 3.8–3.11; testado também em 3.14)

### 10.2 Bibliotecas Principais

| Biblioteca          | Versão         | Uso                                                    |
|---------------------|----------------|--------------------------------------------------------|
| `torch`             | 2.11.0+cu128   | Framework de deep learning, tensores, autograd         |
| `torchvision`       | 0.26.0+cu128   | Modelos pré-treinados (ResNet, EfficientNet), transforms|
| `grad-cam`          | 1.5.5          | Grad-CAM via `pytorch_grad_cam`                        |
| `opencv-python`     | 4.13.0.92      | Otsu, morfologia, componentes conectadas, resize       |
| `Pillow`            | 12.2.0         | Leitura/escrita de PNG/TIFF, conversão para Tk          |
| `numpy`             | 2.4.6          | Operações matriciais, máscaras, normalização           |
| `scipy`             | 1.17.1         | `ndimage.rotate` para data augmentation               |
| `scikit-learn`      | 1.9.0          | Métricas de classificação, matriz de confusão          |
| `matplotlib`        | 3.10.9         | Gráficos de loss/acurácia, exibição na GUI via Tk      |
| `rich`              | 15.0.0         | Handler de log colorido no terminal (`RichHandler`)    |
| `tqdm`              | 4.67.3         | Barra de progresso (dependência indireta)              |
| `ttach`             | 0.0.3          | Test-time augmentation (dependência do grad-cam)       |

### 10.3 Interface Gráfica

| Componente       | Descrição                               |
|------------------|-----------------------------------------|
| `tkinter`        | Framework GUI (biblioteca padrão Python)|
| `tkinter.ttk`    | Widgets temáticos (Notebook, Combobox…) |
| `matplotlib` + `FigureCanvasTkAgg` | Gráficos embutidos no Tkinter |

### 10.4 Aceleração de Hardware

- **CUDA 12.8** — suporte a GPU NVIDIA para treinamento acelerado.
- Detecção automática: `torch.cuda.is_available()` → cai para CPU se GPU ausente.

### 10.5 Testes

| Biblioteca  | Versão | Uso                          |
|-------------|--------|------------------------------|
| `pytest`    | 9.0.3  | Framework de testes unitários|

---

## 11. Testes Automatizados

Arquivo: `tests/test_core.py`

| Teste                                        | Descrição                                                          |
|----------------------------------------------|--------------------------------------------------------------------|
| `test_load_image_normalizes_8_and_16_bit`    | Verifica normalização para `[0,1]` em imagens de 8 e 16 bits      |
| `test_natural_number_and_split_rule`         | Valida extração de número do nome e regra de split (mod 4)        |
| `test_discover_dataset_is_recursive`         | Dataset pode estar em subdiretórios aninhados                      |
| `test_largest_component_fallback_without_cv2`| Fallback puro NumPy da maior componente sem OpenCV                 |
| `test_segment_breast_fallback_without_cv2`   | Segmentação funciona mesmo com `cv2=None` (monkeypatched)         |

---

## 12. Estrutura de Diretórios

```
PAI/
├── src/
│   └── main.py              # Aplicação completa (único arquivo fonte)
├── tests/
│   └── test_core.py         # Testes unitários
├── dataset/
│   ├── D + right + CC/      # 314 imagens — BIRADS I
│   ├── E + right + CC/      # 314 imagens — BIRADS II
│   ├── F + right + CC/      # 314 imagens — BIRADS III
│   └── G + right + CC/      # 314 imagens — BIRADS IV
├── docs/
│   └── TP.pdf               # Enunciado do trabalho prático
├── models/                  # Pesos treinados (gerados localmente, no .gitignore)
├── outputs/                 # Resultados/figuras (gerados localmente, no .gitignore)
├── requirements.txt         # Dependências Python
├── README.md
└── LICENSE
```

---

## 13. Como Executar

```bash
# 1. Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate.ps1       # Windows PowerShell

# 2. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# 3. Executar aplicação
python src/main.py
```

> Para suporte a GPU, instalar PyTorch com CUDA antes do `requirements.txt`:
> consultar https://pytorch.org/get-started/locally/

---

## 14. Fluxo de Uso da Aplicação

```
[Selecionar Dataset] → [Configurar Treino] → [Treinar]
       ↓                                         ↓
  (pasta com subpastas             Gráficos Loss/Acurácia em tempo real
   por classe BIRADS)              Matriz de Confusão + Métricas ao fim
                                         ↓
                                  [Exportar Modelo (.pth)]
                                         ↓
                           [Aba Classificação] → [Importar Modelo]
                                         ↓
                                  [Selecionar Imagem]
                                         ↓
                              Original | Máscara | Segmentada | Grad-CAM
                                         ↓
                                  [Classificar Imagem]
                                         ↓
                           Resultado + Probabilidades por Classe
```

---

## 15. Decisões de Projeto Relevantes

- **Arquivo único:** requisito do enunciado. Todo o sistema (dados, modelos, métricas, GUI) em `src/main.py`.
- **Transfer Learning com feature extraction:** congela backbone para evitar overfitting no dataset relativamente pequeno; treina apenas a cabeça classificadora.
- **Segmentação antes do treino:** opção configurável — permite comparar resultados com e sem segmentação.
- **Split por numeração de arquivo:** reprodutível e determinístico; imagens múltiplas de 4 vão para teste.
- **Augmentation apenas no treino:** evita data leakage para a validação.
- **Thread separada para treino:** impede que o loop de treinamento bloqueie a event loop do Tkinter.
- **Cache de segmentação:** `_segmentation_cache` no `MammographyDataset` evita recomputar a máscara para o mesmo arquivo.
- **Checkpoint com metadados:** o `.pth` armazena `model_type`, `binary_classification` e `dropout_rate` além dos pesos, permitindo carregar o modelo sem precisar reconfigurar manualmente.
