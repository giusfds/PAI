# Arquitetura do Sistema de Classificação de Densidade Mamária

## Visão Geral
A arquitetura organiza o sistema em **7 classes principais**, cada uma responsável por um domínio específico do pipeline de processamento.

## Pipeline de Processamento

```
Imagem Mamográfica
        |
        V
ImageProcessor.load_image()
        |
        V
ImageProcessor.preprocess()
        |
        V
ImageProcessor.segment_breast()
        |
        V
Segmentação + Máscara
        |
        V
DatasetManager (organiza para treino/teste)
        |
        V
CNNTrainer.train() / CNNTrainer.build_model()
        |
        V
Modelo Treinado
        |
        V
Predictor.load_model() → Predictor.evaluate()
        |
        V
Classificação + Métricas
        |
        V
GradCAMGenerator.generate()
        |
        V
Mapa de Interpretação
        |
        V
MammographyApp (GUI)
        |
        V
Exibição ao Usuário
```

## 1. MammogramImage: Entidade de Dados

**Responsabilidade**: Representar uma imagem mamográfica e seus estados durante o pipeline.

**Attributes**:
- `path`: Caminho do arquivo
- `filename`: Nome do arquivo
- `original`: Imagem original normalizada [0,1]
- `preprocessed`: Imagem pré-processada
- `mask`: Máscara binária da mama
- `segmented`: Imagem com máscara aplicada
- `gradcam`: Visualização Grad-CAM
- `label`: Classe real (ground truth)
- `prediction`: Classe prevista
- `confidence`: Confiança da predição
- `width`, `height`, `bit_depth`: Metadados
- Status flags: `loaded`, `processed`, `segmented_ready`

**Uso**:
```python
image_obj = MammogramImage(path)
ImageProcessor.load_image(image_obj)
ImageProcessor.segment_breast(image_obj)
# image_obj.preprocessed e image_obj.segmented estão prontos
```

## 2. ImageProcessor: Processamento de Imagens

**Responsabilidade**: Todas as operações de leitura, pré-processamento e segmentação.

**Métodos principais**:
- `load_image(image_obj)`: Carrega PNG/TIFF e normaliza
- `preprocess(image_obj)`: Pré-processamento
- `segment_breast(image_obj)`: Segmenta mama e aplica máscara
- `to_display_image(array)`: Converte para PIL para visualização
- `_otsu_threshold(gray)`: Calcula limiar de Otsu
- `_largest_component_fallback(mask)`: Encontra maior componente (sem OpenCV)

**Features**:
- Suporte a múltiplos formatos: PNG, TIFF
- Tratamento automático de EXIF
- Normalização de valores [0, 1]
- Otimização com OpenCV (com fallback)

**Uso**:
```python
processor = ImageProcessor()
processor.load_image(image_obj)
processor.segment_breast(image_obj)
display = processor.to_display_image(image_obj.segmented)
```

## 3. DatasetManager: Gerenciamento do Dataset

**Responsabilidade**: Descobrir, indexar e dividir dataset.

**Métodos principais**:
- `discover_dataset(dataset_dir)`: Descobre imagens recursivamente
- `summarize_records(records)`: Gera estatísticas
- `_natural_number_from_name(path)`: Extrai número para split

**Regra de Split**:
```python
if image_number % 4 == 0:
    split = "test"   # 25% para teste
else:
    split = "train"  # 75% para treino
```

**Classe Auxiliar - ImageRecord**:
```python
@dataclass
class ImageRecord:
    path: Path
    class_name: str          # "D", "E", "F", "G"
    class_index: int         # 0, 1, 2, 3
    number: int              # número extraído do nome
    split: str               # "train" ou "test"
    
    @property
    def binary_index(self) -> int:
        return 0 if class_name in {"D", "E"} else 1
```

**Classe Auxiliar - MammographyDataset**:
- Dataset PyTorch para treino
- Implementa augmentação (rotação) apenas em treino
- Normalização automática com pesos ImageNet

**Uso**:
```python
records = DatasetManager.discover_dataset(Path("Dataset/RCC"))
summary = DatasetManager.summarize_records(records)
print(summary)
```

## 4. CNNTrainer: Treinamento de Modelos

**Responsabilidade**: Construir, treinar e salvar redes neurais.

**Arquiteturas Suportadas**:
- ResNet-18 (pesos ImageNet)
- EfficientNet-B0 (pesos ImageNet)

**Métodos principais**:
- `build_model(model_name, num_classes)`: Constrói modelo com pesos pré-treinados
- `train(...)`: Executa treino completo
- `_model_path(...)`: Gera caminho para salvar modelo

**Características**:
- Transfer learning: camadas congeladas exceto FC final
- Adaptável: 2 classes (binary) ou 4 classes (four)
- GPU automático: detecta CUDA se disponível
- Otimizador Adam com learning rate customizável

**Uso**:
```python
path = CNNTrainer.train(
    records=records,
    model_name="resnet18",
    task="binary",
    segmented=True,
    epochs=10,
    batch_size=8,
    learning_rate=0.001,
    progress=print  # callback para progresso
)
```

## 5. Predictor: Inferência e Avaliação

**Responsabilidade**: Carregar modelos e fazer predições.

**Métodos principais**:
- `load_model(model_name, task, segmented)`: Carrega modelo treinado
- `evaluate(records, ...)`: Avalia no conjunto de teste
- `_confusion_matrix(...)`: Calcula matriz de confusão
- `_binary_metrics(...)`: Métricas para classificação binária
- `_multiclass_sensitivity_specificity(...)`: Métricas multiclasse

**Métricas Calculadas**:

**Binário**:
- Sensibilidade (Recall/Verdadeiro Positivo)
- Especificidade
- Precisão
- Acurácia
- F1-Score

**Multiclasse**:
- Sensibilidade Média
- Especificidade Média

**Uso**:
```python
results = Predictor.evaluate(
    records=records,
    model_name="resnet18",
    task="binary",
    segmented=True,
    batch_size=8
)
print(results)
```

## 6. GradCAMGenerator: Interpretabilidade

**Responsabilidade**: Gerar mapas Grad-CAM para interpretação das decisões.

**Métodos principais**:
- `generate(model_name, task, segmented, image_path)`: Gera Grad-CAM
- `_preprocess_for_model(...)`: Pré-processa imagem para modelo

**Funcionalidade**:
- Captura ativações da última camada convolucional
- Calcula gradientes com relação ao output
- Aplica ponderação por gradiente médio
- Normaliza heatmap em [0, 1]
- Sobrepõe com alpha-blending na imagem original
- Colormap Jet (OpenCV) ou Red (fallback)

**Saída**: 
- Rótulo de classificação
- Imagem com heatmap sobreposto

**Uso**:
```python
label, overlay = GradCAMGenerator.generate(
    model_name="resnet18",
    task="binary",
    segmented=True,
    image_path=Path("image.png")
)
overlay.show()
```

## 7. MammographyApp: Interface Gráfica

**Responsabilidade**: Orquestrar interações do usuário e chamar serviços.

**Responsabilidades da GUI**:
- Receber input do usuário (botões, menus)
- Chamar métodos das classes de serviço
- Atualizar visualização
- Exibir mensagens de progresso/erro
- Gerenciar estado da aplicação

**Métodos Principais**:
- `open_image()`: Abre diálogo e carrega imagem
- `segment_current()`: Segmenta imagem visível
- `train_selected()`: Inicia treino em thread
- `evaluate_selected()`: Avalia modelo
- `run_grad_cam()`: Gera e exibe Grad-CAM
- `load_dataset()`: Carrega dataset do diretório

**Threading**:
- Operações pesadas executam em threads daemon
- Interface permanece responsiva
- Progresso exibido em log em tempo real

## Arquitetura em Camadas

```
┌─────────────────────────────────────────┐
│    Apresentação (MammographyApp)        │  ← Interface com Usuário
├─────────────────────────────────────────┤
│  Orquestração (Fluxo de Chamadas)       │
├─────────────────────────────────────────┤
│  Serviços de Negócio                    │
│  ├─ ImageProcessor                       │  ← Processamento
│  ├─ DatasetManager                       │  ← Dados
│  ├─ CNNTrainer                           │  ← Treino
│  ├─ Predictor                            │  ← Inferência
│  └─ GradCAMGenerator                     │  ← Interpretação
├─────────────────────────────────────────┤
│  Entidades (MammogramImage, ImageRecord)│  ← Dados
├─────────────────────────────────────────┤
│  Bibliotecas Externas                   │
│  ├─ PyTorch/Torchvision                  │  ← Deep Learning
│  ├─ Pillow                               │  ← Imagens
│  ├─ NumPy                                │  ← Computação
│  ├─ OpenCV (opcional)                    │  ← Processamento
│  └─ Tkinter                              │  ← GUI
└─────────────────────────────────────────┘
```

## Principais Características da Arquitetura

### 1. Separação de Responsabilidades
Cada classe tem uma responsabilidade única e bem definida:
- **MammogramImage**: Dados
- **ImageProcessor**: Processamento
- **DatasetManager**: Gerenciamento de dados
- **CNNTrainer**: Treinamento
- **Predictor**: Predição
- **GradCAMGenerator**: Interpretação
- **MammographyApp**: Apresentação

### 2. Redução de Acoplamento
- Classes usam interfaces simples e bem definidas
- Dados passados através de objetos estruturados
- Sem dependências circulares
- Fácil substituição de implementações

### 3. Modularidade em Arquivo Único
Mesmo mantendo tudo em um arquivo:
- Separação lógica clara
- Importações organizadas
- Métodos estáticos quando apropriado
- Sem duplicação de código

### 4. Pipeline Reprodutível
Sistema organizado em torno do fluxo de dados:
```
Imagem → Carregamento → Pré-processamento → Segmentação 
→ Classificação → Interpretação → Exibição
```

### 5. Tratamento de Erros
- Lazy imports para dependências opcionais
- Mensagens de erro descritivas
- Fallbacks para quando bibliotecas faltam (ex: sem OpenCV)
- Validação de entrada

### 6. Performance
- GPU automático quando disponível
- Batch processing eficiente
- Threaded GUI (não bloqueia interface)
- Caching de modelos

## Como Estender

### Adicionar Novo Modelo
```python
# 1. Modificar CNNTrainer.build_model()
if model_name == "resnet50":
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    # ... resto do código

# 2. Adicionar ao combobox da GUI
ttk.Combobox(panel, values=["resnet18", "resnet50", "efficientnet_b0"], ...)
```

### Adicionar Nova Métrica
```python
# 1. Adicionar método ao Predictor
@staticmethod
def _auc_score(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)

# 2. Usar em Predictor.evaluate()
```

### Adicionar Novo Tipo de Augmentação
```python
# 1. Modificar MammographyDataset.__getitem__()
if elastic_deform:
    image = elastic_transform(image)
```

## Fluxo de Execução Principal

```python
# 1. Usuário inicia aplicação
app = MammographyApp()

# 2. GUI carrega dataset automaticamente
load_dataset(Path("Dataset/RCC"))
→ DatasetManager.discover_dataset()
→ Exibe resumo

# 3. Usuário clica "Abrir imagem"
open_image()
→ MammogramImage(path)
→ ImageProcessor.load_image()
→ Exibe imagem

# 4. Usuário clica "Segmentar"
segment_current()
→ ImageProcessor.segment_breast()
→ Exibe segmentada

# 5. Usuário clica "Treinar"
train_selected()
→ CNNTrainer.train()
→ Salva modelo

# 6. Usuário clica "Avaliar"
evaluate_selected()
→ Predictor.evaluate()
→ Exibe métricas

# 7. Usuário clica "Grad-CAM"
run_grad_cam()
→ GradCAMGenerator.generate()
→ Exibe com interpretação
```
