"""
Trabalho Pratico - Processamento e Analise de Imagens
Segmentacao e classificacao de imagens mamograficas.

- MammogramImage: entidade de dados para uma imagem mamográfica
- ImageProcessor: operações de leitura, pré-processamento e segmentação
- DatasetManager: descoberta, carregamento e separação de dataset
- CNNTrainer: treinamento e salvamento de modelos
- Predictor: inferência e classificação
- GradCAMGenerator: geração de mapas de interpretação
- MammographyApp: interface gráfica
"""

from __future__ import annotations

import queue
import re
import threading
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

try:
    from PIL import Image, ImageOps, ImageTk
except ImportError:  # pragma: no cover
    raise SystemExit("Instale Pillow para executar a aplicacao: pip install pillow")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover
    raise SystemExit("Tkinter nao esta disponivel neste Python.")

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    models = None
    transforms = None


# ============================================================================
# CONSTANTES E CONFIG's
# ============================================================================

DATASET_FILEPATH: str = "dataset"
MODEL_FILEPATH: str = "models"

IMAGE_EXTENSIONS: set[str] = {".png", ".tif", ".tiff"}
CLASS_NAMES: list[str] = ["D", "E", "F", "G"]
CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}
BIRADS_LABELS: dict[str, str] = {
    "D": "BIRADS I",
    "E": "BIRADS II",
    "F": "BIRADS III",
    "G": "BIRADS IV",
}
ROTATION_ANGLES: list[int] = [-20, -10, 0, 10, 20]
DEFAULT_DATASET_DIR: Path = Path(DATASET_FILEPATH)
MODEL_DIR: Path = Path(MODEL_FILEPATH)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("pai")

# Constantes para evitar números mágicos
IMAGE_DEFAULT_SIZE: int = 224
NORMALIZE_MEAN: list[float] = [0.485, 0.456, 0.406]
NORMALIZE_STD: list[float] = [0.229, 0.224, 0.225]
MORPH_KERNEL_SIZE: tuple[int, int] = (7, 7)
MORPH_CLOSE_ITERATIONS: int = 2
OTSU_MIN_THRESHOLD: float = 0.03
CONNECTIVITY: int = 8

# Constantes da GUI
GEOMETRY: str = "1180x760"
ZOOM_DEFAULT: float = 1.0
GUI_DEFAULT_EPOCHS: int = 3
GUI_DEFAULT_BATCH: int = 8
GUI_DEFAULT_LR: float = 0.001


# ============================================================================
# MAMMOGRAM_IMAGE
# ============================================================================

class MammogramImage:
    """
    Representa uma única imagem mamográfica e todos os seus estados
    durante o pipeline de processamento.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path) if isinstance(path, str) else path
        self.filename = self.path.name

        # Estados da imagem
        self.original: np.ndarray | None = None
        self.preprocessed: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.segmented: np.ndarray | None = None
        self.gradcam: Image.Image | None = None

        # Resultados de classificação
        self.label: int | None = None
        self.prediction: int | None = None
        self.confidence: float | None = None

        # Metadados
        self.width: int | None = None
        self.height: int | None = None
        self.bit_depth: int | None = None

        # Status do pipeline
        self.loaded = False
        self.processed = False
        self.segmented_ready = False


@dataclass(frozen=True)
class ImageRecord:
    """Representa um registro de imagem descoberto no dataset."""
    path: Path
    class_name: str
    class_index: int
    number: int
    split: str

    @property
    def binary_index(self) -> int:
        return 0 if self.class_name in {"D", "E"} else 1

    @property
    def display_name(self) -> str:
        label = BIRADS_LABELS.get(self.class_name, self.class_name)
        return f"{self.path.name} | {self.class_name} ({label}) | {self.split}"

    def label_index(self, task: str) -> int:
        if task == "binary":
            return self.binary_index
        if task == "four":
            return self.class_index
        raise ValueError(f"Tarefa desconhecida: {task}")


# ============================================================================
# IMAGE_PROCESSOR
# ============================================================================

class ImageProcessor:
    """
    Responsável por todas as operações de leitura e processamento de imagens.

    Responsabilidades:
    - Carregar imagens PNG/TIFF
    - Converter profundidade de bits
    - Normalizar valores de pixel
    - Segmentar mama
    - Aplicar máscara
    - Preparar para exibição
    """

    @staticmethod
    def load_image(image_obj: MammogramImage) -> None:
        """Carrega a imagem do arquivo no objeto."""
        image = Image.open(image_obj.path)
        image = ImageOps.exif_transpose(image)
        if image.mode not in {"L", "I;16", "I", "F"}:
            image = image.convert("L")
        array = np.asarray(image).astype(np.float32)
        image_obj.bit_depth = ImageProcessor._bit_depth(image, array)
        max_value = float(np.max(array)) if array.size else 1.0
        if max_value > 0:
            array /= max_value
        image_obj.original = array
        image_obj.preprocessed = array.copy()
        image_obj.width = image.width
        image_obj.height = image.height
        image_obj.loaded = True

    @staticmethod
    def _bit_depth(image: Image.Image, array: np.ndarray) -> int:
        if image.mode in {"I;16", "I;16B", "I;16L"}:
            return 16
        if image.mode == "1":
            return 1
        if image.mode in {"I", "F"}:
            return 32
        if array.dtype == np.uint16:
            return 16
        return 8

    @staticmethod
    def preprocess(image_obj: MammogramImage) -> None:
        """Pré-processamento: normalização e preparação básica."""
        if image_obj.preprocessed is None:
            raise ValueError("Imagem não carregada.")
        image_obj.processed = True

    @staticmethod
    def segment_breast(image_obj: MammogramImage) -> None:
        """Segmenta a mama e aplica máscara."""
        if image_obj.preprocessed is None:
            raise ValueError("Imagem não pré-processada.")
        
        gray = image_obj.preprocessed
        threshold = max(ImageProcessor._otsu_threshold(gray), OTSU_MIN_THRESHOLD)
        mask = (gray > threshold).astype(np.uint8)

        if cv2 is not None:
            kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERATIONS)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=CONNECTIVITY)
            if count > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest = 1 + int(np.argmax(areas))
                mask = (labels == largest).astype(np.uint8)
        else:
            mask = ImageProcessor._largest_component_fallback(mask)

        image_obj.mask = mask.astype(np.float32)
        image_obj.segmented = gray * mask
        image_obj.segmented_ready = True

    @staticmethod
    def to_display_image(array: np.ndarray) -> Image.Image:
        """Converte array normalizado para imagem PIL exibível."""
        clipped = np.clip(array, 0.0, 1.0)
        return Image.fromarray((clipped * 255).astype(np.uint8), mode="L")

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> float:
        """Calcula limiar de Otsu."""
        values = np.clip(gray, 0.0, 1.0)
        hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
        total = values.size
        sum_total = np.dot(hist, np.arange(256))
        weight_bg = 0.0
        sum_bg = 0.0
        best_variance = -1.0
        best_threshold = 0
        for threshold in range(256):
            weight_bg += hist[threshold]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += threshold * hist[threshold]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if variance > best_variance:
                best_variance = variance
                best_threshold = threshold
        return float(bin_edges[best_threshold])

    @staticmethod
    def _largest_component_fallback(mask: np.ndarray) -> np.ndarray:
        """Fallback para encontrar maior componente conectado sem OpenCV."""
        try:
            from scipy import ndimage

            structure = np.ones((3, 3), dtype=np.uint8)
            labels, count = ndimage.label(mask > 0, structure=structure)
            if count == 0:
                return np.zeros_like(mask, dtype=np.uint8)
            areas = np.bincount(labels.ravel())
            areas[0] = 0
            largest = int(np.argmax(areas))
            return (labels == largest).astype(np.uint8)
        except ImportError:
            LOGGER.debug("scipy indisponivel; usando BFS para maior componente.")

        visited = np.zeros(mask.shape, dtype=bool)
        best_component: list[tuple[int, int]] = []
        height, width = mask.shape
        for seed_y in range(height):
            for seed_x in range(width):
                if visited[seed_y, seed_x] or mask[seed_y, seed_x] == 0:
                    continue
                component: list[tuple[int, int]] = []
                stack = [(seed_y, seed_x)]
                visited[seed_y, seed_x] = True
                while stack:
                    y, x = stack.pop()
                    component.append((y, x))
                    for ny in range(max(0, y - 1), min(height, y + 2)):
                        for nx in range(max(0, x - 1), min(width, x + 2)):
                            if not visited[ny, nx] and mask[ny, nx] == 1:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                if len(component) > len(best_component):
                    best_component = component
        cleaned = np.zeros_like(mask)
        for y, x in best_component:
            cleaned[y, x] = 1
        return cleaned


# ============================================================================
# DATASET_MANAGER
# ============================================================================

class DatasetManager:
    """
    Responsável pela manipulação do dataset.

    Responsabilidades:
    - Descobrir imagens nos diretórios
    - Extrair labels
    - Separar treino/teste pela regra: index % 4 == 0 → teste
    - Criar datasets para PyTorch
    """

    @staticmethod
    def discover_dataset(dataset_dir: Path) -> list[ImageRecord]:
        """Descobre e indexa todas as imagens do dataset."""
        records: list[ImageRecord] = []
        if not dataset_dir.exists():
            return records
        for path in sorted(dataset_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            class_name = DatasetManager._class_name_from_path(path, dataset_dir)
            if class_name is None:
                LOGGER.debug("Ignorando imagem sem classe reconhecida: %s", path)
                continue
            number = DatasetManager._natural_number_from_name(path)
            split = "test" if number % 4 == 0 else "train"
            records.append(
                ImageRecord(
                    path=path,
                    class_name=class_name,
                    class_index=CLASS_TO_INDEX[class_name],
                    number=number,
                    split=split,
                )
            )
        return records

    @staticmethod
    def _class_name_from_path(path: Path, dataset_dir: Path) -> str | None:
        try:
            relative_parts = path.relative_to(dataset_dir).parts[:-1]
        except ValueError:
            relative_parts = path.parts[:-1]
        for part in reversed(relative_parts):
            stripped = part.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper in CLASS_TO_INDEX:
                return upper
            class_name = upper[0]
            if class_name in CLASS_TO_INDEX and len(stripped) > 1 and stripped[1] in {" ", "+", "-", "_"}:
                return class_name
        return None

    @staticmethod
    def summarize_records(records: Iterable[ImageRecord]) -> str:
        """Gera resumo estatístico do dataset."""
        records = list(records)
        lines = [f"Total: {len(records)} imagens"]
        for class_name in CLASS_NAMES:
            subset = [item for item in records if item.class_name == class_name]
            train = sum(item.split == "train" for item in subset)
            test = sum(item.split == "test" for item in subset)
            lines.append(f"{class_name} ({BIRADS_LABELS[class_name]}): {len(subset)} | treino={train} teste={test}")
        return "\n".join(lines)

    @staticmethod
    def _natural_number_from_name(path: Path) -> int:
        """Extrai número do nome do arquivo para regra de split."""
        matches = re.findall(r"\((\d+)\)|(\d+)", path.stem)
        if not matches:
            return 1
        last = matches[-1]
        return int(last[0] or last[1])


class MammographyDataset(Dataset):
    """Dataset PyTorch para o pipeline de treino."""

    def __init__(
        self,
        records: list[ImageRecord],
        task: str,
        train: bool,
        segmented: bool,
        image_size: int = IMAGE_DEFAULT_SIZE,
    ) -> None:
        self.records = records
        self.task = task
        self.train = train
        self.segmented = segmented
        self.image_size = image_size
        if transforms is None:
            raise RuntimeError("Instale torchvision para usar MammographyDataset.")
        self.samples: list[tuple[ImageRecord, int]] = []
        angles = ROTATION_ANGLES if train else [0]
        for record in records:
            for angle in angles:
                self.samples.append((record, angle))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        record, angle = self.samples[index]
        processor = ImageProcessor()
        image_obj = MammogramImage(record.path)
        processor.load_image(image_obj)
        if self.segmented:
            processor.segment_breast(image_obj)
            gray = image_obj.segmented
        else:
            gray = image_obj.preprocessed
        image = processor.to_display_image(gray).convert("RGB")
        if angle:
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        label = record.label_index(self.task)
        return self.transform(image), torch.tensor(label, dtype=torch.long)


# ============================================================================
# CNN_TRAINER
# ============================================================================

class CNNTrainer:
    """
    Responsável pelo treinamento das redes neurais.

    Responsabilidades:
    - Construir arquiteturas (ResNet, EfficientNet)
    - Executar treino
    - Calcular métricas
    - Salvar e carregar pesos
    """

    @staticmethod
    def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> Any:
        """Constrói modelo com pesos pré-treinados."""
        if torch is None or models is None:
            raise RuntimeError("Instale PyTorch/torchvision para treinar.")
        
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            for parameter in model.parameters():
                parameter.requires_grad = False
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model
        raise ValueError(f"Modelo desconhecido: {model_name}")

    @staticmethod
    def train(
        records: list[ImageRecord],
        model_name: str,
        task: str,
        segmented: bool,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        progress: Callable[[str], None],
    ) -> Path:
        """Treina o modelo."""
        if torch is None:
            raise RuntimeError("Instale PyTorch para treinar.")
        
        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        train_records = [record for record in records if record.split == "train"]
        if not train_records:
            raise ValueError("Nenhum registro de treino encontrado.")
        
        num_classes = 2 if task == "binary" else 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = MammographyDataset(train_records, task=task, train=True, segmented=segmented)
        use_cuda = torch.cuda.is_available()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2 if use_cuda else 0,
            pin_memory=use_cuda,
        )
        model = CNNTrainer.build_model(model_name, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        
        start = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for batch_index, (inputs, labels) in enumerate(loader, start=1):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * labels.size(0)
                predictions = outputs.argmax(dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.size(0))
                if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                    progress(
                        f"Epoca {epoch + 1}/{epochs} batch {batch_index}/{len(loader)}: "
                        f"loss={running_loss / max(total, 1):.4f}"
                    )
            progress(
                f"Epoca {epoch + 1}/{epochs}: loss={running_loss / max(total, 1):.4f} "
                f"acc={correct / max(total, 1):.4f}"
            )
        
        path = CNNTrainer._model_path(model_name, task, segmented)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epochs,
                "task": task,
                "model_name": model_name,
                "segmented": segmented,
                "num_classes": num_classes,
            },
            path,
        )
        progress(f"Treino concluido em {time.perf_counter() - start:.2f}s. Modelo salvo em {path}")
        return path

    @staticmethod
    def _model_path(model_name: str, task: str, segmented: bool) -> Path:
        """Gera caminho para salvar modelo."""
        suffix = "segmentado" if segmented else "original"
        return MODEL_DIR / f"{model_name}_{task}_{suffix}.pt"


# ============================================================================
# PREDICTOR
# ============================================================================

class Predictor:
    """
    Responsável pela inferência e classificação.

    Responsabilidades:
    - Carregar modelos treinados
    - Fazer predições
    - Calcular confiança
    """

    @staticmethod
    def load_model(model_name: str, task: str, segmented: bool) -> Any:
        """Carrega modelo treinado."""
        if torch is None:
            raise RuntimeError("Instale PyTorch para usar predictor.")
        
        path = CNNTrainer._model_path(model_name, task, segmented)
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {path}")
        
        num_classes = 2 if task == "binary" else 4
        model = CNNTrainer.build_model(model_name, num_classes)
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model

    @staticmethod
    def evaluate(
        records: list[ImageRecord],
        model_name: str,
        task: str,
        segmented: bool,
        batch_size: int,
    ) -> str:
        """Avalia modelo no conjunto de teste."""
        if torch is None:
            raise RuntimeError("Instale PyTorch para avaliar.")
        
        test_records = [record for record in records if record.split == "test"]
        if not test_records:
            raise ValueError("Nenhum registro de teste encontrado.")
        
        num_classes = 2 if task == "binary" else 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Predictor.load_model(model_name, task, segmented).to(device)
        dataset = MammographyDataset(test_records, task=task, train=False, segmented=segmented)
        use_cuda = torch.cuda.is_available()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2 if use_cuda else 0,
            pin_memory=use_cuda,
        )
        
        y_true: list[int] = []
        y_pred: list[int] = []
        start = time.perf_counter()
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs.to(device))
                predictions = outputs.argmax(dim=1).cpu().numpy().tolist()
                y_pred.extend(predictions)
                y_true.extend(labels.numpy().tolist())
        elapsed = time.perf_counter() - start
        
        confusion = Predictor._confusion_matrix(y_true, y_pred, num_classes)
        if task == "binary":
            metrics = Predictor._binary_metrics(confusion)
            return (
                f"Tempo de execucao: {elapsed:.2f}s\n"
                f"Matriz de confusao:\n{confusion}\n"
                f"Sensibilidade: {metrics['sensibilidade']:.4f}\n"
                f"Especificidade: {metrics['especificidade']:.4f}\n"
                f"Precisao: {metrics['precisao']:.4f}\n"
                f"Acuracia: {metrics['acuracia']:.4f}\n"
                f"F1: {metrics['f1']:.4f}"
            )
        sensitivity, specificity = Predictor._multiclass_sensitivity_specificity(confusion)
        return (
            f"Tempo de execucao: {elapsed:.2f}s\n"
            f"Matriz de confusao (linhas=real, colunas=predito):\n{confusion}\n"
            f"Sensibilidade media: {sensitivity:.4f}\n"
            f"Especificidade media: {specificity:.4f}"
        )

    @staticmethod
    def _confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
        """Calcula matriz de confusão."""
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_true, y_pred):
            matrix[true, pred] += 1
        return matrix

    @staticmethod
    def _binary_metrics(matrix: np.ndarray) -> dict[str, float]:
        """Calcula métricas para classificação binária."""
        tn, fp, fn, tp = matrix.ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        accuracy = (tp + tn) / max(matrix.sum(), 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-8)
        return {
            "sensibilidade": sensitivity,
            "especificidade": specificity,
            "precisao": precision,
            "acuracia": accuracy,
            "f1": f1,
        }

    @staticmethod
    def _multiclass_sensitivity_specificity(matrix: np.ndarray) -> tuple[float, float]:
        """Calcula sensibilidade e especificidade médias para multiclasse."""
        sensitivities = []
        specificities = []
        total = matrix.sum()
        for index in range(matrix.shape[0]):
            tp = matrix[index, index]
            fn = matrix[index, :].sum() - tp
            fp = matrix[:, index].sum() - tp
            tn = total - tp - fn - fp
            sensitivities.append(tp / max(tp + fn, 1))
            specificities.append(tn / max(tn + fp, 1))
        return float(np.mean(sensitivities)), float(np.mean(specificities))


# ============================================================================
# GRADCAM_GENERATOR
# ============================================================================

class GradCAMGenerator:
    """
    Responsável pela geração de Grad-CAM.

    Responsabilidades:
    - Gerar mapas de ativação
    - Criar heatmaps
    - Sobrepor na imagem original
    """

    @staticmethod
    def generate(model_name: str, task: str, segmented: bool, image_path: Path) -> tuple[str, Image.Image]:
        """Gera Grad-CAM para uma imagem."""
        if torch is None:
            raise RuntimeError("Instale PyTorch para gerar Grad-CAM.")
        
        model = Predictor.load_model(model_name, task, segmented)
        target_layer = GradCAMGenerator._target_layer(model)
        activations: list[Any] = []
        gradients: list[Any] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output.detach())

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)
        model.eval()
        
        try:
            tensor = GradCAMGenerator._preprocess_for_model(image_path, segmented)
            output = model(tensor)
            predicted = int(output.argmax(dim=1).item())
            model.zero_grad()
            output[0, predicted].backward()
        finally:
            handle_f.remove()
            handle_b.remove()

        if not activations or not gradients:
            raise RuntimeError("Nao foi possivel capturar ativacoes/gradientes para Grad-CAM.")

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[0]).sum(dim=1).squeeze().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / max(float(cam.max()), 1e-8)
        
        processor = ImageProcessor()
        image_obj = MammogramImage(image_path)
        processor.load_image(image_obj)
        base = processor.to_display_image(image_obj.preprocessed).convert("RGB").resize((IMAGE_DEFAULT_SIZE, IMAGE_DEFAULT_SIZE))
        heat = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMAGE_DEFAULT_SIZE, IMAGE_DEFAULT_SIZE), Image.Resampling.BILINEAR)
        
        if cv2 is not None:
            color = cv2.applyColorMap(np.asarray(heat), cv2.COLORMAP_JET)
            color = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        else:
            color = ImageOps.colorize(heat, black="black", white="red")
        
        overlay = Image.blend(base, color, alpha=0.45)
        label = "Baixa densidade (I+II)" if task == "binary" and predicted == 0 else "Alta densidade (III+IV)"
        if task == "four":
            label = f"{CLASS_NAMES[predicted]} - {BIRADS_LABELS[CLASS_NAMES[predicted]]}"
        return label, overlay

    @staticmethod
    def _target_layer(model: Any) -> Any:
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        if hasattr(model, "features"):
            for module in reversed(model.features):
                if isinstance(module, nn.Conv2d) or any(isinstance(child, nn.Conv2d) for child in module.modules()):
                    return module
        raise ValueError("Nao foi possivel identificar a ultima camada convolucional do modelo.")

    @staticmethod
    def _preprocess_for_model(path: Path, segmented: bool, image_size: int = IMAGE_DEFAULT_SIZE) -> Any:
        """Pré-processa imagem para modelo."""
        if transforms is None:
            raise RuntimeError("Instale torchvision para pré-processar.")
        
        processor = ImageProcessor()
        image_obj = MammogramImage(path)
        processor.load_image(image_obj)
        if segmented:
            processor.segment_breast(image_obj)
            gray = image_obj.segmented
        else:
            gray = image_obj.preprocessed
        image = processor.to_display_image(gray).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ]
        )
        return transform(image).unsqueeze(0)


# ============================================================================
# MAMMOGRAPHY_APP:
# ============================================================================

class MammographyApp(tk.Tk):
    """
    Interface gráfica da aplicação.

    Responsabilidades:
    - Receber ações do usuário
    - Chamar serviços das outras classes
    - Atualizar visualização
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("PAI - Segmentacao e classificacao mamografica")
        self.geometry(GEOMETRY)
        self.records: list[ImageRecord] = []
        self.dataset_dir = DEFAULT_DATASET_DIR
        self.current_path: Path | None = None
        self.current_image: MammogramImage | None = None
        self.current_display: Image.Image | None = None
        self.tk_image = None
        self.message_queue: queue.Queue[str] = queue.Queue()

        # Estado da aba de classificação
        self.classif_path: Path | None = None
        self.classif_image: MammogramImage | None = None
        self._tk_orig = None
        self._tk_mask = None
        self._tk_seg = None

        # Flag para cancelar treino entre batches
        self._cancel_flag = threading.Event()

        self.model_var = tk.StringVar(value="resnet18")
        self.task_var = tk.StringVar(value="binary")
        self.segmented_var = tk.BooleanVar(value=True)
        self.zoom_var = tk.DoubleVar(value=ZOOM_DEFAULT)
        self.epochs_var = tk.IntVar(value=GUI_DEFAULT_EPOCHS)
        self.batch_var = tk.IntVar(value=GUI_DEFAULT_BATCH)
        self.lr_var = tk.DoubleVar(value=GUI_DEFAULT_LR)

        self.build_layout()
        self.after(200, self.flush_messages)
        if self.dataset_dir.exists():
            self.load_dataset(self.dataset_dir)

    # ------------------------------------------------------------------ layout

    def build_layout(self) -> None:
        """Constrói a interface com sidebar dinâmica e duas abas."""
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Container da sidebar — largura fixa
        self._sidebar_container = ttk.Frame(self, padding=6, width=215)
        self._sidebar_container.grid(row=0, column=0, sticky="ns")
        self._sidebar_container.grid_propagate(False)
        self._sidebar_container.columnconfigure(0, weight=1)
        self._sidebar_container.rowconfigure(0, weight=1)

        self._sidebar_train = ttk.Frame(self._sidebar_container)
        self._sidebar_train.grid(row=0, column=0, sticky="nsew")
        self._build_sidebar_train(self._sidebar_train)

        self._sidebar_classif = ttk.Frame(self._sidebar_container)
        self._build_sidebar_classif(self._sidebar_classif)

        # Notebook principal
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=1, sticky="nsew")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

        tab_train = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(tab_train, text="  Treinamento  ")
        self._build_train_tab(tab_train)

        tab_classif = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(tab_classif, text="  Classificação  ")
        self._build_classif_tab(tab_classif)

    def _on_tab_change(self, _event=None) -> None:
        idx = self.notebook.index(self.notebook.select())
        if idx == 0:
            self._sidebar_classif.grid_remove()
            self._sidebar_train.grid(row=0, column=0, sticky="nsew")
        else:
            self._sidebar_train.grid_remove()
            self._sidebar_classif.grid(row=0, column=0, sticky="nsew")

    def _build_sidebar_train(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        r = 0

        ttk.Button(parent, text="Selecionar dataset", command=self.choose_dataset).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1
        ttk.Button(parent, text="Abrir imagem", command=self.open_image).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1
        ttk.Button(parent, text="Segmentar imagem", command=self.segment_current).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1

        ttk.Separator(parent).grid(row=r, column=0, sticky="ew", pady=8); r += 1

        ttk.Label(parent, text="Modelo").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Combobox(
            parent, textvariable=self.model_var,
            values=["resnet18", "efficientnet_b0"], state="readonly",
        ).grid(row=r, column=0, sticky="ew"); r += 1

        ttk.Label(parent, text="Tarefa").grid(row=r, column=0, sticky="w", pady=(8, 0)); r += 1
        ttk.Combobox(
            parent, textvariable=self.task_var,
            values=["binary", "four"], state="readonly",
        ).grid(row=r, column=0, sticky="ew"); r += 1

        ttk.Checkbutton(
            parent, text="Usar imagens segmentadas", variable=self.segmented_var,
        ).grid(row=r, column=0, sticky="w", pady=5); r += 1

        ttk.Label(parent, text="Épocas").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(parent, from_=1, to=50, textvariable=self.epochs_var, width=8).grid(
            row=r, column=0, sticky="ew"); r += 1

        ttk.Label(parent, text="Batch").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(parent, from_=1, to=64, textvariable=self.batch_var, width=8).grid(
            row=r, column=0, sticky="ew"); r += 1

        ttk.Label(parent, text="Learning Rate").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Entry(parent, textvariable=self.lr_var).grid(row=r, column=0, sticky="ew"); r += 1

        ttk.Separator(parent).grid(row=r, column=0, sticky="ew", pady=8); r += 1

        ttk.Button(parent, text="Treinar", command=self.train_selected).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1
        ttk.Button(parent, text="Cancelar treinamento", command=self.cancel_training).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1
        ttk.Button(parent, text="Avaliar teste", command=self.evaluate_selected).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1
        ttk.Button(parent, text="Exportar modelo", command=self.export_model).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1

        ttk.Separator(parent).grid(row=r, column=0, sticky="ew", pady=8); r += 1

        ttk.Label(parent, text="Zoom").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Scale(
            parent, from_=0.2, to=3.0, variable=self.zoom_var,
            command=lambda _v: self.refresh_image(),
        ).grid(row=r, column=0, sticky="ew"); r += 1

        ttk.Separator(parent).grid(row=r, column=0, sticky="ew", pady=8); r += 1

        ttk.Button(parent, text="Grad-CAM", command=self.run_grad_cam).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1

    def _build_sidebar_classif(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        r = 0

        ttk.Button(parent, text="Selecionar imagem", command=self.classif_open_image).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1

        ttk.Separator(parent).grid(row=r, column=0, sticky="ew", pady=8); r += 1

        ttk.Label(parent, text="Zoom").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Scale(
            parent, from_=0.2, to=3.0, variable=self.zoom_var,
            command=lambda _v: self.classif_refresh_images(),
        ).grid(row=r, column=0, sticky="ew"); r += 1

        ttk.Separator(parent).grid(row=r, column=0, sticky="ew", pady=8); r += 1

        ttk.Button(parent, text="Classificar", command=self.classif_run).grid(
            row=r, column=0, sticky="ew", pady=3); r += 1

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=2)

        chart1_frame = ttk.LabelFrame(parent, text="Gráfico 1", padding=4)
        chart1_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))
        chart1_frame.rowconfigure(0, weight=1)
        chart1_frame.columnconfigure(0, weight=1)
        self.chart1_canvas = tk.Canvas(chart1_frame, background="#1a1a2e", highlightthickness=0)
        self.chart1_canvas.grid(row=0, column=0, sticky="nsew")

        chart2_frame = ttk.LabelFrame(parent, text="Gráfico 2 / Visualização", padding=4)
        chart2_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 4))
        chart2_frame.rowconfigure(0, weight=1)
        chart2_frame.columnconfigure(0, weight=1)
        self.chart2_canvas = tk.Canvas(chart2_frame, background="#1a1a2e", highlightthickness=0)
        self.chart2_canvas.grid(row=0, column=0, sticky="nsew")

        # Alias usado por refresh_image / run_grad_cam
        self.canvas = self.chart2_canvas

        log_frame = ttk.LabelFrame(parent, text="Log", padding=4)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            wrap="word",
            background="#0d1117",
            foreground="#c9d1d9",
            insertbackground="#c9d1d9",
            font=("Courier", 9),
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_classif_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=2)

        orig_frame = ttk.LabelFrame(parent, text="Imagem Original", padding=4)
        orig_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))
        orig_frame.rowconfigure(0, weight=1)
        orig_frame.columnconfigure(0, weight=1)
        self.classif_canvas_orig = tk.Canvas(orig_frame, background="#111111", highlightthickness=0)
        self.classif_canvas_orig.grid(row=0, column=0, sticky="nsew")

        mask_frame = ttk.LabelFrame(parent, text="Máscara", padding=4)
        mask_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=(0, 4))
        mask_frame.rowconfigure(0, weight=1)
        mask_frame.columnconfigure(0, weight=1)
        self.classif_canvas_mask = tk.Canvas(mask_frame, background="#111111", highlightthickness=0)
        self.classif_canvas_mask.grid(row=0, column=0, sticky="nsew")

        seg_frame = ttk.LabelFrame(parent, text="Imagem Segmentada", padding=4)
        seg_frame.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=(0, 4))
        seg_frame.rowconfigure(0, weight=1)
        seg_frame.columnconfigure(0, weight=1)
        self.classif_canvas_seg = tk.Canvas(seg_frame, background="#111111", highlightthickness=0)
        self.classif_canvas_seg.grid(row=0, column=0, sticky="nsew")

        result_frame = ttk.LabelFrame(parent, text="Resultado da Classificação", padding=8)
        result_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        self.classif_result_text = tk.Text(
            result_frame,
            wrap="word",
            background="#0d1117",
            foreground="#c9d1d9",
            insertbackground="#c9d1d9",
            font=("Courier", 12),
            state="disabled",
        )
        self.classif_result_text.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------ logging

    def log_message(self, message: str) -> None:
        self.message_queue.put(message)

    def flush_messages(self) -> None:
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.log.insert("end", message + "\n")
            self.log.see("end")
        self.after(200, self.flush_messages)

    # ------------------------------------------------------------------ dataset

    def load_dataset(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.records = DatasetManager.discover_dataset(dataset_dir)
        self.log_message(f"Dataset carregado: {dataset_dir}\n{DatasetManager.summarize_records(self.records)}")

    def choose_dataset(self) -> None:
        directory = filedialog.askdirectory(
            initialdir=str(self.dataset_dir if self.dataset_dir.exists() else Path.cwd())
        )
        if directory:
            self.load_dataset(Path(directory))

    # ------------------------------------------------------------------ training-tab image viewer

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff"), ("PNG", "*.png"), ("TIFF", "*.tif *.tiff")]
        )
        if not filename:
            return
        try:
            self.current_path = Path(filename)
            self.current_image = MammogramImage(self.current_path)
            ImageProcessor.load_image(self.current_image)
            self.current_display = ImageProcessor.to_display_image(self.current_image.preprocessed)
            self.zoom_var.set(1.0)
            self.refresh_image()
            self.log_message(f"Imagem aberta: {self.current_path}")
        except Exception as exc:
            messagebox.showerror("Abrir imagem", str(exc))

    def refresh_image(self) -> None:
        if self.current_display is None:
            return
        zoom = float(self.zoom_var.get())
        width = max(1, int(self.current_display.width * zoom))
        height = max(1, int(self.current_display.height * zoom))
        resized = self.current_display.resize((width, height), Image.Resampling.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, width, height))

    def segment_current(self) -> None:
        if self.current_image is None:
            messagebox.showinfo("Segmentacao", "Abra uma imagem primeiro.")
            return
        ImageProcessor.segment_breast(self.current_image)
        self.current_display = ImageProcessor.to_display_image(self.current_image.segmented)
        self.refresh_image()
        self.log_message("Segmentacao aplicada.")

    # ------------------------------------------------------------------ training

    def ensure_records(self) -> bool:
        if not self.records:
            messagebox.showinfo(
                "Dataset",
                f"Selecione o diretorio do dataset antes. Atual: {self.dataset_dir}",
            )
            return False
        return True

    def run_background(self, job: Callable[[], str | None]) -> None:
        def wrapper() -> None:
            try:
                result = job()
                if result:
                    self.log_message(result)
            except Exception as exc:
                LOGGER.exception("Erro em tarefa de background")
                self.log_message(f"ERRO: {exc}")

        threading.Thread(target=wrapper, daemon=True).start()

    def cancel_training(self) -> None:
        self._cancel_flag.set()
        self.log_message("Cancelamento solicitado. O treino parará ao final do batch atual.")

    def export_model(self) -> None:
        src = CNNTrainer._model_path(self.model_var.get(), self.task_var.get(), self.segmented_var.get())
        if not src.exists():
            messagebox.showinfo(
                "Exportar modelo",
                f"Modelo não encontrado: {src}\nTreine o modelo primeiro.",
            )
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch model", "*.pt"), ("Todos os arquivos", "*.*")],
            initialfile=src.name,
        )
        if dest:
            import shutil
            shutil.copy2(src, dest)
            self.log_message(f"Modelo exportado para: {dest}")

    def train_selected(self) -> None:
        if not self.ensure_records():
            return
        self._cancel_flag.clear()
        self.log_message("Iniciando treino. A interface continua responsiva; acompanhe o log.")

        def progress(message: str) -> None:
            if self._cancel_flag.is_set():
                raise InterruptedError("Treinamento cancelado pelo usuario.")
            self.log_message(message)

        def job() -> str:
            try:
                path = CNNTrainer.train(
                    self.records,
                    self.model_var.get(),
                    self.task_var.get(),
                    self.segmented_var.get(),
                    int(self.epochs_var.get()),
                    int(self.batch_var.get()),
                    float(self.lr_var.get()),
                    progress,
                )
                return f"Modelo pronto: {path}"
            except InterruptedError as exc:
                return f"Treino interrompido: {exc}"

        self.run_background(job)

    def evaluate_selected(self) -> None:
        if not self.ensure_records():
            return
        self.log_message("Avaliando conjunto de teste...")

        def job() -> str:
            return Predictor.evaluate(
                self.records,
                self.model_var.get(),
                self.task_var.get(),
                self.segmented_var.get(),
                int(self.batch_var.get()),
            )

        self.run_background(job)

    def run_grad_cam(self) -> None:
        if self.current_path is None:
            messagebox.showinfo("Grad-CAM", "Abra uma imagem primeiro.")
            return
        self.log_message("Gerando Grad-CAM...")

        def job() -> str:
            label, overlay = GradCAMGenerator.generate(
                self.model_var.get(),
                self.task_var.get(),
                self.segmented_var.get(),
                self.current_path,
            )

            def update_display() -> None:
                self.current_display = overlay
                self.refresh_image()

            self.after(0, update_display)
            return f"Grad-CAM concluido. Classificacao: {label}"

        self.run_background(job)

    # ------------------------------------------------------------------ classification tab

    def classif_open_image(self) -> None:
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Imagens", "*.png *.tif *.tiff *.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("TIFF", "*.tif *.tiff"),
            ]
        )
        if not filename:
            return
        try:
            self.classif_path = Path(filename)
            self.classif_image = MammogramImage(self.classif_path)
            ImageProcessor.load_image(self.classif_image)
            ImageProcessor.segment_breast(self.classif_image)
            self.zoom_var.set(1.0)
            # Aguarda o layout ser renderizado antes de desenhar
            self.after(50, self.classif_refresh_images)
            self.log_message(f"Imagem carregada para classificação: {self.classif_path.name}")
        except Exception as exc:
            messagebox.showerror("Selecionar imagem", str(exc))

    def classif_refresh_images(self) -> None:
        if self.classif_image is None:
            return
        zoom = float(self.zoom_var.get())
        if self.classif_image.original is not None:
            self._render_on_canvas(
                self.classif_canvas_orig,
                ImageProcessor.to_display_image(self.classif_image.original),
                zoom,
                "_tk_orig",
            )
        if self.classif_image.mask is not None:
            self._render_on_canvas(
                self.classif_canvas_mask,
                ImageProcessor.to_display_image(self.classif_image.mask),
                zoom,
                "_tk_mask",
            )
        if self.classif_image.segmented is not None:
            self._render_on_canvas(
                self.classif_canvas_seg,
                ImageProcessor.to_display_image(self.classif_image.segmented),
                zoom,
                "_tk_seg",
            )

    def _render_on_canvas(
        self, canvas: tk.Canvas, pil_image: Image.Image, zoom: float, attr: str
    ) -> None:
        cw = max(canvas.winfo_width(), 1)
        ch = max(canvas.winfo_height(), 1)
        iw, ih = pil_image.size
        # zoom=1.0 → fit-to-canvas; qualquer outro valor escala em relação ao fit
        fit_scale = min(cw / iw, ch / ih)
        scale = fit_scale * zoom
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        resized = pil_image.resize((nw, nh), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        setattr(self, attr, tk_img)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")

    @staticmethod
    def _classify_image(image_obj: MammogramImage) -> dict:
        """Stub isolado para classificação — substituir pela chamada ao Predictor real."""
        import random
        raw = {label: random.random() for label in BIRADS_LABELS.values()}
        total = sum(raw.values())
        scores = {k: v / total for k, v in raw.items()}
        best = max(scores, key=scores.__getitem__)
        return {"label": best, "scores": scores}

    def classif_run(self) -> None:
        if self.classif_image is None:
            messagebox.showinfo("Classificar", "Selecione uma imagem primeiro.")
            return
        image_obj = self.classif_image

        def job() -> None:
            result = MammographyApp._classify_image(image_obj)
            self.after(0, lambda: self._show_classif_result(result))

        threading.Thread(target=job, daemon=True).start()

    def _show_classif_result(self, result: dict) -> None:
        label = result["label"]
        scores = result["scores"]
        pct_str = "  |  ".join(f"{k} = {v * 100:.1f}%" for k, v in scores.items())
        text = f"Classificação: {label}\n\n{pct_str}"
        self.classif_result_text.config(state="normal")
        self.classif_result_text.delete("1.0", "end")
        self.classif_result_text.insert("end", text)
        self.classif_result_text.config(state="disabled")


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

def main() -> None:
    """Inicia a aplicação."""
    app = MammographyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
