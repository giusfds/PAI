"""
Trabalho Pratico - Processamento e Analise de Imagens
Segmentação e classificação de imagens mamograficas.
"""

# from __future__ import annotations
import queue
import re
import shutil
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
    raise SystemExit("Instale Pillow para executar a aplicação: pip install pillow")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover
    raise SystemExit("Tkinter não está disponivel neste Python.")

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
DEFAULT_DATASET_DIR: Path = Path("dataset")
MODEL_DIR: Path = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("pai")

IMAGE_DEFAULT_SIZE: int = 224
NORMALIZE_MEAN: list[float] = [0.485, 0.456, 0.406]
NORMALIZE_STD: list[float] = [0.229, 0.224, 0.225]
MORPH_KERNEL_SIZE: tuple[int, int] = (7, 7)
MORPH_CLOSE_ITERATIONS: int = 2
OTSU_MIN_THRESHOLD: float = 0.03
CONNECTIVITY: int = 8

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


@dataclass(frozen=True)
class ClassificationResult:
    """Resultado de classificação isolado da interface."""
    label: str
    probabilities: dict[str, float]


# ============================================================================
# IMAGE_PROCESSOR
# ============================================================================

class ImageProcessor:
    """
    Responsável por todas as operações de leitura e processamento de imagens.
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
        should_cancel: Callable[[], bool] | None = None,
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
            if should_cancel is not None and should_cancel():
                raise InterruptedError("Treinamento cancelado pelo usuario.")
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for batch_index, (inputs, labels) in enumerate(loader, start=1):
                if should_cancel is not None and should_cancel():
                    raise InterruptedError("Treinamento cancelado pelo usuario.")
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
                f"Tempo de execução: {elapsed:.2f}s\n"
                f"Matriz de confusao:\n{confusion}\n"
                f"Sensibilidade: {metrics['sensibilidade']:.4f}\n"
                f"Especificidade: {metrics['especificidade']:.4f}\n"
                f"Precisao: {metrics['precisao']:.4f}\n"
                f"Acuracia: {metrics['acuracia']:.4f}\n"
                f"F1: {metrics['f1']:.4f}"
            )
        sensitivity, specificity = Predictor._multiclass_sensitivity_specificity(confusion)
        return (
            f"Tempo de execução: {elapsed:.2f}s\n"
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
# CLASSIFICATION_SERVICE
# ============================================================================

class ClassificationService:
    """
    Ponto isolado para classificação de uma imagem.

    A implementação atual e mockada para manter a UI pronta para conexao com o
    modelo real sem misturar inferencia com codigo de Tkinter.
    """

    @staticmethod
    def classify_image(_image_path: Path, _image_obj: MammogramImage | None = None) -> ClassificationResult:
        probabilities = {
            "BIRADS I": 30.0,
            "BIRADS II": 45.0,
            "BIRADS III": 15.0,
            "BIRADS IV": 10.0,
        }
        return ClassificationResult(label="BIRADS II", probabilities=probabilities)


# ============================================================================
# GRADCAM_GENERATOR
# ============================================================================

class GradCAMGenerator:
    """
    Responsável pela geração de Grad-CAM.
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
            raise RuntimeError("não foi possivel capturar ativacoes/gradientes para Grad-CAM.")

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
        raise ValueError("não foi possivel identificar a ultima camada convolucional do modelo.")

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
# GUI:
# ============================================================================

class GUI(tk.Tk):
    """
    Interface gráfica da aplicação.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("PAI - Segmentação e Classificação Mamográfica")
        self.geometry(GEOMETRY)
        self.records: list[ImageRecord] = []
        self.dataset_dir = DEFAULT_DATASET_DIR
        self.current_path: Path | None = None
        self.current_image: MammogramImage | None = None
        self.current_display: Image.Image | None = None
        self.classification_images: dict[str, Image.Image] = {}
        self.classification_tk_images: dict[str, ImageTk.PhotoImage] = {}
        self.image_canvases: dict[str, tk.Canvas] = {}
        self.result_bars: dict[str, ttk.Progressbar] = {}
        self.result_percent_labels: dict[str, ttk.Label] = {}
        self.message_queue: queue.Queue[str] = queue.Queue()
        self.training_cancel_requested = False

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

    def build_layout(self) -> None:
        """Constroi a interface grafica com abas e sidebar dinamica."""
        self.minsize(980, 640)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar_frame = ttk.Frame(self, padding=10, width=240)
        self.sidebar_frame.grid(row=0, column=0, sticky="ns")
        self.sidebar_frame.grid_propagate(False)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.training_page = ttk.Frame(self.notebook, padding=10)
        self.classification_page = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.training_page, text="Treinamento")
        self.notebook.add(self.classification_page, text="Classificação")

        self.build_training_page()
        self.build_classification_page()
        self.build_sidebar_for_training()

    def build_training_page(self) -> None:
        """Monta a area principal da aba de treinamento."""
        self.training_page.columnconfigure(0, weight=1)
        self.training_page.rowconfigure(0, weight=3)
        self.training_page.rowconfigure(1, weight=2)

        graph_area = ttk.Frame(self.training_page)
        graph_area.grid(row=0, column=0, sticky="nsew")
        graph_area.columnconfigure(0, weight=1)
        graph_area.columnconfigure(1, weight=1)
        graph_area.rowconfigure(0, weight=1)

        self.graph_canvases: list[tk.Canvas] = []
        for column, title in enumerate(("grafico 1", "grafico 2")):
            frame = ttk.LabelFrame(graph_area, text=title, padding=8)
            frame.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0))
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            canvas = tk.Canvas(frame, background="#f5f5f5", highlightthickness=1, highlightbackground="#cccccc")
            canvas.grid(row=0, column=0, sticky="nsew")
            canvas.bind("<Configure>", lambda _event, item=canvas, label=title: self.draw_placeholder(item, label))
            self.graph_canvases.append(canvas)

        log_frame = ttk.LabelFrame(self.training_page, text="Logs", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=10, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

    def build_classification_page(self) -> None:
        """Monta a area principal da aba de classificação."""
        self.classification_page.columnconfigure(0, weight=1)
        self.classification_page.rowconfigure(0, weight=3)
        self.classification_page.rowconfigure(1, weight=2)

        image_area = ttk.Frame(self.classification_page)
        image_area.grid(row=0, column=0, sticky="nsew")
        image_area.rowconfigure(0, weight=1)
        for column in range(3):
            image_area.columnconfigure(column, weight=1, uniform="classification_images")

        panel_specs = (
            ("original", "Imagem original"),
            ("mask", "Mascara"),
            ("segmented", "Imagem segmentada"),
        )
        for column, (key, title) in enumerate(panel_specs):
            frame = ttk.LabelFrame(image_area, text=title, padding=8)
            frame.grid(row=0, column=column, sticky="nsew", padx=self.panel_padding(column))
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            canvas = tk.Canvas(frame, background="#111111", highlightthickness=0)
            canvas.grid(row=0, column=0, sticky="nsew")
            canvas.bind("<Configure>", lambda _event, name=key: self.render_classification_panel(name))
            self.image_canvases[key] = canvas

        result_frame = ttk.LabelFrame(self.classification_page, text="Resultado final da classificação", padding=10)
        result_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)

        self.result_text = tk.Text(result_frame, height=4, wrap="word")
        self.result_text.grid(row=0, column=0, sticky="ew")
        self.result_text.insert("end", "Nenhuma classificação executada.")
        self.result_text.configure(state="disabled")

        metrics_frame = ttk.Frame(result_frame)
        metrics_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        metrics_frame.columnconfigure(1, weight=1)
        for row, label in enumerate(("BIRADS I", "BIRADS II", "BIRADS III", "BIRADS IV")):
            ttk.Label(metrics_frame, text=label, width=12).grid(row=row, column=0, sticky="w", pady=2)
            bar = ttk.Progressbar(metrics_frame, maximum=100, value=0)
            bar.grid(row=row, column=1, sticky="ew", padx=8, pady=2)
            percent_label = ttk.Label(metrics_frame, text="0%", width=6)
            percent_label.grid(row=row, column=2, sticky="e", pady=2)
            self.result_bars[label] = bar
            self.result_percent_labels[label] = percent_label

    @staticmethod
    def panel_padding(column: int) -> tuple[int, int]:
        if column == 0:
            return (0, 6)
        if column == 1:
            return (6, 6)
        return (6, 0)

    def on_tab_changed(self, _event: tk.Event | None = None) -> None:
        """Atualiza a sidebar conforme a aba ativa."""
        tab_text = self.notebook.tab(self.notebook.select(), "text")
        if tab_text == "Classificação":
            self.build_sidebar_for_classification()
        else:
            self.build_sidebar_for_training()

    def clear_sidebar(self) -> ttk.Frame:
        for child in self.sidebar_frame.winfo_children():
            child.destroy()
        self.sidebar_frame.columnconfigure(0, weight=1)
        return self.sidebar_frame

    def build_sidebar_for_training(self) -> None:
        """Sidebar com controles existentes de treinamento."""
        panel = self.clear_sidebar()
        ttk.Label(panel, text="Treinamento", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")

        ttk.Button(panel, text="Selecionar dataset", command=self.choose_dataset).grid(row=1, column=0, sticky="ew", pady=(12, 3))

        ttk.Separator(panel).grid(row=2, column=0, sticky="ew", pady=8)
        ttk.Label(panel, text="Selecionar modelo").grid(row=3, column=0, sticky="w")
        ttk.Combobox(panel, textvariable=self.model_var, values=["resnet18", "efficientnet_b0"], state="readonly").grid(
            row=4, column=0, sticky="ew"
        )

        ttk.Label(panel, text="Parametros gerais").grid(row=5, column=0, sticky="w", pady=(10, 0))
        ttk.Label(panel, text="Tarefa").grid(row=6, column=0, sticky="w")
        ttk.Combobox(panel, textvariable=self.task_var, values=["binary", "four"], state="readonly").grid(
            row=7, column=0, sticky="ew"
        )
        ttk.Checkbutton(panel, text="Usar imagens segmentadas", variable=self.segmented_var).grid(
            row=8, column=0, sticky="w", pady=5
        )
        ttk.Label(panel, text="Epocas").grid(row=9, column=0, sticky="w")
        ttk.Spinbox(panel, from_=1, to=50, textvariable=self.epochs_var, width=8).grid(row=10, column=0, sticky="ew")
        ttk.Label(panel, text="Batch").grid(row=11, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(panel, from_=1, to=64, textvariable=self.batch_var, width=8).grid(row=12, column=0, sticky="ew")
        ttk.Label(panel, text="Learning rate").grid(row=13, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(panel, textvariable=self.lr_var).grid(row=14, column=0, sticky="ew")

        ttk.Separator(panel).grid(row=15, column=0, sticky="ew", pady=10)
        ttk.Button(panel, text="Treinar dataset/modelo", command=self.train_selected).grid(row=16, column=0, sticky="ew", pady=3)
        ttk.Button(panel, text="Cancelar treinamento", command=self.cancel_training).grid(row=17, column=0, sticky="ew", pady=3)
        ttk.Button(panel, text="Exportar modelo treinado", command=self.export_trained_model).grid(row=18, column=0, sticky="ew", pady=3)

        ttk.Separator(panel).grid(row=19, column=0, sticky="ew", pady=10)
        # ttk.Button(panel, text="Avaliar teste", command=self.evaluate_selected).grid(row=20, column=0, sticky="ew", pady=3)
        # ttk.Button(panel, text="Grad-CAM", command=self.run_grad_cam).grid(row=21, column=0, sticky="ew", pady=3)

    def build_sidebar_for_classification(self) -> None:
        """Sidebar especifica da aba de classificação."""
        panel = self.clear_sidebar()
        ttk.Label(panel, text="Classificação", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(panel, text="Selecionar imagem", command=self.open_classification_image).grid(
            row=1, column=0, sticky="ew", pady=(12, 3)
        )

        ttk.Separator(panel).grid(row=2, column=0, sticky="ew", pady=8)
        ttk.Label(panel, text="Zoom").grid(row=3, column=0, sticky="w")
        ttk.Scale(panel, from_=0.2, to=3.0, variable=self.zoom_var, command=lambda _value: self.refresh_image()).grid(
            row=4, column=0, sticky="ew"
        )
        ttk.Button(panel, text="Zoom 100%", command=self.reset_zoom).grid(row=5, column=0, sticky="ew", pady=3)

        ttk.Separator(panel).grid(row=6, column=0, sticky="ew", pady=8)
        ttk.Button(panel, text="Classificar", command=self.classify_current_image).grid(row=7, column=0, sticky="ew", pady=3)

    @staticmethod
    def draw_placeholder(canvas: tk.Canvas, text: str) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        canvas.create_text(width / 2, height / 2, text=text, fill="#777777", font=("TkDefaultFont", 14))

    def log_message(self, message: str) -> None:
        """Adiciona mensagem à fila para exibição."""
        self.message_queue.put(message)

    def flush_messages(self) -> None:
        """Exibe mensagens da fila no log."""
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.log.insert("end", message + "\n")
            self.log.see("end")
        self.after(200, self.flush_messages)

    def load_dataset(self, dataset_dir: Path) -> None:
        """Carrega dataset do diretório."""
        self.dataset_dir = dataset_dir
        self.records = DatasetManager.discover_dataset(dataset_dir)
        self.log_message(f"Dataset carregado: {dataset_dir}\n{DatasetManager.summarize_records(self.records)}")

    def choose_dataset(self) -> None:
        """Abre diálogo para selecionar dataset."""
        directory = filedialog.askdirectory(initialdir=str(self.dataset_dir if self.dataset_dir.exists() else Path.cwd()))
        if directory:
            self.load_dataset(Path(directory))

    def open_image(self) -> None:
        """Abre diálogo para selecionar imagem."""
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
            self.classification_images.clear()
            self.classification_tk_images.clear()
            self.set_classification_image("original", self.current_display)
            self.update_segmentation_panels()
            self.notebook.select(self.classification_page)
            self.refresh_image()
            self.log_message(f"Imagem aberta: {self.current_path}")
        except Exception as exc:
            messagebox.showerror("Abrir imagem", str(exc))

    def open_classification_image(self) -> None:
        """Ação da sidebar de classificação."""
        self.open_image()

    def refresh_image(self) -> None:
        """Atualiza visualização da imagem com zoom."""
        for name in ("original", "mask", "segmented"):
            self.render_classification_panel(name)

    def reset_zoom(self) -> None:
        """Volta o zoom para o tamanho padrao."""
        self.zoom_var.set(1.0)
        self.refresh_image()

    def set_classification_image(self, panel_name: str, image: Image.Image | None) -> None:
        """Define uma imagem PIL para um painel da aba de classificação."""
        if image is None:
            self.classification_images.pop(panel_name, None)
            self.classification_tk_images.pop(panel_name, None)
        else:
            self.classification_images[panel_name] = image.convert("RGB")
        self.render_classification_panel(panel_name)

    def render_classification_panel(self, panel_name: str) -> None:
        """Renderiza imagem ou placeholder em um canvas de classificação."""
        canvas = self.image_canvases.get(panel_name)
        if canvas is None:
            return
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        image = self.classification_images.get(panel_name)
        if image is None:
            labels = {
                "original": "Imagem original",
                "mask": "Mascara",
                "segmented": "Imagem segmentada",
            }
            canvas.create_text(width / 2, height / 2, text=labels.get(panel_name, panel_name), fill="#777777")
            return

        zoom = float(self.zoom_var.get())
        fit_scale = min(width / image.width, height / image.height)
        scale = max(0.01, fit_scale * zoom)
        display_width = max(1, int(image.width * scale))
        display_height = max(1, int(image.height * scale))
        resized = image.resize((display_width, display_height), Image.Resampling.NEAREST)
        tk_image = ImageTk.PhotoImage(resized)
        self.classification_tk_images[panel_name] = tk_image
        canvas.create_image(width / 2, height / 2, image=tk_image, anchor="center")

    def segment_current(self) -> None:
        """Segmenta a imagem atual."""
        if self.current_image is None:
            messagebox.showinfo("Segmentação", "Abra uma imagem primeiro.")
            return
        self.update_segmentation_panels()
        self.log_message("Segmentação aplicada: fundo e anotacoes foram zerados quando possivel.")

    def update_segmentation_panels(self) -> None:
        """Prepara mascara e imagem segmentada para os paineis de classificação."""
        if self.current_image is None:
            self.set_classification_image("mask", None)
            self.set_classification_image("segmented", None)
            return
        try:
            if not self.current_image.segmented_ready:
                ImageProcessor.segment_breast(self.current_image)
            if self.current_image.mask is not None:
                self.set_classification_image("mask", ImageProcessor.to_display_image(self.current_image.mask))
            if self.current_image.segmented is not None:
                segmented_image = ImageProcessor.to_display_image(self.current_image.segmented)
                self.set_classification_image("segmented", segmented_image)
                self.current_display = segmented_image
        except Exception as exc:
            self.set_classification_image("mask", None)
            self.set_classification_image("segmented", None)
            self.log_message(f"não foi possivel gerar mascara/segmentação: {exc}")

    def classify_current_image(self) -> None:
        """Executa a classificação mockada para a imagem atual."""
        if self.current_path is None:
            messagebox.showinfo("Classificação", "Selecione uma imagem primeiro.")
            return
        result = ClassificationService.classify_image(self.current_path, self.current_image)
        self.show_classification_result(result)
        self.log_message(f"Classificação concluida: {result.label}")

    def show_classification_result(self, result: ClassificationResult) -> None:
        """Atualiza texto e metricas visuais do resultado."""
        parts = [f"{label} = {value:.0f}%" for label, value in result.probabilities.items()]
        summary = f"Classificação: {result.label}\n" + ", ".join(parts)
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", summary)
        self.result_text.configure(state="disabled")
        for label, value in result.probabilities.items():
            if label in self.result_bars:
                self.result_bars[label]["value"] = value
            if label in self.result_percent_labels:
                self.result_percent_labels[label].configure(text=f"{value:.0f}%")

    def ensure_records(self) -> bool:
        """Verifica se há dataset carregado."""
        if not self.records:
            messagebox.showinfo("Dataset", f"Selecione o diretorio do dataset antes. Atual: {self.dataset_dir}")
            return False
        return True

    def run_background(self, job: Callable[[], str | None]) -> None:
        """Executa tarefa em thread separada."""
        def wrapper() -> None:
            try:
                result = job()
                if result:
                    self.log_message(result)
            except InterruptedError as exc:
                self.log_message(str(exc))
            except Exception as exc:
                LOGGER.exception("Erro em tarefa de background")
                self.log_message(f"ERRO: {exc}")

        threading.Thread(target=wrapper, daemon=True).start()

    def cancel_training(self) -> None:
        """Registra uma solicitação de cancelamento de treino."""
        self.training_cancel_requested = True
        self.log_message("Cancelamento solicitado. O treino sera interrompido no proximo lote.")

    def export_trained_model(self) -> None:
        """Exporta o arquivo do modelo treinado selecionado."""
        path = CNNTrainer._model_path(self.model_var.get(), self.task_var.get(), self.segmented_var.get())
        if not path.exists():
            messagebox.showinfo("Exportar modelo", f"Modelo não encontrado: {path}")
            return
        target = filedialog.asksaveasfilename(
            title="Exportar modelo treinado",
            initialfile=path.name,
            defaultextension=".pt",
            filetypes=[("PyTorch", "*.pt"), ("Todos os arquivos", "*.*")],
        )
        if not target:
            return
        try:
            shutil.copy2(path, target)
            self.log_message(f"Modelo exportado para: {target}")
        except Exception as exc:
            messagebox.showerror("Exportar modelo", str(exc))

    def train_selected(self) -> None:
        """Treina modelo selecionado."""
        if not self.ensure_records():
            return
        self.training_cancel_requested = False
        self.log_message("Iniciando treino. A interface continua responsiva; acompanhe o log.")

        def job() -> str:
            path = CNNTrainer.train(
                self.records,
                self.model_var.get(),
                self.task_var.get(),
                self.segmented_var.get(),
                int(self.epochs_var.get()),
                int(self.batch_var.get()),
                float(self.lr_var.get()),
                self.log_message,
                lambda: self.training_cancel_requested,
            )
            return f"Modelo pronto: {path}"

        self.run_background(job)

# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

def main() -> None:
    """Inicia a aplicação."""
    app = GUI()
    app.mainloop()


if __name__ == "__main__":
    main()
