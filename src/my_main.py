import re
import queue
import threading
import torch
import logging
import numpy as np
import tkinter as tk
import torch.nn as nn
from pathlib import Path
from scipy import ndimage
from enum import Enum, StrEnum
from rich.logging import RichHandler
from pytorch_grad_cam import GradCAM
from PIL import Image, ImageOps, ImageTk
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
from torchvision import models, transforms
from tkinter import ttk, filedialog, messagebox
from torch.utils.data import DataLoader, Dataset
from pytorch_grad_cam.utils.image import show_cam_on_image

# =============================================================
# LOGGING
# =============================================================

_rich_handler = RichHandler(
    show_time=True,
    show_path=False,
    markup=True
)

_rich_handler.setFormatter(
    logging.Formatter("%(message)s")
)

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

LOG_LEVEL = LogLevel.DEBUG

LOGGER = logging.getLogger("PAI")
LOGGER.addHandler(_rich_handler)
LOGGER.setLevel(LOG_LEVEL.value)
LOGGER.propagate = False

# =============================================================
# ENUMS, CLASSES E CONFIGS
# =============================================================

class SetType(Enum):
    TRAIN = "train"
    TEST = "test"

class SampleSide(Enum):
    LEFT = "left"
    RIGHT = "right"

class SampleView(Enum):
    CC = "cc"
    MLO = "mlo"

class SampleBiradsClass(StrEnum):
    D = "BIRADS I"
    E = "BIRADS II"
    F = "BIRADS III"
    G = "BIRADS IV"

@dataclass(frozen=True, slots=True)
class Sample:
    path: Path
    filename: str
    birads_class: SampleBiradsClass
    number: int
    set_type: SetType
    side: SampleSide
    view: SampleView

class ModelType(StrEnum):
    RESNET18 = "resnet18"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B3 = "efficientnet_b3"

@dataclass(slots=True)
class SegmentationConfig:
    threshold_offset: int = 0
    closing_iterations: int = 3
    kernel_size: int = 3
    crop: bool = False

@dataclass(slots=True)
class DatasetConfig:
    segmented: bool = True
    augmentation: bool = False
    binary_classification: bool = True
    image_size: int = 224
    segmentation_config: SegmentationConfig = field(default_factory=SegmentationConfig)

@dataclass(slots=True)
class TrainingConfig:
    """Configuração de treino para o ciclo de treinamento do modelo."""
    model_type: ModelType = ModelType.RESNET18
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    binary_classification: bool = True
    model_path: Path = Path("model.pth")

# =============================================================
# DATASET MANAGER
# =============================================================

class DatasetManager:
    IMAGE_EXTENSIONS: set[str] = {".png", ".tif", ".tiff"}
    TEST_SPLIT_MOD: int = 4

    def __init__(self):
        self._dataset_dir: Path | None = None
        self.samples: list[Sample] = []
        self.train_samples: list[Sample] = []
        self.test_samples: list[Sample] = []

    def load_dataset(self, dataset_dir_path: str | Path) -> None:
        self._dataset_dir = Path(dataset_dir_path) if isinstance(dataset_dir_path, str) else dataset_dir_path

        self._validate_directory(self._dataset_dir)
        self._clear_old_data()

        LOGGER.info("Carregando dataset do diretório: %s", self._dataset_dir)
        for file in self._dataset_dir.rglob("*"):
            if not file.is_file():
                continue

            if file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue

            sample = self._create_sample(file)
            LOGGER.debug("Sample criado: %s", sample)
            
            self.samples.append(sample)

            if sample.set_type == SetType.TRAIN:
                self.train_samples.append(sample)
            else:
                self.test_samples.append(sample)

        LOGGER.info("Dataset carregado com sucesso. Total de amostras: %d", len(self.samples))
        LOGGER.info("Amostras de treino: %d", len(self.train_samples))
        LOGGER.info("Amostras de teste: %d", len(self.test_samples))

    def _validate_directory(self, path: Path) -> None:
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Diretório inválido: {path}")
    
    def _clear_old_data(self) -> None:
        self.samples.clear()
        self.train_samples.clear()
        self.test_samples.clear()

    def _create_sample(self, file: Path) -> Sample:
        birads_class = self._get_birads_class(file.name)

        number = self._get_image_number(file.name)

        side = self._get_side(file.name)

        view = self._get_view(file.name)

        set_type = SetType.TEST if number % self.TEST_SPLIT_MOD == 0 else SetType.TRAIN

        return Sample(
            path=file,
            filename=file.name,
            birads_class=birads_class,
            number=number,
            set_type=set_type,
            side=side,
            view=view
        )

    def _get_birads_class(self, filename: str) -> SampleBiradsClass:
        return SampleBiradsClass[filename[0].upper()]

    def _get_image_number(self, filename: str) -> int:
        match = re.search(r"\((\d+)\)", filename)

        if not match:
            return 1

        return int(match.group(1))

    def _get_side(self, filename: str) -> SampleSide:
        low = filename.lower()

        if "left" in low:
            return SampleSide.LEFT
        
        if "right" in low:
            return SampleSide.RIGHT
        
        raise ValueError(f"Lado não encontrado no filename: {filename}")

    def _get_view(self, filename: str) -> SampleView:
        up = filename.lower()

        if "cc" in up:
            return SampleView.CC
        
        if "mlo" in up:
            return SampleView.MLO
        
        raise ValueError(f"View não encontrada no filename: {filename}")

# =============================================================
# IMAGE MANAGER
# =============================================================

class ImageManager:
    @staticmethod
    def load(sample: Sample) -> np.ndarray:
        LOGGER.debug("Carregando imagem do sample: %s", sample)
        image = Image.open(sample.path)

        if image.mode != "L":
            image = image.convert("L")

        image_array = np.array(image)

        return image_array

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        LOGGER.debug("Normalizando imagem com shape: %s e dtype: %s", image.shape, image.dtype)

        image = image.astype(np.float32)

        image -= image.min()

        max_value = image.max()

        if max_value > 0:
            image /= max_value

        return image

    @staticmethod
    def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        LOGGER.debug("Redimensionando imagem para %dx%d", width, height)

        pil_image = Image.fromarray(image)

        pil_image = pil_image.resize((width, height), Image.Resampling.BILINEAR)

        return np.array(pil_image)
    
    @staticmethod
    def to_pil(image: np.ndarray) -> Image.Image:
        image = image.astype(np.float32)

        image -= image.min()

        if image.max() > 0:
            image /= image.max()

        image = (image * 255).astype(np.uint8)

        return Image.fromarray(image)

# =============================================================
# SEGMENTATION PROCESSOR
# =============================================================

class SegmentationProcessor:
    @staticmethod
    def otsu_threshold(image: np.ndarray) -> int:
        LOGGER.debug("Calculando threshold de Otsu...")

        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))

        total = image.size

        sum_total = np.dot(np.arange(256), hist)

        sum_background = 0
        weight_background = 0

        max_variance = 0
        threshold = 0

        for t in range(256):
            weight_background += hist[t]

            if weight_background == 0:
                continue

            weight_foreground = total - weight_background

            if weight_foreground == 0:
                break

            sum_background += t * hist[t]

            mean_background = sum_background / weight_background

            mean_foreground = (sum_total - sum_background) / weight_foreground

            variance = (weight_background* weight_foreground* (mean_background - mean_foreground) ** 2)

            if variance > max_variance:
                max_variance = variance
                threshold = t

        LOGGER.debug("Threshold de Otsu calculado: %d", threshold)

        return threshold

    @staticmethod
    def create_mask(image: np.ndarray, threshold_offset: int = 0) -> np.ndarray:
        LOGGER.debug("Criando máscara de segmentação...")

        threshold = SegmentationProcessor.otsu_threshold(image)

        threshold += threshold_offset

        threshold = max(0, min(255, threshold))

        return (image > threshold).astype(np.uint8)

    @staticmethod
    def largest_component(mask: np.ndarray) -> np.ndarray:
        LOGGER.debug("Extraindo maior componente conectada...")

        labels, num_labels = ndimage.label(mask)

        if num_labels == 0:
            return mask

        sizes = ndimage.sum(mask, labels, range(1, num_labels + 1))

        largest = np.argmax(sizes) + 1

        return (labels == largest).astype(np.uint8)
    
    @staticmethod
    def refine_mask(mask: np.ndarray, closing_iterations: int = 3, kernel_size: int = 3) -> np.ndarray:
        LOGGER.debug("Refinando máscara...")

        structure = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        mask = ndimage.binary_fill_holes(mask)

        mask = ndimage.binary_closing(mask, iterations=closing_iterations, structure=structure)

        mask = ndimage.binary_opening(mask, iterations=1, structure=structure)

        smooth_mask = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.0)

        mask = (smooth_mask > 0.5).astype(np.uint8)

        return mask.astype(np.uint8)

    @staticmethod
    def crop_to_bounding_box(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        LOGGER.debug("Cortando imagem para bounding box da máscara...")

        rows, cols = np.where(mask > 0)

        if len(rows) == 0:
            return image

        top = rows.min()
        bottom = rows.max()

        left = cols.min()
        right = cols.max()

        return image[
            top:bottom + 1,
            left:right + 1
        ]

    @staticmethod
    def segment(image: np.ndarray, config: SegmentationConfig = None) -> tuple[np.ndarray, np.ndarray]:
        LOGGER.debug("Segmentando imagem...")

        if config is None:
            config = SegmentationConfig()

        mask = SegmentationProcessor.create_mask(image, threshold_offset=config.threshold_offset)

        mask = SegmentationProcessor.largest_component(mask)

        mask = SegmentationProcessor.refine_mask(
            mask, 
            closing_iterations=config.closing_iterations, 
            kernel_size=config.kernel_size
        )

        segmented = image * mask

        if config.crop:
            segmented = SegmentationProcessor.crop_to_bounding_box(segmented, mask)
            mask = SegmentationProcessor.crop_to_bounding_box(mask, mask)

        LOGGER.debug("Segmentação concluída.")

        return mask, segmented
    
# =============================================================
# AUMENTO DE DADOS
# =============================================================

class DataAugmentationProcessor:
    ROTATIONS = [-20, -10, 0, 10, 20]
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        LOGGER.debug("Rotacionando imagem em %.1f graus", angle)

        rotated = ndimage.rotate(image, angle=angle, reshape=False, mode="constant", cval=0)

        return rotated.astype(image.dtype)

    @staticmethod
    def generate(image: np.ndarray) -> list[np.ndarray]:
        """Gera imagens rotacionadas para aumentar o conjunto de dados."""
        return [DataAugmentationProcessor.rotate(image, angle) for angle in DataAugmentationProcessor.ROTATIONS]

# =============================================================
# TRANSFORMAÇÃO
# =============================================================

class MammographyDataset(Dataset):
    BINARY_CLASS_MAPPING = {
        SampleBiradsClass.D: 0,
        SampleBiradsClass.E: 0,
        SampleBiradsClass.F: 1,
        SampleBiradsClass.G: 1
    }

    FOUR_CLASS_MAPPING = {
        SampleBiradsClass.D: 0,
        SampleBiradsClass.E: 1,
        SampleBiradsClass.F: 2,
        SampleBiradsClass.G: 3
    }

    def __init__(self, samples: list[Sample], config: DatasetConfig):
        self.samples = samples
        self.config = config

        self.transform = transforms.Compose([transforms.ToTensor()])

        LOGGER.info("Dataset criado com %d amostras", len(samples))

    def __len__(self):
        if self.config.augmentation:
            return len(self.samples) * len(DataAugmentationProcessor.ROTATIONS)
        
        return len(self.samples)

    def __getitem__(self, index):
        if self.config.augmentation:
            sample_index = index // len(DataAugmentationProcessor.ROTATIONS)
            rotation_index = index % len(DataAugmentationProcessor.ROTATIONS)

            sample = self.samples[sample_index]
            angle = DataAugmentationProcessor.ROTATIONS[rotation_index]
        else:
            sample = self.samples[index]
            angle = 0

        image = ImageManager.load(sample)

        if self.config.segmented:
            _, image = SegmentationProcessor.segment(image, self.config.segmentation_config)

        if angle != 0:
            image = DataAugmentationProcessor.rotate(image, angle)

        image = ImageManager.resize(image, self.config.image_size, self.config.image_size)
        image = ImageManager.normalize(image)
        image = self._to_rgb(image)

        tensor_image = self.transform(image)
        label = self._get_label(sample)

        return tensor_image, torch.tensor(label, dtype=torch.long)

    def _get_label(self, sample: Sample) -> int:
        if self.config.binary_classification:
            return self.BINARY_CLASS_MAPPING[sample.birads_class]
        return self.FOUR_CLASS_MAPPING[sample.birads_class]
    
    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=-1)
        return image
        
# =============================================================
# TREINAMENTO E CLASSIFICAÇÃO
# =============================================================

class TrainingManager:
    """Gerencia o ciclo de treino, validação e persistência do modelo."""

    def __init__(
        self,
        train_dataset: MammographyDataset,
        test_dataset: MammographyDataset,
        config: TrainingConfig,
        progress_callback: Callable[[int, int, float, float, float, float], None] | None = None
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.progress_callback = progress_callback

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=2
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=2
        )

        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        self.cancel_requested = False

    def _create_model(self) -> nn.Module:
        """Cria o modelo e adapta a última camada para o número de classes."""
        num_classes = 2 if self.config.binary_classification else 4

        if self.config.model_type == ModelType.RESNET18:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif self.config.model_type == ModelType.EFFICIENTNET_B0:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        else:
            raise ValueError(f"Modelo não suportado: {self.config.model_type}")

        return model.to(self.device)

    def train_epoch(self) -> tuple[float, float]:
        """Executa uma única época de treino e retorna loss e acurácia."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            if self.cancel_requested:
                break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        if total == 0:
            return 0.0, 0.0

        return total_loss / total, correct / total

    def validate(self) -> tuple[float, float]:
        """Avalia o modelo no conjunto de validação."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                if self.cancel_requested:
                    break

                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        if total == 0:
            return 0.0, 0.0

        return total_loss / total, correct / total

    def train(self) -> dict[str, list[float]]:
        """Realiza o treino completo e registra métricas por época."""
        for epoch in range(1, self.config.epochs + 1):
            if self.cancel_requested:
                LOGGER.warning("Treinamento cancelado na época %d.", epoch)
                break

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            if self.progress_callback is not None:
                self.progress_callback(
                    epoch,
                    self.config.epochs,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                )

            LOGGER.info(
                "Época %d/%d | Treino loss %.4f | Treino acc %.4f | Val loss %.4f | Val acc %.4f",
                epoch,
                self.config.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc
            )

        return self.history

    def save_model(self, path: Path | None = None) -> None:
        """Salva o estado do modelo em disco."""
        path = path or self.config.model_path
        torch.save(self.model.state_dict(), path)
        LOGGER.info("Modelo salvo em %s", path)

    def load_model(self, path: Path | None = None) -> None:
        """Carrega o estado do modelo a partir de disco."""
        path = path or self.config.model_path
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        LOGGER.info("Modelo carregado de %s", path)

    def predict(self, image_tensor: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Realiza inferência de uma imagem preparada e retorna classe e probabilidades."""
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))
            probabilities = torch.softmax(outputs, dim=1)[0].cpu()
            prediction = int(probabilities.argmax(dim=0).item())

        return prediction, probabilities

# =============================================================
# GRAD-CAM
# =============================================================

class GradCAMProcessor:
    def __init__(self, model: nn.Module):
        target_layer = self.get_target_layer(model)
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    @staticmethod
    def get_target_layer(model: nn.Module) -> nn.Module:
        if isinstance(model, models.ResNet):
            return model.layer4[-1]

        if isinstance(model, models.EfficientNet):
            return model.features[-1]

        raise ValueError(f"Modelo não suportado: {type(model)}")

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        grayscale_cam = self.cam(input_tensor=image_tensor)
        return grayscale_cam[0]

    def overlay(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image -= image.min()

        if image.max() > 0:
            image /= image.max()

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        visualization = show_cam_on_image(image, heatmap, use_rgb=True)
        return visualization

# =============================================================
# ! INTERFACE GRÁFICA
# =============================================================

class TkinterLogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        msg = self.format(record)
        self.callback(msg)

class GUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PAI - Segmentação e Classificação Mamográfica")
        self.dataset_dir: Path | None = None
        self.dataset_manager: DatasetManager = DatasetManager()
        self.training_manager: TrainingManager | None = None
        self.training_thread: threading.Thread | None = None
        self.current_sample: Sample | None = None
        self.current_image: np.ndarray | None = None
        self.current_display: Image.Image | None = None
        self.classification_images: dict[str, Image.Image] = {}
        self.classification_tk_images: dict[str, ImageTk.PhotoImage] = {}
        self.image_canvases: dict[str, tk.Canvas] = {}
        self.result_bars: dict[str, ttk.Progressbar] = {}
        self.result_percent_labels: dict[str, ttk.Label] = {}
        self.message_queue: queue.Queue[str] = queue.Queue()
        self.training_cancel_requested = False

        self.threshold_offset_var = tk.IntVar(value=0)
        self.closing_iterations_var = tk.IntVar(value=3)
        self.kernel_size_var = tk.IntVar(value=3)
        self.crop_var = tk.BooleanVar(value=False)

        self.model_var = tk.StringVar(value="resnet18")
        self.task_var = tk.StringVar(value="binary")
        self.segmented_var = tk.BooleanVar(value=True)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.epochs_var = tk.IntVar(value=10)
        self.batch_var = tk.IntVar(value=32)
        self.lr_var = tk.DoubleVar(value=0.001)

        self.build_layout()
        self.set_training_buttons_state(True)
        self._setup_logging_handler()
        self.after(200, self.flush_messages)

    def build_layout(self) -> None:
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
        self.start_training_button = ttk.Button(panel, text="Treinar dataset/modelo", command=self.start_training)
        self.start_training_button.grid(row=16, column=0, sticky="ew", pady=3)
        self.cancel_training_button = ttk.Button(panel, text="Cancelar treinamento", command=self.cancel_training)
        self.cancel_training_button.grid(row=17, column=0, sticky="ew", pady=3)
        self.export_model_button = ttk.Button(panel, text="Exportar modelo treinado", command=self.export_trained_model)
        self.export_model_button.grid(row=18, column=0, sticky="ew", pady=3)

        ttk.Label(panel, text="Progresso de treino").grid(row=19, column=0, sticky="w", pady=(10, 0))
        self.training_progress_bar = ttk.Progressbar(panel, maximum=100)
        self.training_progress_bar.grid(row=20, column=0, sticky="ew", pady=4)
        self.training_status_label = ttk.Label(panel, text="Aguardando treino...")
        self.training_status_label.grid(row=21, column=0, sticky="w")

        ttk.Separator(panel).grid(row=22, column=0, sticky="ew", pady=10)

    def build_sidebar_for_classification(self) -> None:
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
        ttk.Button(panel, text="Carregar modelo para classificação", command=self.load_existing_model).grid(row=7, column=0, sticky="ew", pady=3)
        self.classify_button = ttk.Button(panel, text="Classificar imagem", command=self.classify_current_image, state="disabled")
        self.classify_button.grid(row=8, column=0, sticky="ew", pady=3)

        ttk.Separator(panel).grid(row=9, column=0, sticky="ew", pady=8)

        ttk.Label(panel, text="Threshold Offset").grid(row=10, column=0, sticky="w")
        ttk.Spinbox(
            panel,
            from_=-50,
            to=50,
            textvariable=self.threshold_offset_var
        ).grid(row=11, column=0, sticky="ew")

        ttk.Label(panel, text="Closing Iterations").grid(row=12, column=0, sticky="w")
        ttk.Spinbox(
            panel,
            from_=1,
            to=10,
            textvariable=self.closing_iterations_var
        ).grid(row=13, column=0, sticky="ew")

        ttk.Label(panel, text="Kernel Size").grid(row=14, column=0, sticky="w")
        ttk.Spinbox(
            panel,
            from_=3,
            to=15,
            increment=2,
            textvariable=self.kernel_size_var
        ).grid(row=15, column=0, sticky="ew")

        ttk.Checkbutton(
            panel,
            text="Crop ROI",
            variable=self.crop_var
        ).grid(row=16, column=0, sticky="w")

    @staticmethod
    def draw_placeholder(canvas: tk.Canvas, text: str) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        canvas.create_text(width / 2, height / 2, text=text, fill="#777777", font=("TkDefaultFont", 14))

    def log_message(self, message: str) -> None:
        self.message_queue.put(message)

    def flush_messages(self) -> None:
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.log.insert("end", message + "\n")
            self.log.see("end")
        self.after(200, self.flush_messages)

    def load_dataset(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_manager.load_dataset(dataset_dir)

    def choose_dataset(self) -> None:
        directory = filedialog.askdirectory(initialdir=str(self.dataset_dir if self.dataset_dir and self.dataset_dir.exists() else Path.cwd()))
        if directory:
            self.load_dataset(Path(directory))

    def create_training_config(self) -> TrainingConfig:
        """Gera a configuração de treino a partir dos controles da interface."""
        model_type = ModelType(self.model_var.get())
        binary = self.task_var.get() == "binary"
        model_folder = Path("models")
        model_folder.mkdir(parents=True, exist_ok=True)

        model_path = model_folder / f"{model_type.value}_{self.task_var.get()}.pth"

        return TrainingConfig(
            model_type=model_type,
            epochs=self.epochs_var.get(),
            batch_size=self.batch_var.get(),
            learning_rate=self.lr_var.get(),
            binary_classification=binary,
            model_path=model_path
        )

    def create_dataset_config(self) -> DatasetConfig:
        """Gera a configuração de dataset a partir da interface."""
        return DatasetConfig(
            segmented=self.segmented_var.get(),
            augmentation=True,
            binary_classification=self.task_var.get() == "binary",
            image_size=224,
            segmentation_config=SegmentationConfig(
                threshold_offset=self.threshold_offset_var.get(),
                closing_iterations=self.closing_iterations_var.get(),
                kernel_size=self.kernel_size_var.get(),
                crop=self.crop_var.get()
            )
        )

    def start_training(self) -> None:
        """Inicia o treinamento em segundo plano para não travar o GUI."""
        if not self.ensure_records():
            return

        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Treinamento", "Um treinamento já está em execução.")
            return

        try:
            dataset_config = self.create_dataset_config()
            test_config = DatasetConfig(
                segmented=dataset_config.segmented,
                augmentation=False,
                binary_classification=dataset_config.binary_classification,
                image_size=dataset_config.image_size,
                segmentation_config=dataset_config.segmentation_config
            )

            train_dataset = MammographyDataset(self.dataset_manager.train_samples, dataset_config)
            test_dataset = MammographyDataset(self.dataset_manager.test_samples, test_config)

            if len(train_dataset) == 0 or len(test_dataset) == 0:
                messagebox.showinfo(
                    "Treinamento",
                    "O dataset precisa conter amostras de treino e teste. Verifique o conjunto carregado."
                )
                return

            config = self.create_training_config()
            self.training_manager = TrainingManager(
                train_dataset,
                test_dataset,
                config,
                progress_callback=self.update_training_progress
            )
            self.training_manager.cancel_requested = False

            self.training_progress_bar["value"] = 0
            self.training_status_label.config(text="Preparando treino...")
            self.set_training_buttons_state(False)
            self.log_message("Iniciando treinamento em segundo plano...")

            self.training_thread = threading.Thread(
                target=self._run_training,
                args=(self.training_manager,),
                daemon=True
            )
            self.training_thread.start()

        except Exception as exc:
            LOGGER.error("Erro ao iniciar treinamento: %s", exc)
            messagebox.showerror("Erro", str(exc))
            self.set_training_buttons_state(True)

    def _run_training(self, manager: TrainingManager) -> None:
        try:
            manager.train()
            manager.save_model()
            self.log_message("Treinamento finalizado e modelo salvo com sucesso.")
            self.after(0, self.render_training_history)
        except Exception as exc:
            LOGGER.error("Erro durante treinamento: %s", exc)
            self.after(0, lambda: messagebox.showerror("Erro", str(exc)))
        finally:
            self.after(0, lambda: self.set_training_buttons_state(True))

    def cancel_training(self) -> None:
        if not self.training_manager:
            return

        self.training_manager.cancel_requested = True
        self.log_message("Solicitação de cancelamento enviada. O treinamento será interrompido após a etapa atual.")

    def export_trained_model(self) -> None:
        if not self.training_manager:
            messagebox.showinfo("Exportar modelo", "Nenhum modelo treinado disponível. Treine um modelo primeiro.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch model", "*.pth")],
            initialfile=self.training_manager.config.model_path.name,
            initialdir=str(self.training_manager.config.model_path.parent)
        )

        if not path:
            return

        try:
            self.training_manager.save_model(Path(path))
            messagebox.showinfo("Exportar modelo", f"Modelo exportado para {path}")
        except Exception as exc:
            LOGGER.error("Erro ao exportar modelo: %s", exc)
            messagebox.showerror("Erro", str(exc))

    def set_training_buttons_state(self, enabled: bool) -> None:
        if hasattr(self, "start_training_button") and hasattr(self, "cancel_training_button") and hasattr(self, "export_model_button"):
            self.start_training_button.config(state="normal" if enabled else "disabled")
            self.cancel_training_button.config(state="normal" if not enabled else "disabled")
            self.export_model_button.config(state="normal" if enabled else "disabled")

    def update_training_progress(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """Atualiza o progresso de treino na interface."""
        percent = int((epoch / total_epochs) * 100)
        self.after(0, lambda: self.training_progress_bar.config(value=percent))
        self.after(
            0,
            lambda: self.training_status_label.config(
                text=f"Época {epoch}/{total_epochs} | loss {train_loss:.4f} | acc {train_acc:.4f} | val {val_acc:.4f}"
            ),
        )

    def render_training_history(self) -> None:
        """Desenha os gráficos de loss e acurácia ao final do treino."""
        if not self.training_manager:
            return

        history = self.training_manager.history
        self._draw_training_history_chart(
            self.graph_canvases[0],
            history["train_loss"],
            history["val_loss"],
            "Loss"
        )
        self._draw_training_history_chart(
            self.graph_canvases[1],
            history["train_accuracy"],
            history["val_accuracy"],
            "Acurácia"
        )

    def _draw_training_history_chart(
        self,
        canvas: tk.Canvas,
        train_values: list[float],
        val_values: list[float],
        title: str,
    ) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)

        if not train_values:
            canvas.create_text(width / 2, height / 2, text="Sem dados", fill="#777777")
            return

        margin = 30
        chart_width = width - margin * 2
        chart_height = height - margin * 2
        max_val = max(train_values + val_values) if train_values or val_values else 1.0
        min_val = min(train_values + val_values) if train_values or val_values else 0.0
        value_range = max(max_val - min_val, 1e-6)

        canvas.create_text(width / 2, margin / 2, text=title, fill="#333333", font=("TkDefaultFont", 11, "bold"))
        canvas.create_rectangle(margin, margin, width - margin, height - margin, outline="#cccccc")

        def coord(index: int, value: float) -> tuple[float, float]:
            x = margin + (chart_width * index / max(len(train_values), len(val_values), 1))
            y = margin + chart_height - ((value - min_val) / value_range) * chart_height
            return x, y

        train_points = [coord(i, v) for i, v in enumerate(train_values)]
        val_points = [coord(i, v) for i, v in enumerate(val_values)]

        if len(train_points) > 1:
            canvas.create_line(*sum(train_points, ()), fill="#007acc", width=2, smooth=True)
        if len(val_points) > 1:
            canvas.create_line(*sum(val_points, ()), fill="#ff6600", width=2, smooth=True)

        canvas.create_text(width - margin, height - margin + 12, anchor="ne", text="Treino", fill="#007acc")
        canvas.create_text(width - margin, height - margin + 26, anchor="ne", text="Validação", fill="#ff6600")

    def load_existing_model(self) -> None:
        """Carrega um modelo treinado para ser usado na classificação."""
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch model", "*.pth")],
            initialdir=str(Path("models").resolve())
        )

        if not path:
            return

        model_config = self.create_training_config()
        dataset_config = DatasetConfig(
            segmented=self.segmented_var.get(),
            augmentation=False,
            binary_classification=self.task_var.get() == "binary",
            image_size=224,
            segmentation_config=SegmentationConfig(
                threshold_offset=self.threshold_offset_var.get(),
                closing_iterations=self.closing_iterations_var.get(),
                kernel_size=self.kernel_size_var.get(),
                crop=self.crop_var.get()
            )
        )

        train_dataset = MammographyDataset(self.dataset_manager.train_samples, dataset_config)
        test_dataset = MammographyDataset(self.dataset_manager.test_samples, dataset_config)

        self.training_manager = TrainingManager(
            train_dataset,
            test_dataset,
            model_config,
            progress_callback=self.update_training_progress,
        )

        try:
            self.training_manager.load_model(Path(path))
            self.log_message(f"Modelo carregado de {path}")
            self.classify_button.config(state="normal")
        except Exception as exc:
            LOGGER.error("Erro ao carregar modelo: %s", exc)
            messagebox.showerror("Erro", str(exc))

    def classify_current_image(self) -> None:
        """Classifica a imagem carregada usando o modelo disponível."""
        if self.current_image is None:
            messagebox.showinfo("Classificação", "Abra uma imagem para classificar.")
            return

        if not self.training_manager:
            messagebox.showinfo("Classificação", "Carregue um modelo ou treine um modelo primeiro.")
            return

        config = DatasetConfig(
            segmented=self.segmented_var.get(),
            augmentation=False,
            binary_classification=self.task_var.get() == "binary",
            image_size=224,
            segmentation_config=SegmentationConfig(
                threshold_offset=self.threshold_offset_var.get(),
                closing_iterations=self.closing_iterations_var.get(),
                kernel_size=self.kernel_size_var.get(),
                crop=self.crop_var.get()
            )
        )

        processed_image = self.current_image.copy()
        if config.segmented:
            _, processed_image = SegmentationProcessor.segment(processed_image, config.segmentation_config)

        processed_image = ImageManager.resize(processed_image, config.image_size, config.image_size)
        processed_image = ImageManager.normalize(processed_image)
        processed_image = np.stack([processed_image] * 3, axis=-1)

        image_tensor = torch.from_numpy(processed_image.transpose((2, 0, 1))).unsqueeze(0).float()

        prediction, probabilities = self.training_manager.predict(image_tensor)

        class_labels = [
            "BIRADS I",
            "BIRADS II",
            "BIRADS III",
            "BIRADS IV"
        ]

        if config.binary_classification:
            label = "BIRADS I/II" if prediction == 0 else "BIRADS III/IV"
            self.result_text.configure(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Resultado binário: {label}\n")
            self.result_text.insert("end", f"Confiança: {probabilities[prediction].item() * 100:.2f}%\n")
            self.result_text.configure(state="disabled")

            confidence_0 = probabilities[0].item() * 100
            confidence_1 = probabilities[1].item() * 100
            for label_name, value in zip(
                ["BIRADS I", "BIRADS II", "BIRADS III", "BIRADS IV"],
                [confidence_0, confidence_0, confidence_1, confidence_1]
            ):
                self.result_bars[label_name].config(value=value)
                self.result_percent_labels[label_name].config(text=f"{value:.1f}%")
        else:
            label = class_labels[prediction]
            self.result_text.configure(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Resultado: {label}\n")
            self.result_text.insert("end", "Probabilidades por classe:\n")
            for index, class_name in enumerate(class_labels):
                self.result_text.insert("end", f"  {class_name}: {probabilities[index].item() * 100:.2f}%\n")
            self.result_text.configure(state="disabled")

            for class_name, prob_value in zip(class_labels, probabilities.tolist()):
                self.result_bars[class_name].config(value=prob_value * 100)
                self.result_percent_labels[class_name].config(text=f"{prob_value * 100:.1f}%")

        self.log_message(f"Classificação concluída: {label}")

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.tif *.tiff")]
        )

        if not filename:
            return

        try:
            path = Path(filename)

            sample = Sample(
                path=path,
                filename=path.name,
                birads_class=SampleBiradsClass.D,
                number=1,
                set_type=SetType.TEST,
                side=SampleSide.RIGHT,
                view=SampleView.CC
            )

            self.current_sample = sample

            self.current_image = ImageManager.load(sample)

            config = SegmentationConfig(
                threshold_offset=self.threshold_offset_var.get(),
                closing_iterations=self.closing_iterations_var.get(),
                kernel_size=self.kernel_size_var.get(),
                crop=self.crop_var.get()
            )

            mask, segmented = SegmentationProcessor.segment(self.current_image, config)

            image_pil = ImageManager.to_pil(self.current_image)

            self.set_classification_image("original",image_pil)
            self.set_classification_image("mask",ImageManager.to_pil(mask * 255))
            self.set_classification_image("segmented",ImageManager.to_pil(segmented * 255))

            self.zoom_var.set(1.0)

            self.refresh_image()

            LOGGER.info(f"Imagem aberta: {filename}")

        except Exception as exc:
            LOGGER.error(f"Erro ao abrir imagem: {exc}")
            messagebox.showerror("Erro", str(exc))

    def open_classification_image(self) -> None:
        self.open_image()

    def refresh_image(self) -> None:
        for name in ("original", "mask", "segmented"):
            self.render_classification_panel(name)

    def reset_zoom(self) -> None:
        self.zoom_var.set(1.0)
        self.refresh_image()

    def set_classification_image(self, panel_name: str, image: Image.Image | None) -> None:
        if image is None:
            self.classification_images.pop(panel_name, None)
            self.classification_tk_images.pop(panel_name, None)
        else:
            self.classification_images[panel_name] = image.convert("RGB")
        self.render_classification_panel(panel_name)

    def render_classification_panel(self, panel_name: str) -> None:
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

    def ensure_records(self) -> bool:
        if not self.dataset_manager.samples:
            messagebox.showinfo("Dataset", "Selecione um dataset primeiro.")
            return False
        return True
    
    def _setup_logging_handler(self) -> None:
        handler = TkinterLogHandler(self.log_message)

        handler.setFormatter(
            logging.Formatter(
                "[%(levelname)s] %(message)s"
            )
        )

        LOGGER.addHandler(handler)

# =============================================================
# MAIN
# =============================================================

# def main():
#     app = GUI()
#     app.mainloop()

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_single_image(path: Path) -> np.ndarray:
    image = Image.open(path)

    if image.mode != "L":
        image = image.convert("L")

    return np.array(image)


def preprocess(image: np.ndarray, config: DatasetConfig) -> np.ndarray:
    if config.segmented:
        _, image = SegmentationProcessor.segment(
            image,
            config.segmentation_config
        )

    image = ImageManager.resize(image, config.image_size, config.image_size)
    image = ImageManager.normalize(image)

    return image


def show_rotations_from_path(image_path: str):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {path}")

    # config igual ao dataset
    config = DatasetConfig(
        segmented=True,
        augmentation=False,
        image_size=224,
        segmentation_config=SegmentationConfig()
    )

    print(f"📷 Carregando imagem: {path}")

    image = load_single_image(path)

    rotations = DataAugmentationProcessor.ROTATIONS

    plt.figure(figsize=(12, 3))

    rotated_imgs = DataAugmentationProcessor.generate(image)
    for i, img in enumerate(rotated_imgs):
        processed_img = preprocess(img, config)
        rgb = np.stack([processed_img] * 3, axis=-1)

        plt.subplot(1, len(rotations), i + 1)
        plt.imshow(rgb, cmap="gray")
        # plt.title(f"{angle}°")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Caminho da imagem para testar rotações")
    parser.add_argument("--gui", action="store_true", help="Inicia a interface gráfica")

    args = parser.parse_args()

    if args.image:
        show_rotations_from_path(args.image)
        return

    app = GUI()
    app.mainloop()


if __name__ == "__main__":
    main()
