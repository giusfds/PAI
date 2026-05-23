"""Smoke test rapido do pipeline de imagem e forward da ResNet-18.

Uso:
    python scripts/smoke_test.py --dataset dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import (  # noqa: E402
    CLASS_NAMES,
    DatasetManager,
    IMAGE_DEFAULT_SIZE,
    ImageProcessor,
    MammogramImage,
    CNNTrainer,
    torch,
    transforms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa validacao rapida do pipeline PAI.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset", help="Diretorio do dataset.")
    parser.add_argument("--image", type=Path, default=None, help="Imagem especifica para testar.")
    parser.add_argument("--task", choices=["binary", "four"], default="binary")
    return parser.parse_args()


def pick_image(dataset_dir: Path, image: Path | None) -> Path:
    if image is not None:
        if not image.exists():
            raise FileNotFoundError(image)
        return image
    records = DatasetManager.discover_dataset(dataset_dir)
    if not records:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {dataset_dir}")
    return records[0].path


def main() -> None:
    if torch is None or transforms is None:
        raise RuntimeError("Instale torch e torchvision para executar o smoke test.")

    args = parse_args()
    image_path = pick_image(args.dataset, args.image)

    image_obj = MammogramImage(image_path)
    ImageProcessor.load_image(image_obj)
    ImageProcessor.segment_breast(image_obj)

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_DEFAULT_SIZE, IMAGE_DEFAULT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = ImageProcessor.to_display_image(image_obj.segmented).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    num_classes = 2 if args.task == "binary" else len(CLASS_NAMES)
    model = CNNTrainer.build_model("resnet18", num_classes=num_classes, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(tensor)

    print(f"OK image={image_path} shape={tuple(tensor.shape)} output={tuple(output.shape)}")


if __name__ == "__main__":
    main()
