# modules.py
import torch
import torch.nn as nn
from torchvision import models


class ConvNeXtClassifier(nn.Module):
    """
    Thin wrapper around a torchvision ConvNeXt.
    - Swaps the final classifier to match `num_classes`
    - Gracefully handles grayscale inputs by repeating to 3 channels
    - Optional backbone freezing for quick experiments
    """

    def __init__(
        self,
        num_classes: int = 2,
        variant: str = "tiny",       # tiny | small | base | large (tiny is plenty here)
        pretrained: bool = True,     # use ImageNet weights for a better starting point
        freeze_backbone: bool = False
    ):
        super().__init__()

        # pick a ConvNeXt variant
        if variant == "tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
        elif variant == "small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
        elif variant == "base":
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_base(weights=weights)
        elif variant == "large":
            weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_large(weights=weights)
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")

        # replace the final linear layer to match our classes
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        # optionally freeze the feature extractor (good for a quick baseline)
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("classifier"):  # leave the new head trainable
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # our loaders give grayscale (C=1). ConvNeXt expects 3 channels.
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)   # cheap + effective channel expansion
        return self.backbone(x)


def build_model(
    num_classes: int = 2,
    variant: str = "tiny",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> ConvNeXtClassifier:
    """Small helper so train.py can import a single function."""
    return ConvNeXtClassifier(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


# ---------------------------------------------------------------------------
# quick self-test: run this file directly to verify shapes & forward pass
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=2, variant="tiny", pretrained=False).to(device)

    # fake batch: (batch=4, channels=1, H=224, W=224) -> should output (4, 2)
    x = torch.randn(4, 1, 224, 224, device=device)
    with torch.no_grad():
        logits = model(x)

    print("OK forward pass works.")
    print("Input :", tuple(x.shape))
    print("Output:", tuple(logits.shape))
