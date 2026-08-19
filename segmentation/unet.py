"""
segmentation/unet.py
---------------------
U-Net (Ronneberger et al., 2015) for binary lesion segmentation, using a
pretrained (ImageNet) CNN encoder in place of a from-scratch downsampling
path. Used to produce the lesion mask that feeds the topology module
(topology/gudhi_features.py, topology/persistence.py).

Implementation note: rather than hand-rolling the encoder, this wraps
`segmentation_models_pytorch` (https://github.com/qubvel-org/segmentation_models.pytorch),
which builds the standard U-Net decoder (upsampling + skip connections)
on top of any torchvision/timm classification backbone. Swapping the
encoder for an ImageNet-pretrained one is a well-established way to get
better segmentation performance for free when you have the GPU budget to
fine-tune it (Sec. 10: RTX A4500 recommended) -- the model starts from
features that already understand edges/textures/shapes instead of
learning them from zero on ~2.6k ISIC2018 training masks.

The only from-scratch parts are the decoder (upsampling blocks + skip
connections) and the final 1x1 segmentation head, exactly as in a
standard U-Net -- only the encoder is pretrained.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config


class UNet(nn.Module):
    """
    U-Net with a pretrained encoder.

    Args:
        in_channels: input image channels (3 for RGB dermoscopic images)
        out_channels: output mask channels (1 for binary lesion mask)
        encoder_name: any encoder supported by segmentation_models_pytorch,
            e.g. "resnet34" (default -- good accuracy/speed trade-off),
            "resnet50" (more capacity, still fast), "efficientnet-b3"
            (matches the classification backbone family used elsewhere in
            this project), or "tu-<any timm model name>" for the full timm
            encoder zoo. Full list: https://smp.readthedocs.io/en/latest/encoders.html
        encoder_weights: "imagenet" (default) to fine-tune from pretrained
            weights, or None to train the encoder from scratch too (only
            useful for offline smoke tests where ImageNet weights can't be
            downloaded).
        freeze_encoder: if True, freezes the pretrained encoder and only
            trains the (randomly initialized) decoder + segmentation head.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 encoder_name: str = None, encoder_weights: str = "imagenet",
                 freeze_encoder: bool = False):
        super().__init__()
        encoder_name = encoder_name or config.UNET_ENCODER_NAME

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        if freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.model(x)  # raw logits; apply sigmoid outside for probability mask


@torch.no_grad()
def predict_mask(model: UNet, image: torch.Tensor, threshold: float = 0.5, device=None) -> torch.Tensor:
    """
    Run inference with a trained U-Net and return a binary mask.
    `image`: Tensor[B, 3, H, W], already normalized the same way as training.
    Returns: Tensor[B, 1, H, W] of {0., 1.}
    """
    model.eval()
    if device is not None:
        image = image.to(device)
    logits = model(image)
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


if __name__ == "__main__":
    # quick shape sanity check (encoder_weights=None avoids requiring
    # internet access to download ImageNet weights during a smoke test;
    # real training should use the default encoder_weights="imagenet")
    net = UNet(in_channels=3, out_channels=1, encoder_name="resnet34", encoder_weights=None)
    dummy = torch.randn(2, 3, 224, 224)
    out = net(dummy)
    print("UNet output shape:", out.shape)
    assert out.shape == (2, 1, 224, 224)
    print("OK")
