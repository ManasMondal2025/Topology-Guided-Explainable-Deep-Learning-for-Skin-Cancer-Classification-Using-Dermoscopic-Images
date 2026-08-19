"""
inference.py
--------------
End-to-end single-image inference (Sec. 11 of the proposal: "Expected
Outputs" — prediction + confidence, Grad-CAM heatmap, SHAP plot).

Pipeline for one image:
    1. U-Net segments the lesion -> binary mask
    2. topology/extract.py computes the topological feature vector from the mask
    3. The hybrid model (ViT/EfficientNet/MobileNetV3 + fusion) predicts benign/malignant
    4. Grad-CAM shows WHERE the model looked
    5. SHAP shows WHICH topological features contributed most

Usage:
    python inference.py --image path/to/lesion.jpg --checkpoint checkpoints/hybrid_model_best.pth \
        --unet-checkpoint checkpoints/unet_segmentation_best.pth --output-dir outputs/inference_result
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from models.hybrid_model import HybridSkinCancerModel
from segmentation.unet import UNet
from segmentation.predict_masks import predict_mask_for_image
from topology.extract import extract_topological_features
from topology.gudhi_features import compute_shape_descriptors, compute_betti_numbers
from explainability.gradcam import get_gradcam_heatmap, overlay_heatmap_on_image
from explainability.shap_explain import explain_topological_contributions, format_shap_report
from utils.common import get_device, load_checkpoint, setup_logging
from utils.transforms import IMAGENET_MEAN, IMAGENET_STD, get_classification_transforms


def parse_args():
    p = argparse.ArgumentParser(description="Run inference on a single dermoscopic image")
    p.add_argument("--image", type=str, required=True, help="Path to a dermoscopic image (jpg/png/bmp).")
    p.add_argument("--checkpoint", type=str, default=str(config.HYBRID_CHECKPOINT_BEST))
    p.add_argument("--unet-checkpoint", type=str, default=str(config.UNET_CHECKPOINT))
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--output-dir", type=str, default=str(config.OUTPUT_DIR / "inference_result"))
    p.add_argument("--skip-shap", action="store_true", help="Skip the (slower) SHAP explanation step.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold on P(malignant). Lower than 0.5 flags more borderline cases "
                        "as malignant (higher recall, more false alarms) -- use evaluate.py "
                        "--sweep-thresholds on your validation set to choose this deliberately.")
    return p.parse_args()


def load_hybrid_model(checkpoint_path, device):
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    backbone_name = ckpt.get("backbone_name", config.BACKBONE_NAME)
    fusion_type = ckpt.get("fusion_type", config.FUSION_TYPE)
    model = HybridSkinCancerModel(backbone_name=backbone_name, fusion_type=fusion_type, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, backbone_name


def load_unet_model(checkpoint_path, device):
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model = UNet(
        in_channels=3, out_channels=1,
        encoder_name=ckpt.get("encoder_name", config.UNET_ENCODER_NAME),
        encoder_weights=None,  # weights come from the checkpoint's state_dict below
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def preprocess_image_for_model(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (image_size, image_size))
    norm = (resized.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def main():
    args = parse_args()
    setup_logging()
    device = get_device(config.DEVICE)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --- 1. Segmentation -----------------------------------------------
    logging.info("Step 1/5: segmenting lesion with U-Net...")
    unet = load_unet_model(args.unet_checkpoint, device)
    mask = predict_mask_for_image(unet, image_bgr, args.image_size, device)  # HxW uint8 {0,255}, original resolution
    cv2.imwrite(str(output_dir / "lesion_mask.png"), mask)

    # --- 2. Topological features -----------------------------------------
    logging.info("Step 2/5: extracting topological features...")
    mask_resized = cv2.resize(mask, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
    topo_vector = extract_topological_features(mask_resized)
    topo_tensor = torch.from_numpy(topo_vector).float().unsqueeze(0)

    shape_descriptors = compute_shape_descriptors(mask_resized)
    betti = compute_betti_numbers(mask_resized)

    # --- 3. Classification ------------------------------------------------
    logging.info("Step 3/5: running hybrid classifier...")
    model, backbone_name = load_hybrid_model(args.checkpoint, device)
    image_tensor = preprocess_image_for_model(image_rgb, args.image_size)

    with torch.no_grad():
        logits = model(image_tensor.to(device), topo_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        malignant_prob = float(probs[1].item())
        pred_class = 1 if malignant_prob >= args.threshold else 0
        confidence = malignant_prob if pred_class == 1 else 1 - malignant_prob

    prediction = {
        "predicted_class": config.CLASS_NAMES[pred_class],
        "confidence": confidence,
        "decision_threshold": args.threshold,
        "probabilities": {name: float(p) for name, p in zip(config.CLASS_NAMES, probs.cpu().numpy())},
        "betti_numbers": betti,
        "shape_descriptors": shape_descriptors,
    }
    logging.info(f"Prediction: {prediction['predicted_class']} (confidence={confidence:.2%})")

    # --- 4. Grad-CAM --------------------------------------------------
    logging.info("Step 4/5: generating Grad-CAM explanation...")
    grayscale_cam = get_gradcam_heatmap(
        model, image_tensor, topo_tensor, backbone_name=backbone_name, target_class=pred_class, device=device
    )
    display_image = cv2.resize(image_rgb, (args.image_size, args.image_size)).astype(np.float32) / 255.0
    cam_overlay = overlay_heatmap_on_image(display_image, grayscale_cam)
    cv2.imwrite(str(output_dir / "gradcam_heatmap.png"), cam_overlay)

    # --- 5. SHAP --------------------------------------------------------
    shap_report_text = None
    if not args.skip_shap:
        logging.info("Step 5/5: computing SHAP feature contributions "
                      "(this can take up to a minute)...")
        shap_result = explain_topological_contributions(model, image_tensor, topo_tensor, device=device)
        shap_report_text = format_shap_report(shap_result)
        prediction["shap_contributions"] = shap_result
        logging.info("\n" + shap_report_text)
    else:
        logging.info("Step 5/5: skipped (--skip-shap).")

    # --- Save everything --------------------------------------------------
    with open(output_dir / "prediction.json", "w") as f:
        json.dump(prediction, f, indent=2)
    if shap_report_text:
        with open(output_dir / "shap_report.txt", "w") as f:
            f.write(shap_report_text)

    logging.info(f"All outputs saved to {output_dir}/ "
                  "(prediction.json, lesion_mask.png, gradcam_heatmap.png"
                  + (", shap_report.txt)" if shap_report_text else ")"))


if __name__ == "__main__":
    main()
