import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import ResNet50_Weights

def get_text_detector(num_classes=2):
    """
    Build a Faster R-CNN model for text detection with:
      - ResNet50-FPN backbone (pretrained on ImageNet)
      - RPN anchor sizes tuned for small text (8,16,32 px)
      - ROI pooling over all FPN levels
      - New predictor head with `num_classes` (background + text)
    """
    # Define small anchor sizes for all FPN levels
    anchor_sizes = (8, 16, 32)
    anchor_generator = AnchorGenerator(
        sizes=(anchor_sizes,) * 5,
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    # ROI pooler across all feature maps
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
                                    output_size=7, sampling_ratio=2)
    # Load Faster R-CNN model (backbone pretrained, detection head not pretrained)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    # Replace classifier head for the given number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Alias (for use in inference code)
get_model = get_text_detector

if __name__ == "__main__":
    # Quick sanity check
    m = get_text_detector()
    print(m)
