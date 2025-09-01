import torch
from sentence_transformers import SparseEncoder, SparseEncoderModelCardData
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from sentence_transformers.sparse_encoder.losses import (
    SparseMarginMSELoss,
    SparseDistillKLDivLoss,
    SpladeLoss,
)
from .config import ModelCfg, LossCfg


def build_model(cfg: ModelCfg) -> SparseEncoder:
    """Build SPLADE v3 model"""
    return SparseEncoder(
        modules=[
            MLMTransformer(cfg.backbone_id),
            SpladePooling(pooling_strategy=cfg.pooling_strategy),
        ],
        model_card_data=SparseEncoderModelCardData(
            language=cfg.language,
            license=cfg.license,
            model_name=cfg.model_name,
        ),
    )


class SPLADEv3CombinedLoss(torch.nn.Module):
    """
    Combined loss for SPLADE v3 that mixes MarginMSE and KL-Divergence losses.

    From SPLADE v3:
    - λ_KL = 1.0 for KL-Div (focuses on Precision)
    - λ_MSE = 0.05 for MarginMSE (focuses on Recall)
    """

    def __init__(
        self,
        model: SparseEncoder,
        margin_weight: float = 0.05,  # λ_MSE from paper
        kl_weight: float = 1.0,  # λ_KL from paper
        kl_temperature: float = 2.0,
    ):
        super().__init__()
        self.model = model
        self.margin_loss = SparseMarginMSELoss(model=model)
        self.kl_loss = SparseDistillKLDivLoss(model=model, temperature=kl_temperature)
        self.margin_weight = margin_weight
        self.kl_weight = kl_weight

    def forward(self, sentence_features, labels):
        """
        Forward pass combining both losses.

        Args:
            sentence_features: Input sentence features
            labels: Should contain both margin and kl labels
                   - For margin loss: difference scores between pos and neg
                   - For KL loss: probability distribution scores
        """
        total_loss = 0.0

        # MarginMSE loss
        if isinstance(labels, dict) and "margin" in labels:
            margin_loss_val = self.margin_loss(sentence_features, labels["margin"])
            total_loss += self.margin_weight * margin_loss_val
        elif not isinstance(labels, dict):
            margin_loss_val = self.margin_loss(sentence_features, labels)
            total_loss += self.margin_weight * margin_loss_val

        # KL divergence loss
        if isinstance(labels, dict) and "kl" in labels:
            kl_loss_val = self.kl_loss(sentence_features, labels["kl"])
            total_loss += self.kl_weight * kl_loss_val

        return total_loss


def build_loss(model: SparseEncoder, loss_cfg: LossCfg) -> SpladeLoss:
    """
    Build SPLADE v3 loss function with combined MarginMSE + KL-Div loss.

    Uses exact weights from SPLADE v3:
    - λ_KL = 1.0 for KL-Div (focuses on Precision)
    - λ_MSE = 0.05 for MarginMSE (focuses on Recall)
    """

    # Create the combined loss with SPLADE v3 weights
    combined_loss = SPLADEv3CombinedLoss(
        model=model,
        margin_weight=loss_cfg.margin_weight,
        kl_weight=loss_cfg.kl_weight,
        kl_temperature=loss_cfg.kl_temperature,
    )

    # Wrap with SpladeLoss for sparsity regularization
    return SpladeLoss(
        model=model,
        loss=combined_loss,
        query_regularizer_weight=loss_cfg.query_regularizer_weight,
        document_regularizer_weight=loss_cfg.document_regularizer_weight,
    )
