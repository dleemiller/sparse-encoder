from sentence_transformers import SparseEncoder, SparseEncoderModelCardData
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from sentence_transformers.sparse_encoder.losses import SparseMarginMSELoss, SpladeLoss
from .config import ModelCfg, LossCfg

def build_model(cfg: ModelCfg) -> SparseEncoder:
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

def build_loss(model: SparseEncoder, loss_cfg: LossCfg) -> SpladeLoss:
    return SpladeLoss(
        model=model,
        loss=SparseMarginMSELoss(model=model),
        query_regularizer_weight=loss_cfg.query_regularizer_weight,
        document_regularizer_weight=loss_cfg.document_regularizer_weight,
    )

