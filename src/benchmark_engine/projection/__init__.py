"""Optional model projection package."""

from .base import ModelProjection, ProjectionMapping
from .deepseek_v4 import DEEPSEEK_V4_PROJECTION, DeepSeekV4Projection
from .deepseek_v4_flash import (
    DEEPSEEK_V4_FLASH_MI300X_PROJECTION,
    DeepSeekV4FlashMi300xProjection,
)
from .glm5 import GLM5_PROJECTION, Glm5Projection

PROJECTIONS = (
    DEEPSEEK_V4_PROJECTION,
    DEEPSEEK_V4_FLASH_MI300X_PROJECTION,
    GLM5_PROJECTION,
)


def projection_for_id(projection_id):
    return next((projection for projection in PROJECTIONS
                 if projection.projection_id == projection_id), None)


def projection_for_case(operator_id, case):
    for projection in PROJECTIONS:
        mapping = projection.mapping_for_case(operator_id, case)
        if mapping is not None:
            return projection, mapping
    return None, None

__all__ = [
    "DEEPSEEK_V4_PROJECTION",
    "DeepSeekV4Projection",
    "DEEPSEEK_V4_FLASH_MI300X_PROJECTION",
    "DeepSeekV4FlashMi300xProjection",
    "GLM5_PROJECTION",
    "Glm5Projection",
    "ModelProjection",
    "PROJECTIONS",
    "ProjectionMapping",
    "projection_for_case",
    "projection_for_id",
]
