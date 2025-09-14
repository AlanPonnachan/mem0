from typing import Optional

from pydantic import BaseModel, Field


class VespaConfig(BaseModel):
    collection_name: str = Field(
        default="mem0_vespa",
        description="Name of the Vespa schema.",
    )
    embedding_model_dims: int = Field(
        default=1536,
        description="Dimensions of the embedding model.",
    )
    url: Optional[str] = Field(
        default=None,
        description="URL of an existing Vespa instance (e.g., 'http://localhost:8080'). If provided, `local_deployment` is ignored."
    )
    local_deployment: bool = Field(
        default=False,
        description="If True, deploys a local Vespa instance via Docker. `url` should be None."
    )
    port: int = Field(
        default=8080,
        description="Port to use for local Docker deployment.",
    )
    content_cluster_name: str = Field(
        default="content_default",
        description="The name of the content cluster in Vespa."
    )