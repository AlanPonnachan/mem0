import json
import logging
import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel

try:
    from vespa.application import Application, Vespa
    from vespa.deployment import VespaDocker
    from vespa.package import (
        ApplicationPackage,
        Document,
        Field,
        FieldSet,
        RankProfile,
        Schema,
    )
except ImportError:
    raise ImportError(
        "The 'pyvespa' library is required. Please install it using 'pip install mem0[vespa]'."
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[Dict]


class VespaDB(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        url: Optional[str] = None,
        local_deployment: bool = False,
        port: int = 8080,
        content_cluster_name: str = "content_default",
    ):
        """
        Initialize the Vespa vector store.

        This client can either connect to an existing Vespa instance or deploy a new
        local instance using Docker.

        Args:
            collection_name (str): Name of the Vespa schema.
            embedding_model_dims (int): Dimensions of the embedding model.
            url (Optional[str]): URL of an existing Vespa instance (e.g., 'http://localhost:8080').
            local_deployment (bool): If True, deploys a local Vespa instance via Docker. `url` is ignored.
            port (int): Port to use for local Docker deployment.
            content_cluster_name (str): The name of the content cluster in Vespa.
                                        Defaults to 'content_default', which is standard for local deployments.
        """
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.port = port
        self.content_cluster_name = content_cluster_name
        self.vespa_docker = None
        
        if local_deployment:
            self.app = self._deploy_local_app()
        elif url:
            self.app = Vespa(url=url)
        else:
            raise ValueError("Either `url` must be provided to connect to an existing Vespa instance, or `local_deployment` must be set to True.")

    def _deploy_local_app(self) -> Application:
        """Deploys a Vespa application using a Docker container."""
        logger.info(f"Deploying local Vespa application '{self.collection_name}'...")
        schema = Schema(
            name=self.collection_name,
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["summary", "attribute"]),
                    Field(name="payload", type="string", indexing=["summary"]),
                    Field(
                        name="vector",
                        type=f"tensor<float>(d0[{self.embedding_model_dims}])", # Using conventional 'd0'
                        indexing=["attribute", "summary"],
                        attribute=["distance-metric: cosine"],
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["id", "payload"])],
            rank_profiles=[
                RankProfile(
                    name="default",
                    inputs=[("query(q_vec)", f"tensor<float>(d0[{self.embedding_model_dims}])")], # Match 'd0'
                    first_phase="closeness(vector)",
                )
            ],
        )

        package = ApplicationPackage(name=self.collection_name.lower(), schemas=[schema])
        
        try:
            self.vespa_docker = VespaDocker(port=self.port)
            app = self.vespa_docker.deploy(application_package=package)
            logger.info("Vespa application deployed successfully.")
            return app
        except Exception as e:
            logger.error(f"Failed to deploy Vespa application: {e}")
            self.close()
            raise RuntimeError(
                "Could not deploy Vespa. Make sure Docker is running and the port is available."
            ) from e

    def create_col(self, name, vector_size, distance):
        """Collection is configured at initialization. This method is a no-op."""
        logger.info("Vespa collection (schema) is configured at initialization.")
        pass

    def insert(self, vectors: List[List[float]], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Insert or update vectors. Vespa's feed operation is an upsert.

        Args:
            vectors: List of vectors to insert.
            payloads: List of payloads corresponding to vectors.
            ids: List of IDs corresponding to vectors.
        """
        if not ids:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if not payloads:
            payloads = [{} for _ in vectors]

        batch = []
        for vec_id, vector, payload in zip(ids, vectors, payloads):
            data_point = {
                "id": str(vec_id),
                "fields": {
                    "id": str(vec_id),
                    "vector": vector,
                    "payload": json.dumps(payload),
                },
            }
            batch.append(data_point)
        
        try:
            logger.debug(f"Inserting {len(batch)} vectors into schema {self.collection_name}")
            self.app.feed_batch(batch=batch, schema=self.collection_name)
        except Exception as e:
            logger.error(f"Failed to insert batch into Vespa: {e}")
            raise RuntimeError("Vespa insert operation failed.") from e

    def search(self, query: str, vectors: List[float], limit: int = 5, filters: Optional[Dict] = None) -> List[OutputData]:
        """
        Search for similar vectors.

        Note: The `query` parameter (text) is currently unused. Search is purely vector-based.
        Filtering is performed client-side, which may result in fewer than `limit` results if filters are strict.

        Args:
            query (str): The text query (currently unused).
            vectors (List[float]): The vector to search for.
            limit (int): The maximum number of results to return.
            filters (Optional[Dict]): A dictionary of metadata to filter by.
        
        Returns:
            List[OutputData]: A list of search results.
        """
        nearest_neighbor_clause = f"({{targetHits: {limit}}})nearestNeighbor(vector, q_vec)"
        yql = f"select * from sources * where {nearest_neighbor_clause};"

        body = {
            "yql": yql,
            "hits": limit * 5, # Fetch more for potential client-side filtering
            "input.query(q_vec)": vectors, # Corrected input shape
            "ranking.profile": "default",
        }

        try:
            response = self.app.query(body=body)
        except Exception as e:
            logger.error(f"Vespa query failed: {e}")
            raise RuntimeError("Vespa search operation failed.") from e
        
        results = []
        if not response.hits:
            return results

        for hit in response.hits:
            try:
                payload = json.loads(hit["fields"].get("payload", "{}"))
                
                if filters:
                    match = all(payload.get(key) == value for key, value in filters.items())
                    if not match:
                        continue

                results.append(
                    OutputData(
                        id=hit["fields"]["id"],
                        score=hit["relevance"],
                        payload=payload,
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing hit {hit.get('id', 'N/A')}: {e}")
        
        return results[:limit]

    def delete(self, vector_id: str):
        self.app.delete_data(schema=self.collection_name, doc_id=str(vector_id))

    def get(self, vector_id: str) -> Optional[OutputData]:
        response = self.app.get_data(schema=self.collection_name, doc_id=str(vector_id))
        if response.is_successful() and "fields" in response.json:
            fields = response.json["fields"]
            payload = json.loads(fields.get("payload", "{}"))
            return OutputData(id=fields["id"], score=None, payload=payload)
        return None

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict] = None):
        """
        Updates a vector's payload or replaces the entire document if a vector is provided.
        
        Args:
            vector_id: The ID of the document to update.
            vector: If provided, the entire document is replaced with this new vector.
            payload: Metadata to update. If `vector` is not provided, this performs a partial update.
        
        Raises:
            ValueError: If the document with `vector_id` does not exist.
        """
        if not vector and not payload:
            logger.warning("Update called without vector or payload.")
            return

        existing_doc = self.get(str(vector_id))
        if not existing_doc:
            raise ValueError(f"Document with ID {vector_id} not found, cannot update.")

        if vector:
            # Full replacement (upsert)
            updated_payload = (existing_doc.payload or {}).copy()
            updated_payload.update(payload or {})
            self.insert(vectors=[vector], payloads=[updated_payload], ids=[str(vector_id)])
        else:
            # Partial update for payload only
            updated_payload = (existing_doc.payload or {}).copy()
            updated_payload.update(payload or {})
            update_fields = {"payload": json.dumps(updated_payload)}
            self.app.update_data(
                schema=self.collection_name,
                doc_id=str(vector_id),
                fields=update_fields
            )

    def list_cols(self) -> List[str]:
        # TODO: Implement a more robust method by querying Vespa's application management API.
        return [self.collection_name]

    def delete_col(self):
        logger.info(f"Clearing all documents from Vespa schema '{self.collection_name}'.")
        try:
            # For modern pyvespa versions
            self.app.delete_all_docs(content_cluster_name=self.content_cluster_name, schema=self.collection_name)
        except AttributeError:
            logger.warning("`delete_all_docs` not found in this version of pyvespa. Falling back to manual deletion. This may be slow.")
            # Fallback for older pyvespa versions
            docs_to_delete = self.list(limit=10000) # Arbitrary large number
            for doc in docs_to_delete:
                self.delete(doc.id)

    def close(self):
        """Stops and removes the Vespa Docker container if it was locally deployed."""
        if self.vespa_docker and self.vespa_docker.container:
            logger.info("Stopping and removing local Vespa Docker container...")
            try:
                self.vespa_docker.container.stop()
                self.vespa_docker.container.remove()
            except Exception as e:
                logger.error(f"Error while closing Vespa Docker container: {e}")
        self.vespa_docker = None

    def col_info(self) -> dict:
        response = self.app.query(body={"yql": f"select * from {self.collection_name} where true limit 0;"})
        return {
            "name": self.collection_name,
            "count": response.json.get("root", {}).get("fields", {}).get("totalCount", 0),
            "dimension": self.embedding_model_dims,
        }

    def list(self, filters: Optional[Dict] = None, limit: int = 100) -> List[OutputData]:
        response = self.app.query(body={"yql": f"select * from sources * where true limit {limit};", "hits": limit})
        
        results = []
        if not response.hits:
            return results

        for hit in response.hits:
            payload = json.loads(hit["fields"].get("payload", "{}"))
            if filters:
                match = all(payload.get(key) == value for key, value in filters.items())
                if not match:
                    continue
            results.append(OutputData(id=hit["fields"]["id"], score=None, payload=payload))

        return results

    def reset(self):
        """Resets the vector store."""
        logger.warning(f"Resetting schema {self.collection_name}...")
        if self.vespa_docker:
            # If it was a local deployment, fully restart it
            self.close()
            self.app = self._deploy_local_app()
        else:
            # If connected to a remote instance, just clear the documents
            self.delete_col()
    
    def __del__(self):
        """
        Best-effort cleanup of the Docker container.
        
        Note: Python does not guarantee __del__ is called on process exit.
        For reliable cleanup in applications, explicitly call the .close() method.
        """
        self.close()