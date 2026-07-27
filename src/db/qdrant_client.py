import os
import uuid
from dataclasses import dataclass
from typing import List, Optional
from qdrant_client import QdrantClient, models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastembed import SparseTextEmbedding
from src.schemas.document_schemas import ParsedResume

# Section filters used by gap-analysis retrieval.
SKILL_RETRIEVAL_SECTIONS = ["technical_skills", "experience", "projects"]
DUTY_RETRIEVAL_SECTIONS = ["experience", "projects"]


@dataclass
class RetrievalHit:
    text: str
    section_type: str
    score: float


class VectorDatabase:
    def __init__(self, collection_name: str = "resume_chunks"):
        """
        Initializes the VectorDatabase client, establishes the connection to Qdrant, and sets up the embedding models.
        
        Inputs:
        - collection_name (str): The name of the Qdrant collection to use (defaults to 'resume_chunks').
        
        Returns:
        - None
        """
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_api_key:
            self.client = QdrantClient(url=self.qdrant_url, api_key=qdrant_api_key)
        else:
            self.client = QdrantClient(url=self.qdrant_url)
        self.collection_name = collection_name
        
        # Initialize Dense Embeddings via Gemini
        self.dense_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2"
        )
        # Sparse Embeddings (Exact Keyword / BM25 Search via FastEmbed)
        self.sparse_embeddings = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        self._ensure_collection()
        self._ensure_payload_indexes()

    def _ensure_collection(self):
        """
        Creates the Qdrant collection with BOTH dense and sparse vector configurations if it does not already exist.
        
        Inputs:
        - None (Uses instance attributes)
        
        Returns:
        - None
        """
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=3072, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )

    def _ensure_payload_indexes(self):
        """Create keyword indexes on filter fields (required by Qdrant Cloud)."""
        for field_name in ("resume_id", "section_type", "candidate_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                # Index already exists on redeploy / warm start.
                if "already exists" not in str(exc).lower():
                    print(f"Warning: payload index for {field_name}: {exc}")

    def index_resume(self, parsed_resume: ParsedResume, resume_id: str, candidate_id: str = "unknown"):
        """
        Chunks the parsed resume sections, generates dual embeddings (Dense + Sparse), and upserts them into Qdrant.
        
        Inputs:
        - parsed_resume (ParsedResume): The structured Pydantic object containing the extracted resume data.
        - resume_id (str): A unique identifier for the specific resume being indexed.
        - candidate_id (str): A unique identifier for the candidate (defaults to 'unknown').
        
        Returns:
            List of staged chunk payload dicts (text, section_type, resume_id, candidate_id).
        """
        all_texts = []
        all_payloads = []
        
        # Helper to stage data for batching
        def _stage_section(texts, section_type):
            for text in texts:
                if not text.strip():
                    continue
                all_texts.append(text)
                all_payloads.append({
                    "text": text,
                    "section_type": section_type,
                    "resume_id": resume_id,
                    "candidate_id": candidate_id
                })

        print("Staging texts for batch embedding...")
        _stage_section(parsed_resume.technical_skills, "technical_skills")
        _stage_section(parsed_resume.soft_skills, "soft_skills")
        _stage_section(parsed_resume.experience_sections, "experience")
        _stage_section(parsed_resume.projects, "projects")
        _stage_section(parsed_resume.domain_expertise, "domain_expertise")

        if not all_texts:
            print("Warning: No valid data found to index.")
            return []

        self.delete_resume_chunks(resume_id)

        print(f"Generating Dense embeddings for {len(all_texts)} chunks via Gemini API...")
        dense_vectors = self.dense_embeddings.embed_documents(all_texts)
        
        print("Generating Sparse vectors for keyword matching locally via FastEmbed...")
        # fastembed returns a generator, so we convert it to a list
        sparse_vectors = list(self.sparse_embeddings.embed(all_texts))
        
        points = []
        for i, text in enumerate(all_texts):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    # We pass both the Dense and Sparse vectors under their named keys
                    vector={
                        "dense": dense_vectors[i],
                        "sparse": models.SparseVector(
                            indices=sparse_vectors[i].indices.tolist(),
                            values=sparse_vectors[i].values.tolist()
                        )
                    },
                    payload=all_payloads[i]
                )
            )

        print(f"Upserting {len(points)} dual-vectors into Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Successfully indexed resume {resume_id} for Hybrid Search!")
        return all_payloads

    def delete_resume_chunks(self, resume_id: str) -> int:
        """Remove all Qdrant points for a resume_id before re-indexing."""
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="resume_id",
                            match=models.MatchValue(value=resume_id),
                        )
                    ]
                )
            ),
        )
        deleted = getattr(result, "operation_id", None)
        print(f"Deleted existing chunks for resume {resume_id} (operation={deleted})")
        return 0

    def list_indexed_chunks(self, resume_id: str, limit: int = 200) -> list:
        """
        Scroll Qdrant for all chunk payloads indexed under a resume_id.

        Returns:
            List of payload dicts (text, section_type, resume_id, ...).
        """
        chunks = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="resume_id",
                            match=models.MatchValue(value=resume_id),
                        )
                    ]
                ),
                limit=min(limit - len(chunks), 50),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in records:
                if point.payload:
                    chunks.append(dict(point.payload))
            if offset is None or len(chunks) >= limit:
                break
        return chunks

    def _resume_filter(
        self, resume_id: str, section_types: Optional[List[str]] = None
    ) -> models.Filter:
        must_conditions = [
            models.FieldCondition(
                key="resume_id",
                match=models.MatchValue(value=resume_id),
            )
        ]
        should_conditions = []
        if section_types:
            should_conditions = [
                models.FieldCondition(
                    key="section_type",
                    match=models.MatchValue(value=st),
                )
                for st in section_types
            ]
        return models.Filter(
            must=must_conditions,
            should=should_conditions or None,
        )

    def retrieve_evidence_hits(
        self,
        query: str,
        resume_id: str,
        limit: int = 3,
        score_threshold: float = 0.4,
        section_types: Optional[List[str]] = None,
    ) -> List[RetrievalHit]:
        """
        Hybrid search returning ranked hits with section_type and RRF score.
        """
        dense_vector = self.dense_embeddings.embed_query(query)
        sparse_result = list(self.sparse_embeddings.query_embed(query))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )
        resume_filter = self._resume_filter(resume_id, section_types)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=20,
                    filter=resume_filter,
                    score_threshold=score_threshold,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=20,
                    filter=resume_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        hits: List[RetrievalHit] = []
        for point in search_result.points:
            text = point.payload.get("text", "")
            if not text:
                continue
            hits.append(
                RetrievalHit(
                    text=text,
                    section_type=point.payload.get("section_type", ""),
                    score=float(point.score or 0.0),
                )
            )
        return hits

    def retrieve_evidence(
        self,
        query: str,
        resume_id: str,
        limit: int = 3,
        score_threshold: float = 0.4,
        section_types: Optional[List[str]] = None,
    ) -> list:
        """
        Executes a Hybrid Search (Dense + Sparse) against Qdrant to find the most relevant resume chunks for a given query.
        
        Inputs:
        - query (str): The search string (usually a job requirement from the Job Description).
        - resume_id (str): The unique identifier of the resume to restrict the search to.
        - limit (int): The maximum number of text chunks to return (defaults to 3).
        - score_threshold (float): The minimum cosine similarity score required for the semantic match (defaults to 0.4).
        - section_types (Optional[List[str]]): If set, restrict search to these payload section_type values (e.g. experience, projects).
        
        Returns:
        - list: A list of strings representing the top matching evidence extracted from the candidate's resume.
        """
        hits = self.retrieve_evidence_hits(
            query=query,
            resume_id=resume_id,
            limit=limit,
            score_threshold=score_threshold,
            section_types=section_types,
        )
        return [hit.text for hit in hits]
