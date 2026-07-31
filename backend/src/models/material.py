from __future__ import annotations

from datetime import datetime
from typing import Any

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from pymongo import IndexModel

from src.models.common import Modality, PipelineStatus, SourceLanguage, utc_now


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class MaterialBlock(BaseModel):
    block_id: str
    block_index: int
    block_type: str
    content: str
    language: str = SourceLanguage.UNKNOWN.value
    bbox: BoundingBox | None = None
    ocr_confidence: float | None = None
    reading_order: int
    extra: dict[str, Any] = Field(default_factory=dict)


class MaterialPage(BaseModel):
    page_number: int
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    ocr_confidence: float | None = None
    blocks: list[MaterialBlock] = Field(default_factory=list)


class MaterialPageDocument(Document):
    owner_id: str
    collection_id: PydanticObjectId
    material_id: PydanticObjectId
    page_number: int
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    ocr_confidence: float | None = None
    blocks: list[MaterialBlock] = Field(default_factory=list)

    class Settings:
        name = "material_pages"
        indexes = [
            IndexModel([("material_id", 1), ("page_number", 1)], name="material_pages_material_page", unique=True),
            IndexModel([("owner_id", 1), ("collection_id", 1), ("material_id", 1)], name="material_pages_scope"),
            IndexModel([("blocks.block_id", 1)], name="material_pages_block_id"),
        ]

    def to_material_page(self) -> MaterialPage:
        return MaterialPage(
            page_number=self.page_number,
            image_path=self.image_path,
            width=self.width,
            height=self.height,
            ocr_confidence=self.ocr_confidence,
            blocks=self.blocks,
        )


class Material(Document):
    owner_id: str
    collection_id: PydanticObjectId
    filename: str
    original_name: str
    file_type: str
    modality: str = Modality.MIXED.value
    language: str = SourceLanguage.UNKNOWN.value
    subject: str | None = None
    topic: str | None = None
    version: str = "v1.0"
    checksum_sha256: str
    page_count: int | None = None
    file_size_bytes: int
    storage_path: str
    status: str = PipelineStatus.UPLOADED.value
    error_message: str | None = None
    failed_stage: str | None = None
    retry_count: int = 0
    parse_version: str
    chunk_version: str
    embedding_version: str
    index_version: str
    extra_metadata: dict[str, Any] = Field(default_factory=dict)
    pages: list[MaterialPage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "materials"
        indexes = [
            IndexModel(
                [("owner_id", 1), ("collection_id", 1), ("status", 1), ("language", 1), ("subject", 1)],
                name="materials_scope_status_language_subject",
            )
        ]


async def replace_material_pages(material: Material, pages: list[MaterialPage]) -> None:
    await MaterialPageDocument.find(MaterialPageDocument.material_id == material.id).delete()
    if not pages:
        return
    documents = [
        MaterialPageDocument(
            owner_id=material.owner_id,
            collection_id=material.collection_id,
            material_id=material.id,
            page_number=page.page_number,
            image_path=page.image_path,
            width=page.width,
            height=page.height,
            ocr_confidence=page.ocr_confidence,
            blocks=page.blocks,
        )
        for page in pages
    ]
    # Insert in batches of 10 pages to avoid MongoDB 16MB BSON limit on bulk command size.
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        await MaterialPageDocument.insert_many(documents[i : i + batch_size])


async def get_material_pages(material: Material) -> list[MaterialPage]:
    try:
        documents = await MaterialPageDocument.find(
            MaterialPageDocument.material_id == material.id
        ).sort("page_number").to_list()
    except Exception:
        documents = []
    if documents:
        return [document.to_material_page() for document in documents]
    return list(getattr(material, "pages", []) or [])


async def get_material_pages_by_material_ids(materials: list[Material]) -> dict[str, list[MaterialPage]]:
    material_by_id = {material.id: material for material in materials if material.id is not None}
    if not material_by_id:
        return {}
    pages_by_material: dict[str, list[MaterialPage]] = {str(material_id): [] for material_id in material_by_id}
    try:
        documents = await MaterialPageDocument.find(
            {"material_id": {"$in": list(material_by_id.keys())}}
        ).to_list()
        for document in sorted(documents, key=lambda doc: (str(doc.material_id), doc.page_number)):
            pages_by_material[str(document.material_id)].append(document.to_material_page())
    except Exception:
        # Fallback for embedded/test backends that cannot serialize PydanticObjectId in $in.
        import asyncio as _asyncio

        async def _fetch(mat_id: PydanticObjectId) -> list[MaterialPageDocument]:
            try:
                return await MaterialPageDocument.find(
                    MaterialPageDocument.material_id == mat_id
                ).sort("page_number").to_list()
            except Exception:
                return []

        results = await _asyncio.gather(*[_fetch(mid) for mid in material_by_id])
        for mat_id, docs in zip(material_by_id, results):
            key = str(mat_id)
            if docs:
                pages_by_material[key] = [d.to_material_page() for d in docs]
    for material_id, material in material_by_id.items():
        key = str(material_id)
        if not pages_by_material.get(key):
            pages_by_material[key] = list(getattr(material, "pages", []) or [])
    return pages_by_material


async def get_material_page(material: Material, page_number: int) -> MaterialPage | None:
    try:
        document = await MaterialPageDocument.find_one(
            MaterialPageDocument.material_id == material.id,
            MaterialPageDocument.page_number == page_number,
        )
    except Exception:
        document = None
    if document is not None:
        return document.to_material_page()
    return next((page for page in getattr(material, "pages", []) or [] if page.page_number == page_number), None)
