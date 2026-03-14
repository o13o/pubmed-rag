"""FastAPI dependencies — extract services from app.state."""

from fastapi import Request
from pymilvus import Collection

from src.retrieval.reranker import BaseReranker
from src.shared.config import Settings
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase


def get_collection(request: Request) -> Collection:
    return request.app.state.collection


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


def get_mesh_db(request: Request) -> MeSHDatabase:
    return request.app.state.mesh_db


def get_reranker_dep(request: Request) -> BaseReranker:
    return request.app.state.reranker


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
