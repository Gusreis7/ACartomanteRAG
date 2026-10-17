from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis

from src.config.config_manager import ConfigManager
from src.generation.graph import MachadoRAG
from src.generation.provider import create_generation_provider
from src.services.chat import ChatService
from src.services.conversations import ConversationService
from src.services.usage import UsageStats
from src.storage.redis_chat_repository import RedisChatRepository
from src.storage.redis_message_repository import RedisMessageRepository

logger = logging.getLogger("acartomante.runtime")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ConfigManager.load()

    redis_url = config.environment.redis.url
    if not redis_url:
        raise RuntimeError("REDIS_URL é obrigatório para o runtime da API.")

    redis = Redis.from_url(
        redis_url,
        decode_responses=True,
    )

    try:
        # Metadata das conversas usa um client Redis próprio.
        redis.ping()

        # RedisSaver.from_conn_string() retorna um context manager.
        # O saver precisa permanecer aberto por todo o lifespan da API.
        with RedisSaver.from_conn_string(redis_url) as checkpointer:
            checkpointer.setup()

            generation_provider = create_generation_provider(config.generation)

            rag = MachadoRAG(
                config,
                generation_provider=generation_provider,
                checkpointer=checkpointer,
            )

            generation_provider.warmup()

            repository = RedisChatRepository(redis)
            message_repository = RedisMessageRepository(redis)
            conversation_service = ConversationService(repository)
            chat_service = ChatService(
                repository=repository,
                message_repository=message_repository,
                rag=rag,
            )

            app.state.config = config
            app.state.redis = redis
            app.state.usage = UsageStats(redis)
            app.state.checkpointer = checkpointer
            app.state.generation_provider = generation_provider
            app.state.rag = rag
            app.state.message_repository = message_repository
            app.state.conversation_service = conversation_service
            app.state.chat_service = chat_service

            logger.info(
                "runtime_ready strategy=%s index=%s",
                config.retrieval_strategy,
                rag.retrieval.resolved.index_id,
            )

            yield
    finally:
        redis.close()
