# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alexandra Bernadotte
"""Assemble the LLM router for a deployment (shared by the API and the worker).

ДИАЛЕКТ БЭКЕНДА — ПО АДРЕСУ: «/v1» на конце пути = OpenAI-совместимый сервер (SGLang/vLLM),
иначе Ollama. Правило одно для адреса чата, адреса эмбеддера и каждого маршрута по модели
(``model=http://host:30000/v1``), и оно же должно читаться пробами здоровья хостов — иначе проба
спрашивала бы ``/api/tags`` у сервера, который отвечает только на ``/models``, и краснела бы на
живом хосте. Заведено 12.09.2026 под пилот SGLang: доверие к адресу при этом НЕ зависит от
диалекта — ``check_model_url`` проходят все адреса до того, как здесь из них построят провайдера.

``settings`` — :class:`llmkit.config.ModelConfig` или любой объект с теми же полями.
"""

from __future__ import annotations

from collections.abc import Callable

from llmkit.base import LLMProvider
from llmkit.claude_cli import ClaudeCliProvider
from llmkit.cloud import AnthropicProvider
from llmkit.config import ModelConfig
from llmkit.hosts import ResolvedHosts
from llmkit.metering import MeteredRouter
from llmkit.ollama import OllamaProvider
from llmkit.openai_compat import OpenAICompatProvider, is_openai_url
from llmkit.router import LLMRouter
from llmkit.vllm import ModelRoutedProvider, SplitTrustedProvider, VLLMProvider

#: A hook over the deployment's cloud provider: it receives the default one (``None`` when none is
#: configured) and returns the provider to register (or ``None``) — for example a provider that
#: answers some callers with another key and falls back to the default for everyone else. One
#: cloud provider either way, so every "is there a cloud model" check reads the same.
CloudWrapper = Callable[[LLMProvider | None], LLMProvider | None]


def _build_trusted(settings: ModelConfig, hosts: ResolvedHosts | None = None) -> LLMProvider:
    """The trusted chat+embed provider.

    ``chat_backend="vllm"`` routes generation to a continuous-batched vLLM server and keeps
    embeddings on their own trusted backend (a composite, so the router stays untouched).

    ``embed_base_url`` splits the OLLAMA path the same way, and for the same reason — which is not
    hypothetical. Measured 01.09.2026: one host served both, a research job filled its two slots
    with generations that outran their own client timeout, and every embedding queued behind them
    for hours. The resource watchdog had been saying so out loud ("Model(s) pinned in VRAM
    indefinitely — this can starve the embedder and stall ingest"). Chat is heavy, rare and
    tolerant of latency; embeddings are light, constant, and block ingest when they stall. They do
    not belong on one queue merely because they speak one protocol.

    Empty ``embed_base_url`` keeps the single-provider shape byte for byte, so a deployment that
    has not asked for the split does not silently acquire a second backend."""
    if settings.chat_backend == "vllm":
        chat = VLLMProvider(
            base_url=settings.vllm_base_url,
            chat_model=settings.vllm_chat_model,
            embed_model=settings.ollama_embed_model,
            timeout=settings.vllm_timeout,
        )
        embedder = OllamaProvider(
            base_url=(hosts.embed or hosts.chat) if hosts
            else settings.trusted_embed_base_url or settings.trusted_ollama_base_url,
            default_model=settings.ollama_chat_model,
            embed_model=settings.ollama_embed_model,
            timeout=settings.ollama_timeout,
            keep_alive=settings.ollama_keep_alive,
            query_prefix=settings.ollama_embed_query_prefix,
        )
        return SplitTrustedProvider(chat=chat, embed_provider=embedder)
    def _ollama(base_url: str) -> OllamaProvider:
        return OllamaProvider(
            base_url=base_url,
            default_model=settings.ollama_chat_model,
            embed_model=settings.ollama_embed_model,
            timeout=settings.ollama_timeout,
            keep_alive=settings.ollama_keep_alive,
            query_prefix=settings.ollama_embed_query_prefix,
        )

    def _backend(base_url: str) -> LLMProvider:
        """«/v1» = OpenAI-совместимый сервер (SGLang/vLLM), иначе Ollama — см. докстроку модуля.

        Имена моделей, таймаут и префикс запроса — ТЕ ЖЕ настройки: у сервера другой провод,
        а не другая модель. ``keep_alive`` в протоколе OpenAI отсутствует, там резидентность
        решает сам сервер."""
        if is_openai_url(base_url):
            return OpenAICompatProvider(
                base_url=base_url,
                default_model=settings.ollama_chat_model,
                embed_model=settings.ollama_embed_model,
                timeout=settings.ollama_timeout,
                query_prefix=settings.ollama_embed_query_prefix,
            )
        return _ollama(base_url)

    # АДРЕСА БЕРУТСЯ ТОЛЬКО ЧЕРЕЗ ПРОВЕРЕННЫЕ СВОЙСТВА. `Plane.TRUSTED` — атрибут класса
    # провайдера и об адресе ничего не знает; без этой проверки «доверенная плоскость»
    # означала бы «мы воспользовались клиентом Ollama», а не «байты остались внутри периметра».
    # АДРЕСА МОГУТ ПРИЙТИ СНАРУЖИ. `hosts` — уже разрешённые и уже проверенные значения
    # (переопределение администратора поверх окружения, см. llmkit.hosts). Без них берём
    # окружение, как раньше, — так ведут себя скрипты и все, кому реестр недоступен.
    chat_url = hosts.chat if hosts else settings.trusted_ollama_base_url
    trusted: LLMProvider = _backend(chat_url)
    #: Модели, чьи веса лежат на другой машине, обслуживаются ею — см. ModelRoutedProvider.
    routes = {
        model: _backend(url)
        for model, url in (hosts.model_hosts if hosts else settings.trusted_model_host_map).items()
        if url != chat_url
    }
    if routes:
        trusted = ModelRoutedProvider(default=trusted, routes=routes, default_model=settings.ollama_chat_model)
    embed_url = hosts.embed if hosts else settings.trusted_embed_base_url
    if not embed_url or embed_url == chat_url:
        return trusted
    return SplitTrustedProvider(
        chat=trusted,
        embed_provider=_backend(embed_url),
    )


def build_router(settings: ModelConfig, hosts: ResolvedHosts | None = None, *,
                 wrap_cloud: CloudWrapper | None = None) -> MeteredRouter:
    """The trusted provider is always registered. A cloud provider is added only when
    one is configured — and even then the router invariant ensures it can only ever receive
    de-identified, ≤INTERNAL data.

    Cloud backend precedence: a real Anthropic API key wins; otherwise the MVP on-machine
    ``claude`` CLI (reusing the operator's OAuth subscription) is used when enabled. ``wrap_cloud``
    (see :data:`CloudWrapper`) then gets the last word on the cloud provider — except under
    air-gap, where there is no cloud plane to wrap."""
    providers = [_build_trusted(settings, hosts)]
    # Air-gap: register NO cloud provider, so the router physically cannot route off-premise.
    if settings.airgap:
        return MeteredRouter(LLMRouter(providers))

    cloud: LLMProvider | None = None
    if settings.anthropic_api_key:
        cloud = AnthropicProvider(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    elif settings.claude_cli_enabled:
        cloud = ClaudeCliProvider(
            binary=settings.claude_cli_path,
            model=settings.claude_cli_model,
            timeout=settings.claude_cli_timeout,
        )
    if wrap_cloud is not None:
        cloud = wrap_cloud(cloud)
    if cloud is not None:
        providers.append(cloud)
    return MeteredRouter(LLMRouter(providers))
