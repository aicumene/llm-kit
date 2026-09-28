# llm-kit

A model toolkit: providers behind a privacy-gated router, a task dispatcher with memoization,
response shaping, per-task token metering and token savers. Python package `llmkit`.

The core rule is the router's invariant: a request whose sensitivity is above `INTERNAL` can never
reach a cloud-plane provider, and a "trusted" provider is only built on an address inside the
configured perimeter (`trust.check_model_url`).

## Layout

```
src/llmkit/
  base.py            LLMRequest / LLMResponse / LLMMessage, LLMProvider, Plane, DOCUMENT / QUERY
  classification.py  Sensitivity and the crossing rule (max_crossable, may_cross_to_cloud)
  errors.py          LLMError / LLMUnavailable / LLMBackendError, transport-error decorators
  router.py          LLMRouter, PrivacyViolation — the single chokepoint
  tracing.py         optional span hook (set_span_factory)
  ollama.py          OllamaProvider
  openai_compat.py   OpenAICompatProvider for "/v1" addresses (SGLang, vLLM), is_openai_url
  vllm.py            VLLMProvider, ModelRoutedProvider, SplitTrustedProvider
  claude_cli.py      ClaudeCliProvider (headless `claude` CLI, cloud plane)
  cloud.py           AnthropicProvider (cloud plane; needs the `anthropic` extra)
  config.py          ModelConfig — everything the factory, the host resolver and dispatch read
  trust.py           check_model_url, DEFAULT_TRUSTED_CIDRS, UntrustedModelHost
  hosts.py           HostRegistry, InMemoryHostRegistry, resolve, describe / fingerprint / report
  factory.py         build_router, CloudWrapper
  refresh.py         RouterHolder, resolved_router, describe_embed_host
  dispatch.py        plan_for, run — task-kind policy and memoization over the router
  shape.py           ask / choice / label / obj — getting a parseable answer, mechanism ladders
  usage.py           UsageRouter — per-run token totals
  metering.py        Meter, stage, MeteredRouter, UsageRow / UsageSink / InMemoryUsageSink, flush
  savers/            catalog, count (estimate), passages (whitespace, dedupe, apply_to_chunks)
tests/               pytest suite, no network, no database
```

## Install and test

```
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

Runtime dependency: `httpx`. Optional extra `anthropic` for `AnthropicProvider` (imported lazily).
Test extra: `pytest`, `pytest-asyncio` (`anyio`, used by some tests, comes with `httpx`).
Python 3.11 or newer.

## Plugging in a deployment

The package has no database, settings system or tracing backend of its own. A deployment supplies
them through small interfaces:

| Interface | What the deployment provides | In-package default |
|---|---|---|
| `config.ModelConfig` | model addresses, models, timeouts, perimeter, cloud options. Any object with the same attribute names works too (e.g. a pydantic settings class). | defaults of the original deployment |
| `hosts.HostRegistry` (`load()`, `stamp()`) | where the administrator's address override lives (a table row, a file). `load()` may return any object with the fields `ollama_base_url`, `embed_base_url`, `ollama_model_hosts`, `updated_at`, `updated_by`. Every write must move the stamp. | `InMemoryHostRegistry` |
| `metering.UsageSink` (`record(row)`, `commit()`) | persistence of token usage; `flush(sink, identity=...)` writes one `UsageRow` per (kind, model, plane) bucket and commits. | `InMemoryUsageSink` |
| `flush(..., identity=callable)` | the author of the spend when it is known only at flush time (e.g. the signed-in user); used only when the meter has no `profile_id`. | none |
| `tracing.set_span_factory(f)` | `f(name, **attrs)` returning a context manager, e.g. an OpenTelemetry facade. Process-wide. | no-op spans |
| `factory.build_router(..., wrap_cloud=f)` | `f(default_cloud_or_None)` returns the cloud provider to register — e.g. one that answers some callers with another key and falls back to the default. Not called under air-gap. | none |
| `RouterHolder(..., build=f)`, `resolved_router(..., build=f)` | the deployment's own router builder (e.g. one that passes `wrap_cloud`). | `factory.build_router` |
| `savers.apply_to_chunks(..., scope_attr=...)`, `savers.passages.dedupe(..., scope_attr=...)` | the attribute a passage carries its scope in. | `"scope_id"` |
| `savers.catalog.enabled(..., catalog=...)`, `describe(..., catalog=...)` | the same savers described in the deployment's own words. | `CATALOG` |
| `shape.LADDERS[name] = (...)` | measured mechanism ladders for the deployment's own tuned models. | base models only |
| `openai_compat.SCHEMA_NAME` | the schema name the model server logs (read per request). | `"llmkit"` |

## Tests

`pytest` runs 173 tests (168 functions), with no network and no database. Most are the source's own
tests with neutral test data; tests that needed the product database were re-expressed against the
in-memory registry and sink where the logic under test moved into this package. The product's
tests of its admin routes, health tiles, ops monitor, worker wiring, billing ledger and relay stayed
with the product.

## How llm-kit compares

llm-kit sits next to model gateways, learned routers and response caches. Each centres on a
different question.

| | centres on | how a request finds its model | reuse of results | sensitive data |
|---|---|---|---|---|
| **llm-kit** | where a request is allowed to run, and what it costs | by its sensitivity first: confidential work runs on hosts inside the trusted perimeter, checked by address; then by task kind, local models first | exact memoization of deterministic tasks, keyed by task kind, model, temperature, scope (tenant or project) and the exact messages | an invariant of the router: a confidential request reaches trusted-plane providers only |
| LiteLLM [7] | one OpenAI-format interface and proxy for many providers | load balancing, fallbacks, cost- and latency-based routing | exact and semantic response caching | guardrail integrations, such as PII masking with Presidio |
| Portkey AI Gateway [8] | a gateway for production traffic | fallbacks, load balancing, retries, conditional routing | simple and semantic caching | guardrails on requests and responses |
| RouteLLM [6] | cost at a target quality | a learned router picks a strong or a weak model for each query | — | — |
| FrugalGPT [5] | cost at a target quality | a cascade of models of rising cost, stopped by a scorer | a completion cache among its techniques | — |
| GPTCache [9] | reuse of answers to similar questions | — | a semantic cache over embeddings | — |

What llm-kit adds to that picture:

- The sensitivity of the data decides the plane, and the address of a trusted host is checked
  against a configured perimeter before a provider is built on it.
- Reuse is exact and scoped: a memoized result is served only for the same task kind, model,
  temperature, scope and messages, and confidential work is memoized only within a scope, so one
  tenant's result is never another's.
- Small local models get a shaping ladder (the model's own answer, then a prefilled reply, then
  decoding constrained by a JSON schema) that makes their answers parse, so utility steps can run
  on them while the cloud writes only what needs it.
- Metering is part of the call path: tokens per stage, model and plane, together with the tokens
  each saver kept out of the prompt. The saver catalog records where each technique comes from.

llm-kit can run beside a gateway: a deployment keeps its gateway for traffic management and puts
llm-kit where the decision about sensitive data is made.

## References

1. W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention.
   SOSP 2023. arXiv:2309.06180. (vLLM)
2. Ollama. https://github.com/ollama/ollama
3. llama.cpp. https://github.com/ggml-org/llama.cpp
4. Anthropic. Prompt caching. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
   (the `prompt_cache` saver)
5. L. Chen, M. Zaharia, J. Zou. FrugalGPT: How to Use Large Language Models While Reducing Cost and
   Improving Performance. 2023. arXiv:2305.05176.
6. I. Ong et al. RouteLLM: Learning to Route LLMs with Preference Data. 2024. arXiv:2406.18665.
7. LiteLLM. https://github.com/BerriAI/litellm
8. Portkey AI Gateway. https://github.com/Portkey-AI/gateway
9. GPTCache. https://github.com/zilliztech/GPTCache
10. NVIDIA, NTU, MIT. SoL-Pi: Scaling Auto-Research Loops for Efficient Agent Harnesses. 2026.
    https://github.com/NVlabs/SoL-Pi (the `evidence_reducer`, `observation_pack`, `action_fusion`
    and `context_compact` savers)
11. Spotify Portal. Introducing Xirp. https://portal.spotify.com/blog/introducing-xirp (model
    switching by price and performance; the `shape_routing` saver)

## Author

llm-kit is written by Dr. Alexandra Bernadotte and developed under the AICumene name.
Copyright 2026 Alexandra Bernadotte (see `NOTICE`). To cite it, use `CITATION.cff`
(GitHub shows it as "Cite this repository").

## License

Apache License 2.0; see `LICENSE` and `NOTICE`.
