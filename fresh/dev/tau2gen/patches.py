"""Подмены в τ²-bench, чтобы новые диалоги были устроены как данные организаторов (видно по valid).

Что совпадает с организаторами:
  • к инструкции агента дописано «Always communicate with the user in Russian. Keep tool names, …»;
  • клиент пишет по-русски, приветствие агента — «Здравствуйте! Чем могу помочь?»;
  • политика домена — дословно из valid (банковскую организаторы собирали старой версией τ²: grep вместо shell);
  • в банке у агента KB_search_bm25, KB_search_dense и grep; документы базы знаний переведены на русский
    (это делает translate_kb.py до запуска).
Отличие: KB_search_dense у нас тоже работает на BM25 — внешних эмбеддингов нет; имя и схема инструмента для
агента те же.

Модуль импортирует tau2 только внутри apply_patches(): локально (без tau2) его можно читать и тестировать.
"""
from __future__ import annotations

import threading

RUSSIAN_AGENT_RULE = ("Always communicate with the user in Russian. Keep tool names, tool call arguments, IDs, codes, "
                      "and other data values exactly as they appear in the system — do not translate or transliterate "
                      "them.")
RUSSIAN_USER_RULES = """

## Language
- Always write to the agent in Russian, the way a native Russian speaker would.
- Keep names, IDs, e-mails, phone numbers, codes, product names and other data values exactly as given in your \
instructions — do not translate or transliterate them.
- Special tokens ('###STOP###', '###TRANSFER###', '###OUT-OF-SCOPE###') stay exactly as they are."""
RUSSIAN_GREETING = "Здравствуйте! Чем могу помочь?"
RETRIEVAL_VARIANT = "veritas_alltools"

_agent_setups: dict = {}               # task_id -> {"system_prompt": ..., "tools": [...]} — что видел агент
_lock = threading.Lock()


def agent_setups() -> dict:
    with _lock:
        return dict(_agent_setups)


def _patch_agent_instruction() -> None:
    import tau2.agent.llm_agent as llm_agent
    if RUSSIAN_AGENT_RULE not in llm_agent.AGENT_INSTRUCTION:
        llm_agent.AGENT_INSTRUCTION = llm_agent.AGENT_INSTRUCTION + "\n\n" + RUSSIAN_AGENT_RULE


def _patch_user_language() -> None:
    import tau2.user.user_simulator as user_simulator
    original = user_simulator.get_global_user_sim_guidelines
    if getattr(original, "russian", False):
        return

    def russian_guidelines(use_tools: bool = False) -> str:
        return original(use_tools=use_tools) + RUSSIAN_USER_RULES

    russian_guidelines.russian = True
    user_simulator.get_global_user_sim_guidelines = russian_guidelines


def _patch_greeting() -> None:
    import tau2.orchestrator.orchestrator as orchestrator
    orchestrator.DEFAULT_FIRST_AGENT_MESSAGE.content = RUSSIAN_GREETING


def _register_banking_retrieval() -> None:
    """Набор инструментов организаторов: KB_search_bm25 + KB_search_dense + grep.

    В текущей версии τ² «все инструменты» — это bm25 + dense + shell, а набора с grep нет: фабрика
    build_tools превратила бы наш вариант в один grep. Поэтому свой класс набора и своя ветка фабрики.
    """
    from tau2.domains.banking_knowledge import environment, retrieval
    from tau2.domains.banking_knowledge.retrieval_mixins import (GrepMixin, KBSearchBm25AllToolsMixin,
                                                                 KBSearchDenseAllToolsMixin)
    from tau2.domains.banking_knowledge.tools import KnowledgeTools

    class KnowledgeToolsBm25DenseGrep(KBSearchBm25AllToolsMixin, KBSearchDenseAllToolsMixin, GrepMixin,
                                      KnowledgeTools):
        def __init__(self, db, kb_bm25_pipeline, kb_dense_pipeline, grep_pipeline):
            super().__init__(db)
            self._kb_bm25_pipeline = kb_bm25_pipeline
            self._kb_dense_pipeline = kb_dense_pipeline
            self._grep_pipeline = grep_pipeline

    retrieval.RETRIEVAL_VARIANTS[RETRIEVAL_VARIANT] = retrieval.RetrievalVariant(
        name=RETRIEVAL_VARIANT,
        prompt_template=retrieval.PROMPTS_DIR / "all_tools.md",     # текст политики всё равно заменяется на valid
        build_prompt=retrieval.standard_prompt,
        kb_search_bm25=retrieval.PipelineSpec(type="bm25"),
        kb_search_dense=retrieval.PipelineSpec(type="bm25"),
        grep=retrieval.GrepSpec(),
    )
    original_build_tools = environment.build_tools
    if getattr(original_build_tools, "organizers_set", False):
        return

    def build_tools(variant, db, knowledge_base, read_log_allowlist=None):
        if variant.name != RETRIEVAL_VARIANT:
            return original_build_tools(variant, db, knowledge_base, read_log_allowlist=read_log_allowlist)
        tools = KnowledgeToolsBm25DenseGrep(db,
                                            retrieval._create_kb_pipeline(variant.kb_search_bm25, knowledge_base),
                                            retrieval._create_kb_pipeline(variant.kb_search_dense, knowledge_base),
                                            retrieval._create_grep_pipeline(variant.grep, knowledge_base))
        tools.set_read_log_allowlist(read_log_allowlist)
        return tools

    build_tools.organizers_set = True
    environment.build_tools = build_tools      # environment импортировал имя к себе — подменяем там


def check_banking_tools() -> list:
    """Имена инструментов агента в банке с нашим вариантом поиска (для проверки в ноутбуке)."""
    from tau2.domains.banking_knowledge.environment import get_environment
    environment = get_environment(retrieval_variant=RETRIEVAL_VARIANT)
    return sorted(tool.name for tool in environment.get_tools())


def _record_agents(policy_overrides: dict) -> None:
    """Подменяет политику агента на текст организаторов и запоминает, что агент видел (промпт и инструменты)."""
    import tau2.runner.build as build
    original = build.build_agent
    if getattr(original, "recording", False):
        return

    def recording_build_agent(agent_name, environment, *args, **kwargs):
        agent = original(agent_name, environment, *args, **kwargs)
        domain = getattr(environment, "domain_name", None)
        if domain in policy_overrides and hasattr(agent, "domain_policy"):
            agent.domain_policy = policy_overrides[domain]
        task = kwargs.get("task")
        if task is not None and hasattr(agent, "system_prompt"):
            setup = {"domain": domain, "system_prompt": agent.system_prompt,
                     "tools": [tool.openai_schema for tool in agent.tools]}
            with _lock:
                _agent_setups[str(task.id)] = setup
        return agent

    recording_build_agent.recording = True
    build.build_agent = recording_build_agent


def apply_patches(policy_overrides: dict) -> None:
    """policy_overrides: домен -> текст политики организаторов (между <policy> и </policy> в valid)."""
    _patch_agent_instruction()
    _patch_user_language()
    _patch_greeting()
    _register_banking_retrieval()
    _record_agents(policy_overrides)
