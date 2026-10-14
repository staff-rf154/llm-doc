"""Стартовые промты: каталог составлен по анализу сайта example.org
(разделы «Продукты», «Решения», «Бизнес-задачи», «Контроль сотрудников»)
и привязан к документации docs.example.org, лежащей в индексе.
"""

import time
from pathlib import Path

import config
import promptdb
import settings
import vectorstore as vs

SITE = "https://example.org/"

SEED = [
    ("Как настроить ICAP-сервис в Search Server?", "СёрчИнформ КИБ", ["icap_settings", "search_server"], "https://docs.example.org/ICAP_settings.pdf"),
    ("Какие задачи решает перехват информации в DLP?", "СёрчИнформ КИБ", ["dlp-capabilities", "user_manual"], "https://docs.example.org/dlp-capabilities-windows-company.pdf"),
    ("Как подготовить систему к установке агента KIB?", "СёрчИнформ КИБ", ["preparation_for_installation", "preparation_for_installation-linux"], "https://docs.example.org/preparation_for_installation.pdf"),
    ("Как развернуть СёрчИнформ КИБ на Linux?", "СёрчИнформ КИБ", ["kib_company_installation_on_linux", "siem_installation_on_linux"], "https://docs.example.org/KIB_company_installation_on_Linux.pdf"),
    ("Как установить клиентские части агента KIB?", "СёрчИнформ КИБ", ["client_parts_installation"], "https://docs.example.org/Client_parts_installation.pdf"),
    ("Как настроить перехват электронной почты?", "СёрчИнформ КИБ", ["kib_company_mail_int_installation"], "https://docs.example.org/KIB_company_Mail_Int_installation.pdf"),
    ("Как настроить интеграцию с внешним API?", "СёрчИнформ КИБ", ["installation_external_api"], "https://docs.example.org/installation_external_api.pdf"),
    ("Как обновить СёрчИнформ KIB до новой версии?", "СёрчИнформ КИБ", ["kib_company_update_basic", "kib_company_update_on_linux_basic"], "https://docs.example.org/KIB_company_update_basic.pdf"),
    ("Как выполнить миграцию на СёрчИнформ KIB?", "СёрчИнформ KIB", ["kib_migration", "siem_migration"], "https://docs.example.org/KIB_migration.pdf"),
    ("Как установить ИИ-Ассистент в СёрчИнформ КИБ?", "ИИ-Ассистент", ["ai-assistant-installation"], "https://docs.example.org/ai-assistant-installation-windows-company.pdf"),
    ("Как ИИ-Ассистент отвечает на вопросы по документации?", "ИИ-Ассистент", ["ai_si_long", "ai_si_short"], "https://docs.example.org/ai_si_long.pdf"),
    ("Как настроить аудит данных в Файловом аудиторе?", "Файловый аудитор", ["fileauditor_description"], "https://docs.example.org/fileauditor_description.pdf"),
    ("Как FileAuditor сканирует файлы на агентах?", "Файловый аудитор", ["fileauditor_agent_scanning"], "https://docs.example.org/fileauditor_agent_scanning.pdf"),
    ("Как настроить метки и классификацию файлов?", "Файловый аудитор", ["fileauditor_labels_guide"], "https://docs.example.org/FileAuditor_Labels_Guide.pdf"),
    ("Как установить FileAuditor на Windows и Linux?", "Файловый аудитор", ["fileauditor_installation", "fileauditor_installation_linux"], "https://docs.example.org/FileAuditor_installation.pdf"),
    ("Как подготовить систему к установке FileAuditor?", "Файловый аудитор", ["preparation_for_installation_fa", "preparation_for_installation_fa_linux"], "https://docs.example.org/preparation_for_installation_FA.pdf"),
    ("Какие возможности у СёрчИнформ SIEM?", "SIEM", ["siem_capabilities"], "https://docs.example.org/SIEM_capabilities.pdf"),
    ("Как устроен сбор и корреляция событий в SIEM?", "SIEM", ["siem_help", "siem_description"], "https://docs.example.org/SIEM_help.pdf"),
    ("Как создать правило корреляции событий в SIEM?", "SIEM", ["cross-correlation"], "https://docs.example.org/Cross-Correlation_Rules_Setup.pdf"),
    ("Как настроить автоматическое реагирование на инциденты?", "SIEM", ["automatic_incident_response"], "https://docs.example.org/Automatic_incident_response.pdf"),
    ("Как настроить интеграцию SIEM с ГосСОПКА?", "SIEM", ["configure_integration_siem_with_gossopka"], "https://docs.example.org/Configure_integration_SIEM_with_GosSOPKA.pdf"),
    ("Как установить SIEM на Linux?", "SIEM", ["installation_siem", "siem_installation_on_linux"], "https://docs.example.org/installation_SIEM.pdf"),
    ("Как настроить мониторинг безопасности и оповещения?", "SIEM", ["monitoring_settings", "alertcenter_help"], "https://docs.example.org/Monitoring_settings.pdf"),
    ("Как настроить карантин файлов при инциденте?", "SIEM", ["quarantine", "monitoring_settings"], "https://docs.example.org/Quarantine_manual.pdf"),
    ("Как проверить правила коннекторов (инциденты SIEM)?", "SIEM", ["проверка_правил"], "https://docs.example.org/"),
    ("Какие психотипы личности выделяет Центр профилирования?", "ProfileCenter", ["profilecenter_description"], "https://docs.example.org/ProfileCenter_description.pdf"),
    ("Как профилирование помогает выявлять инсайдера?", "ProfileCenter", ["profilecenter_user_manual", "dlp-capabilities"], SITE),
    ("Как контролировать рабочее время сотрудников и законно ли это?", "Контроль времени", [], SITE),
    ("Как оценить эффективность сотрудников средствами company?", "Контроль сотрудников", [], SITE),
    ("Что показывает веб-аналитика company?", "Веб-аналитика", ["webanalytic_manual"], "https://docs.example.org/WebAnalytic_manual.pdf"),
    ("Как восстановить работоспособность модулей после сбоя?", "Эксплуатация", ["restoring_functionality"], "https://docs.example.org/restoring_functionality.pdf"),
    ("Как проверить работоспособность системы и собрать диагностику?", "Эксплуатация", ["bugs_diagnostics"], "https://docs.example.org/Bugs_diagnostics.pdf"),
    ("Как обновить MongoDB в составе company?", "Эксплуатация", ["mongodb_update"], "https://docs.example.org/MongoDB_update.pdf"),
    ("Как настроить учётные записи и права доступа?", "Эксплуатация", ["accounts_and_access_rights"], "https://docs.example.org/accounts_and_access_rights.pdf"),
    ("Как активировать продукт и управлять лицензиями?", "Лицензирование", ["company_activation"], "https://docs.example.org/company_activation.pdf"),
    ("Как устроено лицензирование продуктов company?", "Лицензирование", ["products_licensing"], "https://docs.example.org/company_Products_Licensing.pdf"),
    ("Как настроить EndpointController на клиентах?", "Search Server", ["endpointcontroller_help"], "https://docs.example.org/EndpointController_help.pdf"),
    ("Как работает Search Server и как подключить источники данных?", "Search Server", ["search_server_ru"], "https://docs.example.org/Search_Server_RU.pdf"),
    ("Как использовать AAServer для автоматизации задач?", "Search Server", ["aaserver_help"], "https://docs.example.org/AAServer_Help.pdf"),
    ("Как настроить ESHub?", "Search Server", ["eshub_help"], "https://docs.example.org/ESHub_help.pdf"),
    ("Как настроить DataCenter?", "Search Server", ["datacenter_help"], "https://docs.example.org/DataCenter_help.pdf"),
]

_cache: dict = {"at": 0.0, "items": []}
_sources_cache: dict = {"at": 0.0, "sources": []}


def seed(force: bool = False) -> int:
    items = [
        {
            "text": text,
            "product": product,
            "keywords": keywords,
            "source_url": source_url,
        }
        for text, product, keywords, source_url in SEED
    ]
    return promptdb.upsert_prompts(items, replace=force)


def seed_if_empty() -> int:
    if promptdb.count_prompts() == 0 or config.PROMPTS_SEED_ON_START:
        return seed()
    return promptdb.count_prompts()


def indexed_sources() -> list[str]:
    now = time.time()
    if now - _sources_cache["at"] < config.PROMPTS_CACHE_TTL:
        return _sources_cache["sources"]
    client = vs.get_qdrant()
    target = vs.target_name()
    sources: list[str] = []
    seen: set[str] = set()
    if client.collection_exists(target):
        offset = None
        while len(sources) < 600:
            points, offset = client.scroll(
                collection_name=target,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                source = (point.payload or {}).get("source")
                if source and source not in seen:
                    seen.add(source)
                    sources.append(source)
            if offset is None:
                break
    _sources_cache.update(at=now, sources=sources)
    return sources


def build_prompts() -> list[dict]:
    rows = promptdb.list_prompts()
    if not rows:
        return []
    haystack = " ".join(s.lower() for s in indexed_sources())
    matched = []
    unconditional = []
    for row in rows:
        keywords = row["keywords"].split()
        if not keywords:
            unconditional.append(row)
        elif any(k in haystack for k in keywords):
            matched.append(row)
    limit = settings.int_value("PROMPTS_MAX")
    selected = (matched + unconditional or rows)[:limit]
    return [
        {"text": r["text"], "product": r["product"], "source_url": r["source_url"]}
        for r in selected[:limit]
    ]


def get_prompts() -> list[dict]:
    seed_if_empty()
    now = time.time()
    if _cache["items"] and now - _cache["at"] < config.PROMPTS_CACHE_TTL:
        return _cache["items"]
    items = build_prompts()
    _cache.update(at=now, items=items)
    return items


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Наполнение БД стартовых промтов")
    parser.add_argument("--replace", action="store_true", help="полностью пересоздать каталог")
    args = parser.parse_args()
    Path(config.PROMPTS_DB).parent.mkdir(parents=True, exist_ok=True)
    total = seed(force=args.replace)
    print(f"промтов в БД: {total}")


if __name__ == "__main__":
    main()