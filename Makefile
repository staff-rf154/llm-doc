.PHONY: up down build ingest ingest-dry ask evaluate evaluate-ragas evaluate-regression logs pull-llm ps clean reindex reindex-status reindex-pause reindex-resume prompts synonyms synonyms-expand seed-synonyms super-test backup restore backup-status

up:
	docker compose up -d --build

down:
	docker compose down

build:
	docker compose build app

ps:
	docker compose ps

logs:
	docker compose logs -f app

pull-llm:
	docker compose --profile local-llm up -d ollama
	docker compose exec ollama ollama pull $(LLM_MODEL)


ingest-dry: 
	docker compose run --rm app python ingest.py --dry-run

ingest: 
	docker compose run --rm app python ingest.py

reindex:
	curl -s -X POST localhost:$${APP_PORT:-8091}/reindex

reindex-status:
	curl -s localhost:$${APP_PORT:-8091}/reindex

reindex-pause:
	curl -s -X POST localhost:$${APP_PORT:-8091}/reindex/pause

reindex-resume:
	curl -s -X POST localhost:$${APP_PORT:-8091}/reindex/resume

seed-prompts:
	docker compose run --rm app python prompts.py

prompts:
	curl -s localhost:$${APP_PORT:-8091}/prompts

synonyms:
	docker compose exec -T app python -c "import synonymdb;[print(i['term'],'->',i['synonyms']) for i in synonymdb.list_terms()]"

synonyms-expand:
	docker compose exec -T app python -c "import sys,synonymdb;[print('-',v) for v in synonymdb.expand(' '.join(sys.argv[1:]))]" $(Q)

seed-synonyms:
	docker compose exec -T app python synonyms.py --replace

ask:
	docker compose run --rm app python -c "import sys,rag; print(rag.ask(' '.join(sys.argv[1:])).text)" $(Q)

evaluate:
	docker compose run --rm app python eval/eval.py

evaluate-ragas:
	docker compose run --rm app python eval/eval.py --ragas

evaluate-regression:
	docker compose exec -T app python eval/eval.py
	docker compose exec -T app python eval/regression.py

backup:
	./scripts/backup.sh

restore:
	@test -n "$(BACKUP)" || (echo "Укажите BACKUP=data/backups/<копия>" >&2; exit 2)
	RESTORE=yes ./scripts/restore.sh $(BACKUP) $(if $(DOCS),--docs,)

backup-status:
	docker compose exec -T app python backup.py

super-test:
	docker compose exec -T app python eval/super_test.py $(ARGS)

clean:
	docker compose down -v
	rm -rf app/.cache qdrant/storage