.PHONY: up down build ingest ingest-dry ask evaluate evaluate-ragas logs pull-llm ps clean chm reindex reindex-status reindex-pause reindex-resume prompts synonyms synonyms-expand seed-synonyms super-test

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

chm:
	docker compose run --rm app python chm_extract.py

ingest-dry: chm
	docker compose run --rm app python ingest.py --dry-run

ingest: chm
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

super-test:
	docker compose exec -T app python eval/super_test.py $(ARGS)

clean:
	docker compose down -v
	rm -rf app/.cache qdrant/storage data/extracted