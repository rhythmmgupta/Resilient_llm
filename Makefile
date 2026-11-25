.PHONY: up down logs ps test dev-backend dev-frontend

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

test:
	cd backend && .venv/bin/pytest -v

dev-backend:
	cd backend && DATABASE_URL="sqlite:///./data/app.db" CACHE_DIR="./data/litellm_cache" .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend:
	cd frontend && npm run dev
