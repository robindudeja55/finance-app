
001 — Commands log: infra bootstrap (MySQL + Redis)
Date: <2025 - august - 08>


Machine: macOS + VS Code
Repo setup (already done)


cd ~/Projects
mkdir -p finance-app
cd finance-app
git init -b main
echo "# Personal Finance Insights" > README.md
touch .gitignore
git add .
git commit -m "chore: init repo with README and .gitignore"

If not set earlier:
git remote add origin https://github.com/<robindudeja55>/finance-app.git
git push -u origin main
Project structure (folders only)
mkdir -p backend docs scripts

Create env file for Docker (do NOT commit .env)


.env (repo root)
MYSQL_DATABASE=financedb
MYSQL_USER=robin
MYSQL_PASSWORD=robinroby10
MYSQL_ROOT_PASSWORD=robinroby10

docker compose config

Start infra


docker compose up -d
docker compose ps
docker compose logs db --tail=50
docker compose logs redis --tail=50

Redis check
docker exec -it finance_redis redis-cli ping # -> PONG

MySQL checks
First attempt (kept for history; fails due to quoting/env expansion)
docker exec -it finance_db mysql -u"${MYSQL_USER}" -p"${MYSQL_PASSWORD}" -e "SHOW DATABASES;"

Recommended checks (safer; expand vars inside the container)
docker compose exec db sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SELECT VERSION();"'
docker compose exec db sh -lc 'mysql -urobin -p"$MYSQL_PASSWORD" -e "SHOW DATABASES;"'

If user plugin ever needs reset (MySQL 8.4 uses caching_sha2_password by default)
docker compose exec db sh -lc 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "ALTER USER '''robin'''@'''%''' IDENTIFIED BY '''$MYSQL_PASSWORD'''; FLUSH PRIVILEGES;"'
Reset DB completely (DESTROYS DATA) if you need a clean re-init
docker compose down -v && docker compose up -d
Commit infra/config docs to Git (but NOT .env)
Create a shareable example env (edit secrets before committing)
cp .env .env.example

