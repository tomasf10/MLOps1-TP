#!/usr/bin/env bash
set -e
set -u

function create_database() {
  local database=$1
  echo "Creating database '${database}'"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
    SELECT 'CREATE DATABASE ${database}'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${database}')\gexec
EOSQL
}

if [[ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]]; then
  IFS=',' read -ra databases <<< "$POSTGRES_MULTIPLE_DATABASES"
  for database in "${databases[@]}"; do
    create_database "$database"
  done
fi
