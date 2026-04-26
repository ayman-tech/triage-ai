#!/usr/bin/env bash
# Run the FastAPI app over HTTPS for local development (trusted cert via mkcert).
#
# One-time setup (from repo root):
#   brew install mkcert
#   mkcert -install
#   mkdir -p .certs && cd .certs && mkcert localhost 127.0.0.1 ::1 && cd -
#
# Then open: https://127.0.0.1:8001  (or the port you set)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
CERT_DIR="$ROOT/.certs"

if [[ ! -d "$CERT_DIR" ]]; then
  echo "Missing $CERT_DIR"
  echo "Create certificates first:"
  echo "  brew install mkcert && mkcert -install"
  echo "  mkdir -p .certs && cd .certs && mkcert localhost 127.0.0.1 ::1"
  exit 1
fi

# mkcert produces e.g. localhost+2.pem and localhost+2-key.pem
CERT="$(find "$CERT_DIR" -maxdepth 1 -name '*.pem' ! -name '*-key.pem' 2>/dev/null | head -1)"

if [[ -z "${CERT}" || ! -f "$CERT" ]]; then
  echo "No certificate (*.pem except *-key.pem) found in $CERT_DIR"
  echo "Run: cd .certs && mkcert localhost 127.0.0.1 ::1"
  exit 1
fi

BASE="${CERT%.pem}"
KEY="${BASE}-key.pem"
if [[ ! -f "$KEY" ]]; then
  echo "Missing private key: $KEY"
  exit 1
fi

PY="$ROOT/.venv/bin/python3"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

echo "Starting https://127.0.0.1:${PORT} (bind ${HOST}:${PORT})"
exec "$PY" -m uvicorn main:app --reload --host "$HOST" --port "$PORT" \
  --ssl-keyfile "$KEY" \
  --ssl-certfile "$CERT"
