#!/usr/bin/env bash
set -euo pipefail

# Install Ollama and the Gemma 2B model on Linux, then verify.

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required. Please install curl and re-run." >&2
  exit 1
fi

echo "Installing Ollama..."
if [ "${EUID}" -ne 0 ]; then
  echo "Using official installer (will prompt for sudo if needed)."
fi

curl -fsSL https://ollama.com/install.sh | sh

echo "Ensuring Ollama service is running..."
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl enable --now ollama || true
  sudo systemctl start ollama || true
fi

echo "Waiting for Ollama to be ready on :11434..."
for i in {1..20}; do
  if curl -sSf http://localhost:11434/api/version >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Pulling Gemma 2B model..."
if command -v ollama >/dev/null 2>&1; then
  ollama pull gemma2:2b || ollama run gemma2:2b || true
else
  echo "ollama command not found in PATH. Please open a new shell and re-run: ollama pull gemma2:2b" >&2
fi

echo "Verifying model availability..."
if curl -sSf http://localhost:11434/api/tags | grep -q 'gemma2:2b'; then
  echo "Gemma 2B is installed and available."
else
  echo "Warning: Gemma 2B not found in tags. Try: 'ollama run gemma2:2b' manually." >&2
fi

echo "Done."
