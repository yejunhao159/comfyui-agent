#!/usr/bin/env bash
# ComfyUI Agent — Bare-metal installation script
# Usage: bash deploy/install.sh /path/to/comfyui
set -euo pipefail

COMFYUI_DIR="${1:-}"
AGENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "$COMFYUI_DIR" ]; then
    echo "Usage: bash deploy/install.sh /path/to/ComfyUI"
    echo ""
    echo "This script will:"
    echo "  1. Create a Python venv and install the agent"
    echo "  2. Symlink the plugin into ComfyUI custom_nodes"
    echo "  3. Build the plugin UI"
    echo "  4. Create a default config"
    exit 1
fi

echo "=== ComfyUI Agent Installation ==="
echo "Agent dir:  $AGENT_DIR"
echo "ComfyUI dir: $COMFYUI_DIR"
echo ""

# 1. Python venv
echo "[1/4] Setting up Python environment..."
if [ ! -d "$AGENT_DIR/.venv" ]; then
    python3.12 -m venv "$AGENT_DIR/.venv"
fi
"$AGENT_DIR/.venv/bin/pip" install -q --upgrade pip
"$AGENT_DIR/.venv/bin/pip" install -q -e "$AGENT_DIR"
echo "  ✓ Python packages installed"

# 2. Symlink plugin
echo "[2/4] Linking plugin to ComfyUI..."
CUSTOM_NODES="$COMFYUI_DIR/custom_nodes"
LINK_TARGET="$CUSTOM_NODES/comfyui-agent"
if [ -L "$LINK_TARGET" ]; then
    rm "$LINK_TARGET"
fi
ln -s "$AGENT_DIR/comfyui-plugin" "$LINK_TARGET"
echo "  ✓ Plugin linked: $LINK_TARGET → $AGENT_DIR/comfyui-plugin"

# 3. Build UI
echo "[3/4] Building plugin UI..."
if command -v npm &> /dev/null; then
    (cd "$AGENT_DIR/comfyui-plugin/ui" && npm ci --ignore-scripts && npm run build)
    echo "  ✓ UI built"
else
    echo "  ⚠ npm not found — skip UI build. Install Node.js and run:"
    echo "    cd $AGENT_DIR/comfyui-plugin/ui && npm ci && npm run build"
fi

# 4. Config
echo "[4/4] Checking configuration..."
if [ ! -f "$AGENT_DIR/config.yaml" ]; then
    COMFYUI_PORT="${COMFYUI_PORT:-6006}"
    cat > "$AGENT_DIR/config.yaml" << EOF
comfyui:
  base_url: "http://127.0.0.1:${COMFYUI_PORT}"
  ws_url: "ws://127.0.0.1:${COMFYUI_PORT}/ws"
  timeout: 30

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-5-20250929"
  max_tokens: 8192
  temperature: 0.7

agent:
  max_iterations: 20
  session_db: "data/sessions.db"

server:
  host: "0.0.0.0"
  port: 5200
  cors_origins: ["*"]

logging:
  level: "INFO"
  format: "console"
  log_dir: "data/logs"

identity:
  rolex_dir: "~/.rolex"
  role_name: ""
EOF
    echo "  ✓ Default config created"
else
    echo "  ✓ Config exists"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Set your API key:"
echo "     export ANTHROPIC_API_KEY=sk-ant-xxxxx"
echo ""
echo "  2. Start the agent:"
echo "     cd $AGENT_DIR"
echo "     .venv/bin/python -m comfyui_agent"
echo ""
echo "  3. Start/restart ComfyUI (it will load the plugin automatically)"
echo ""
echo "  4. Open ComfyUI in browser → click 'Agent' tab in sidebar"
