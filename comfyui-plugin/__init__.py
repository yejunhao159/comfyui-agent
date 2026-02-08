"""ComfyUI Agent Plugin — Sidebar chat panel for the ComfyUI Agent.

This custom node adds no new nodes to ComfyUI. It only provides:
1. A sidebar tab with a chat panel (via WEB_DIRECTORY)
2. Static file serving for the compiled React frontend
"""

import os

from aiohttp import web

WEB_DIRECTORY = "./entry"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Serve compiled React app as static files
dist_path = os.path.join(os.path.dirname(__file__), "ui", "dist")
if os.path.exists(dist_path):
    import server

    server.PromptServer.instance.app.router.add_static(
        "/agent_web/", dist_path, name="agent_web"
    )
