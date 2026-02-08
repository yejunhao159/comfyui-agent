/**
 * ComfyUI Agent — Entry point (auto-loaded by ComfyUI via WEB_DIRECTORY).
 *
 * This script dynamically imports the compiled React chat panel and
 * registers it as a sidebar tab in ComfyUI.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "comfyui-agent";

app.registerExtension({
  name: EXTENSION_NAME,

  async setup() {
    // Load the compiled React app
    const base = api.api_base || "";

    // Inject scoped CSS
    try {
      const cssLink = document.createElement("link");
      cssLink.rel = "stylesheet";
      cssLink.href = `${base}/agent_web/style.css`;
      document.head.appendChild(cssLink);
    } catch (e) {
      console.warn("[comfyui-agent] Failed to load CSS:", e);
    }

    // Register sidebar tab
    app.extensionManager.registerSidebarTab({
      id: "comfyui-agent-chat",
      icon: "pi pi-comments",
      title: "Agent",
      tooltip: "ComfyUI Agent Chat",
      type: "custom",
      render: (el) => {
        el.style.height = "100%";
        el.style.overflow = "hidden";

        const container = document.createElement("div");
        container.id = "comfyui-agent-root";
        container.style.cssText = "width:100%;height:100%;display:flex;flex-direction:column;";
        el.appendChild(container);

        // Dynamically import the React app entry
        import(`${base}/agent_web/main.js`)
          .then((mod) => {
            if (mod.mount) {
              mod.mount(container);
            }
          })
          .catch((err) => {
            console.error("[comfyui-agent] Failed to load chat panel:", err);
            container.innerHTML = `
              <div style="padding:20px;color:#ff6b6b;font-size:14px;">
                <p><strong>ComfyUI Agent</strong></p>
                <p>Failed to load chat panel.</p>
                <p style="font-size:12px;color:#888;margin-top:8px;">
                  Make sure to build the UI first:<br>
                  <code>cd comfyui-plugin/ui && npm run build</code>
                </p>
                <p style="font-size:11px;color:#666;margin-top:8px;">${err.message}</p>
              </div>`;
          });
      },
    });
  },

  // Add right-click menu to nodes
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    const origMenu = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
      if (origMenu) origMenu.apply(this, arguments);
      options.push({
        content: "Ask Agent about this node",
        callback: () => {
          window.dispatchEvent(
            new CustomEvent("comfyui-agent:ask-node", {
              detail: {
                nodeType: nodeData.name,
                nodeTitle: nodeData.display_name || nodeData.name,
              },
            })
          );
        },
      });
    };
  },
});
