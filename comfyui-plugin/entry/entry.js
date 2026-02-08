/**
 * ComfyUI Agent — Entry point (auto-loaded by ComfyUI via WEB_DIRECTORY).
 *
 * This script dynamically imports the compiled React chat panel and
 * registers it as a sidebar tab in ComfyUI.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "comfyui-agent";

/**
 * Convert ComfyUI API format to LiteGraph graph format.
 * API: {node_id: {class_type, inputs}} → Graph: {nodes: [...], links: [...]}
 */
function convertApiToGraph(apiWorkflow) {
  const nodes = [];
  const links = [];
  let linkId = 1;
  const nodeIds = Object.keys(apiWorkflow);

  // Create nodes with basic grid layout
  for (let i = 0; i < nodeIds.length; i++) {
    const nodeId = nodeIds[i];
    const nodeData = apiWorkflow[nodeId];
    const col = i % 3;
    const row = Math.floor(i / 3);

    // Separate widget values (primitives) from connections (arrays)
    const widgetValues = [];
    for (const [, value] of Object.entries(nodeData.inputs || {})) {
      if (!Array.isArray(value)) {
        widgetValues.push(value);
      }
    }

    nodes.push({
      id: parseInt(nodeId),
      type: nodeData.class_type,
      pos: [100 + col * 350, 100 + row * 300],
      size: [300, 150],
      flags: {},
      order: i,
      mode: 0,
      properties: { "Node name for S&R": nodeData.class_type },
      widgets_values: widgetValues,
    });
  }

  // Create links from input connections
  for (const nodeId of nodeIds) {
    const nodeData = apiWorkflow[nodeId];
    let inputSlot = 0;
    for (const [, value] of Object.entries(nodeData.inputs || {})) {
      if (Array.isArray(value) && value.length === 2) {
        const [sourceId, outputSlot] = value;
        links.push([
          linkId,
          parseInt(sourceId),
          outputSlot,
          parseInt(nodeId),
          inputSlot,
          "*",
        ]);
        linkId++;
      }
      inputSlot++;
    }
  }

  return {
    last_node_id: Math.max(...nodeIds.map(Number)),
    last_link_id: linkId - 1,
    nodes,
    links,
    groups: [],
    config: {},
    extra: { ds: { scale: 1, offset: [0, 0] } },
    version: 0.4,
  };
}

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

    // Listen for workflow load events from the React chat panel
    window.addEventListener("comfyui-agent:load-workflow", async (e) => {
      const { workflow } = e.detail || {};
      if (!workflow) return;

      try {
        // Use ComfyUI's built-in API format loader
        await app.loadGraphData(
          await convertApiToGraph(workflow),
          true,  // clean — clear existing graph
          true,  // restore_view
          workflow,  // pass original API workflow
        );
        console.log("[comfyui-agent] Workflow loaded to canvas");
      } catch (err) {
        console.warn("[comfyui-agent] loadGraphData failed, trying file fallback:", err);
        try {
          // Fallback: load as JSON file
          const blob = new Blob([JSON.stringify(workflow)], { type: "application/json" });
          const file = new File([blob], "agent_workflow.json", { type: "application/json" });
          await app.handleFile(file);
          console.log("[comfyui-agent] Workflow loaded via handleFile");
        } catch (err2) {
          console.error("[comfyui-agent] Failed to load workflow:", err2);
        }
      }
    });

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
