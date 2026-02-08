import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles/index.css";

/**
 * Mount the chat panel into a container element.
 * Called by entry.js when the sidebar tab is activated.
 */
export function mount(container: HTMLElement) {
  const root = ReactDOM.createRoot(container);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
