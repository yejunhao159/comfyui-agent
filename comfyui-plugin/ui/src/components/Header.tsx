import React from "react";
import type { ConnectionStatus } from "../types";

interface Props {
  status: ConnectionStatus;
  onClear: () => void;
}

export const Header: React.FC<Props> = ({ status, onClear }) => {
  const statusConfig: Record<
    ConnectionStatus,
    { color: string; label: string }
  > = {
    connecting: { color: "#ffd740", label: "连接中..." },
    connected: { color: "#00e676", label: "已连接" },
    disconnected: { color: "#ff5252", label: "已断开" },
  };

  const { color, label } = statusConfig[status];

  return (
    <div className="cua-header">
      <div className="cua-header-left">
        <span className="cua-header-title">ComfyUI Agent</span>
        <span className="cua-header-status">
          <span className="cua-dot" style={{ background: color }} />
          {label}
        </span>
      </div>
      <button className="cua-header-clear" onClick={onClear} title="清空对话">
        🗑
      </button>
    </div>
  );
};
