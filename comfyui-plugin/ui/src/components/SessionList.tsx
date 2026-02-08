import React from "react";
import type { Session } from "../types";

interface Props {
  sessions: Session[];
  currentSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onNew: () => void;
  onDelete: (sessionId: string) => void;
  onClose: () => void;
}

export const SessionList: React.FC<Props> = ({
  sessions,
  currentSessionId,
  onSelect,
  onNew,
  onDelete,
  onClose,
}) => {
  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000);
    const now = new Date();
    if (d.toDateString() === now.toDateString()) {
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  };

  return (
    <div className="cua-sessions-overlay" onClick={onClose}>
      <div className="cua-sessions-panel" onClick={(e) => e.stopPropagation()}>
        <div className="cua-sessions-header">
          <span className="cua-sessions-title">Sessions</span>
          <button className="cua-sessions-new" onClick={onNew}>
            + New
          </button>
        </div>
        <div className="cua-sessions-list">
          {sessions.length === 0 && (
            <div className="cua-sessions-empty">No sessions yet</div>
          )}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`cua-session-item ${
                s.id === currentSessionId ? "cua-session-active" : ""
              }`}
              onClick={() => onSelect(s.id)}
            >
              <div className="cua-session-info">
                <div className="cua-session-name">
                  {s.title || "Untitled"}
                </div>
                <div className="cua-session-time">
                  {formatTime(s.updated_at)}
                </div>
              </div>
              <button
                className="cua-session-delete"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(s.id);
                }}
                title="Delete session"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
