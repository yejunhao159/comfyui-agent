import React, { useCallback, useEffect, useState } from "react";
import { useAgentWS } from "./hooks/useAgentWS";
import { useSessionAPI } from "./hooks/useSessionAPI";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { InputArea } from "./components/InputArea";
import { SessionList } from "./components/SessionList";
import { SettingsPanel } from "./components/SettingsPanel";

// Default agent URL — can be overridden via settings
const DEFAULT_AGENT_WS = "ws://127.0.0.1:5200/api/chat/ws";
const DEFAULT_AGENT_HTTP = "http://127.0.0.1:5200";

interface Props {
  agentUrl?: string;
  agentHttpUrl?: string;
}

const App: React.FC<Props> = ({
  agentUrl = DEFAULT_AGENT_WS,
  agentHttpUrl = DEFAULT_AGENT_HTTP,
}) => {
  const {
    status,
    items,
    sessionId,
    isProcessing,
    sendMessage,
    cancelRequest,
    switchSession,
    newSession,
  } = useAgentWS(agentUrl);

  const {
    sessions,
    fetchSessions,
    deleteSession,
    loadMessages,
  } = useSessionAPI(agentHttpUrl);

  const [showSessions, setShowSessions] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // Load session list on mount and when connected
  useEffect(() => {
    if (status === "connected") {
      fetchSessions();
      if (sessionId && items.length === 0) {
        loadMessages(sessionId).then((loaded) => {
          if (loaded.length > 0) {
            switchSession(sessionId, loaded);
          }
        });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  const handleSelectSession = useCallback(
    async (id: string) => {
      const loaded = await loadMessages(id);
      switchSession(id, loaded);
      setShowSessions(false);
    },
    [loadMessages, switchSession]
  );

  const handleNewSession = useCallback(() => {
    newSession();
    setShowSessions(false);
  }, [newSession]);

  const handleDeleteSession = useCallback(
    async (id: string) => {
      await deleteSession(id);
      if (id === sessionId) {
        newSession();
      }
    },
    [deleteSession, sessionId, newSession]
  );

  const handleToggleSessions = useCallback(() => {
    if (!showSessions) {
      fetchSessions();
    }
    setShowSessions((v) => !v);
  }, [showSessions, fetchSessions]);

  const handleToggleSettings = useCallback(() => {
    setShowSettings((v) => !v);
  }, []);

  return (
    <div className="cua-root">
      <Header
        status={status}
        onClear={handleNewSession}
        onToggleSessions={handleToggleSessions}
        onToggleSettings={handleToggleSettings}
      />
      <MessageList items={items} isStreaming={isProcessing} />
      <InputArea
        onSend={sendMessage}
        onCancel={cancelRequest}
        isProcessing={isProcessing}
        disabled={status !== "connected"}
      />
      {showSessions && (
        <SessionList
          sessions={sessions}
          currentSessionId={sessionId}
          onSelect={handleSelectSession}
          onNew={handleNewSession}
          onDelete={handleDeleteSession}
          onClose={() => setShowSessions(false)}
        />
      )}
      {showSettings && (
        <SettingsPanel
          baseUrl={agentHttpUrl}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
};

export default App;
