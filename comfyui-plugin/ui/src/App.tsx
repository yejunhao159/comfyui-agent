import React from "react";
import { useAgentWS } from "./hooks/useAgentWS";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { InputArea } from "./components/InputArea";

// Default agent URL — can be overridden via settings
const DEFAULT_AGENT_WS = "ws://127.0.0.1:5200/api/chat/ws";

interface Props {
  agentUrl?: string;
}

const App: React.FC<Props> = ({ agentUrl = DEFAULT_AGENT_WS }) => {
  const { status, items, isProcessing, sendMessage, cancelRequest, clearItems } =
    useAgentWS(agentUrl);

  return (
    <div className="cua-root">
      <Header status={status} onClear={clearItems} />
      <MessageList items={items} isStreaming={isProcessing} />
      <InputArea
        onSend={sendMessage}
        onCancel={cancelRequest}
        isProcessing={isProcessing}
        disabled={status !== "connected"}
      />
    </div>
  );
};

export default App;
