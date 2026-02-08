import { useCallback, useState } from "react";
import type { ChatItem, Session } from "../types";

const STORAGE_KEY = "comfyui-agent-session";

/** HTTP client for session CRUD + message loading. */
export function useSessionAPI(baseUrl: string) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(false);

  const api = useCallback(
    async (path: string, opts?: RequestInit) => {
      const res = await fetch(`${baseUrl}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...opts,
      });
      return res.json();
    },
    [baseUrl]
  );

  const fetchSessions = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api("/api/sessions");
      setSessions(data.sessions ?? []);
    } finally {
      setLoading(false);
    }
  }, [api]);

  const createSession = useCallback(
    async (title = "New Session"): Promise<string> => {
      const data = await api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({ title }),
      });
      await fetchSessions();
      return data.session_id;
    },
    [api, fetchSessions]
  );

  const deleteSession = useCallback(
    async (sessionId: string) => {
      await api(`/api/sessions/${sessionId}`, { method: "DELETE" });
      // Clear localStorage if deleting current session
      if (getSavedSessionId() === sessionId) {
        clearSavedSessionId();
      }
      await fetchSessions();
    },
    [api, fetchSessions]
  );

  const loadMessages = useCallback(
    async (sessionId: string): Promise<ChatItem[]> => {
      const data = await api(`/api/sessions/${sessionId}/messages`);
      return (data.items ?? []) as ChatItem[];
    },
    [api]
  );

  return {
    sessions,
    loading,
    fetchSessions,
    createSession,
    deleteSession,
    loadMessages,
  };
}

// --- localStorage helpers ---

export function getSavedSessionId(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function saveSessionId(id: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // ignore
  }
}

export function clearSavedSessionId(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
