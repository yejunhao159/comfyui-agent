import React, { useCallback, useEffect, useState } from "react";

interface ConfigData {
  llm: {
    provider: string;
    model: string;
    max_tokens: number;
    base_url: string;
    api_key_set: boolean;
    api_key_masked: string;
  };
  web: {
    tavily_api_key_set: boolean;
    tavily_api_key_masked: string;
  };
  comfyui: {
    base_url: string;
  };
}

interface Props {
  baseUrl: string;
  onClose: () => void;
}

export const SettingsPanel: React.FC<Props> = ({ baseUrl, onClose }) => {
  const [config, setConfig] = useState<ConfigData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [llmBaseUrl, setLlmBaseUrl] = useState("");
  const [maxTokens, setMaxTokens] = useState("");
  const [tavilyKey, setTavilyKey] = useState("");

  const fetchConfig = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${baseUrl}/api/config`);
      const data: ConfigData = await res.json();
      setConfig(data);
      setModel(data.llm.model);
      setLlmBaseUrl(data.llm.base_url);
      setMaxTokens(String(data.llm.max_tokens));
    } catch {
      setMessage("无法加载配置");
    } finally {
      setLoading(false);
    }
  }, [baseUrl]);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setMessage("");
    try {
      const body: Record<string, unknown> = {};
      const llm: Record<string, unknown> = {};
      if (apiKey) llm.api_key = apiKey;
      if (model) llm.model = model;
      if (llmBaseUrl) llm.base_url = llmBaseUrl;
      if (maxTokens) llm.max_tokens = Number(maxTokens);
      if (Object.keys(llm).length > 0) body.llm = llm;

      const web: Record<string, unknown> = {};
      if (tavilyKey) web.tavily_api_key = tavilyKey;
      if (Object.keys(web).length > 0) body.web = web;

      const res = await fetch(`${baseUrl}/api/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data.status === "ok") {
        setMessage(`已保存: ${data.updated.join(", ")}`);
        setApiKey("");
        setTavilyKey("");
        await fetchConfig();
      } else {
        setMessage("保存失败");
      }
    } catch {
      setMessage("保存失败");
    } finally {
      setSaving(false);
    }
  }, [baseUrl, apiKey, model, llmBaseUrl, maxTokens, tavilyKey, fetchConfig]);

  return (
    <div className="cua-settings-overlay" onClick={onClose}>
      <div className="cua-settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="cua-settings-header">
          <span className="cua-settings-title">设置</span>
          <button className="cua-header-btn" onClick={onClose}>✕</button>
        </div>
        <div className="cua-settings-body">
          {loading ? (
            <div className="cua-settings-loading">加载中...</div>
          ) : (
            <>
              <div className="cua-settings-section">
                <div className="cua-settings-section-title">LLM 配置</div>
                <label className="cua-settings-label">
                  API Key
                  {config?.llm.api_key_set && (
                    <span className="cua-settings-hint">
                      当前: {config.llm.api_key_masked}
                    </span>
                  )}
                </label>
                <input
                  className="cua-settings-input"
                  type="password"
                  placeholder="输入新的 API Key..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
                <label className="cua-settings-label">模型</label>
                <input
                  className="cua-settings-input"
                  placeholder="claude-sonnet-4-5-20250929"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                />
                <label className="cua-settings-label">Base URL</label>
                <input
                  className="cua-settings-input"
                  placeholder="https://api.anthropic.com"
                  value={llmBaseUrl}
                  onChange={(e) => setLlmBaseUrl(e.target.value)}
                />
                <label className="cua-settings-label">Max Tokens</label>
                <input
                  className="cua-settings-input"
                  type="number"
                  placeholder="8192"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(e.target.value)}
                />
              </div>
              <div className="cua-settings-section">
                <div className="cua-settings-section-title">搜索工具</div>
                <label className="cua-settings-label">
                  Tavily API Key
                  {config?.web.tavily_api_key_set && (
                    <span className="cua-settings-hint">
                      当前: {config.web.tavily_api_key_masked}
                    </span>
                  )}
                </label>
                <input
                  className="cua-settings-input"
                  type="password"
                  placeholder="留空则使用 DuckDuckGo"
                  value={tavilyKey}
                  onChange={(e) => setTavilyKey(e.target.value)}
                />
              </div>
              {message && <div className="cua-settings-message">{message}</div>}
              <button
                className="cua-btn cua-btn-send cua-settings-save"
                onClick={handleSave}
                disabled={saving}
              >
                {saving ? "保存中..." : "保存"}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
