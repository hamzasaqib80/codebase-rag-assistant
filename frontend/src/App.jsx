import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { 
  Send, 
  Terminal, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  Database,
  Search,
  BookOpen,
  Loader
} from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [question, setQuestion] = useState('');
  const [repoUrl, setRepoUrl] = useState('');
  const [messages, setMessages] = useState([
    { 
      id: 'welcome',
      role: 'assistant', 
      content: "Hello! I am your technical onboarding assistant. Ask me anything about the indexed repository (e.g. codebase structure, class behaviour, or git commits). Make sure you run the ingestion pipeline if you haven't already!" 
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [ingestStatus, setIngestStatus] = useState('idle'); // idle, running, success, failed
  const [logs, setLogs] = useState([]);

  const logsEndRef = useRef(null);
  const chatEndRef = useRef(null);

  // Poll ingestion status and logs
  useEffect(() => {
    let interval;
    if (ingestStatus === 'running') {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/ingest/status`, {
            headers: { 'ngrok-skip-browser-warning': 'true' }
          });
          const data = await res.json();
          setIngestStatus(data.status);
          setLogs(data.logs || []);
        } catch (err) {
          console.error("Failed to poll status", err);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [ingestStatus]);

  // Scroll to bottom of logs when they update
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Scroll to bottom of chat
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, loading]);

  // Initial status check
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/ingest/status`, {
          headers: { 'ngrok-skip-browser-warning': 'true' }
        });
        const data = await res.json();
        setIngestStatus(data.status);
        setLogs(data.logs || []);
      } catch (err) {
        console.error("Backend not running or offline", err);
      }
    };
    checkStatus();
  }, []);

  const handleIngest = async (e) => {
    e.preventDefault();
    if (ingestStatus === 'running') return;
    
    setIngestStatus('running');
    setLogs(['Triggering ingestion pipeline...']);
    
    try {
      const res = await fetch(`${API_BASE}/ingest`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true' 
        },
        body: JSON.stringify({ repo_url: repoUrl || null })
      });
      if (!res.ok) {
        throw new Error(`Failed to start ingestion (Status: ${res.status})`);
      }
      const data = await res.json();
      if (data.status === 'started' || data.status === 'already_running') {
        // Polling will take care of the rest
      } else {
        setIngestStatus('failed');
      }
    } catch (err) {
      setIngestStatus('failed');
      setLogs(prev => [...prev, `[ERROR] Connection failed: ${err.message}`]);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    const userMsg = { id: Date.now().toString(), role: 'user', content: question };
    setMessages(prev => [...prev, userMsg]);
    setQuestion('');
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({ question: userMsg.content })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to generate answer');
      }

      const data = await res.json();
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        sources: data.sources
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${err.message}. Make sure you run the ingestion pipeline first.`,
        isError: true
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar / Controls */}
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>RAG Pipeline Control</h2>
          <p>Manage ingestion and vector index</p>
        </div>

        <div className="status-card">
          <div className="status-item">
            <span className="status-label">Pipeline Status:</span>
            <span className={`status-value ${ingestStatus}`}>
              {ingestStatus.toUpperCase()}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Vector DB:</span>
            <span className="status-value success">Pinecone</span>
          </div>
          <div className="status-item">
            <span className="status-label">Embeddings:</span>
            <span className="status-value">Local (all-mpnet)</span>
          </div>
        </div>

        <div className="repo-input-container" style={{ marginBottom: '1rem' }}>
          <input 
            type="text" 
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            placeholder="GitHub Repo URL (optional)"
            className="chat-input"
            style={{ width: '100%', padding: '0.6rem' }}
          />
        </div>

        <button 
          onClick={handleIngest} 
          disabled={ingestStatus === 'running'} 
          className="btn"
        >
          {ingestStatus === 'running' ? (
            <>
              <RefreshCw className="spin" size={18} />
              Indexing Codebase...
            </>
          ) : (
            <>
              <Database size={18} />
              Trigger Ingestion
            </>
          )}
        </button>

        <div className="log-console">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', borderBottom: '1px solid rgba(16, 185, 129, 0.2)', paddingBottom: '0.4rem', marginBottom: '0.5rem' }}>
            <Terminal size={14} />
            <span style={{ fontWeight: 600 }}>Pipeline Logs</span>
          </div>
          {logs.length === 0 ? (
            <span style={{ color: 'var(--text-muted)' }}>Console idle. Click 'Trigger Ingestion' to start.</span>
          ) : (
            logs.map((log, i) => <div key={i}>{log}</div>)
          )}
          <div ref={logsEndRef} />
        </div>
      </div>

      {/* Main Chat Panel */}
      <div className="chat-area">
        <div className="chat-header">
          <div className="chat-header-info">
            <h1>RAG Codebase Assistant</h1>
            <p>Enterprise Technical Onboarding & Q&A Assistant</p>
          </div>
        </div>

        <div className="messages-list">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-wrapper ${msg.role}`}>
              <div className="avatar">
                {msg.role === 'user' ? 'U' : 'AI'}
              </div>
              <div className="message-content" style={msg.isError ? { borderColor: 'var(--accent-error)', color: 'var(--accent-error)' } : {}}>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
                
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources-container">
                    <div className="sources-title">Sources referenced:</div>
                    <div className="sources-list">
                      {msg.sources.map((src, i) => (
                        <div key={i} className="source-tag" title={src.annotation}>
                          {src.label} {src.annotation && <span>[{src.annotation}]</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message-wrapper assistant">
              <div className="avatar">AI</div>
              <div className="message-content" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Loader className="spin" size={16} />
                Thinking...
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="chat-input-container">
          <form onSubmit={handleSend} className="chat-input-form">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about the codebase (e.g. 'How does parent retrieving work?')..."
              className="chat-input"
              disabled={loading}
            />
            <button 
              type="submit" 
              className="chat-submit-btn" 
              disabled={loading || !question.trim()}
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
