import React, { useState, useRef, useEffect } from 'react';
import audioArbiter, { AudioPriority } from '../services/AudioArbiter';

export default function Chatbot({
  userId = 1,
  currentExercise = 'Squats',
  apiBase = 'http://localhost:8000',
  onVoiceSpeak
}) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hello! I am your AI Gym Coach. I can analyze your real-time exercise form, explain biomechanics using sports science research, and tailor workout advice to your profile. How can I help you today?",
      tools_called: [],
      rag_diagnostic: null
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(true);
  const [showRAGInspector, setShowRAGInspector] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const quickPrompts = [
    `How do I perform ${currentExercise} with proper form?`,
    `Explain my recent ${currentExercise} performance and form errors.`,
    `Why is camera positioning important for ${currentExercise}?`,
    `Suggest workout adjustments based on my fitness level.`
  ];

  const handleSend = async (messageText = null) => {
    const textToSend = messageText || inputMessage;
    if (!textToSend || !textToSend.trim() || isLoading) return;

    const userMsg = {
      role: 'user',
      content: textToSend.trim()
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Build recent history for context
      const chatHistory = newMessages.slice(-6).map((m) => ({
        role: m.role,
        content: m.content
      }));

      // Call single agent
      const res = await fetch(`${apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: textToSend.trim(),
          user_id: userId,
          exercise: currentExercise,
          chat_history: chatHistory,
          request_voice: true
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();

      // Optionally fetch RAG diagnostic trace for transparency
      let ragDiag = null;
      try {
        const diagRes = await fetch(`${apiBase}/api/rag/diagnostic?query=${encodeURIComponent(textToSend.trim())}&exercise=${encodeURIComponent(currentExercise)}`);
        if (diagRes.ok) {
          ragDiag = await diagRes.json();
        }
      } catch (e) {
        // Optional diagnostic fetching failure does not block chat
      }

      const assistantMsg = {
        role: 'assistant',
        content: data.response || 'I have analyzed your request.',
        tools_called: data.tools_called || [],
        model_used: data.model_used,
        rag_diagnostic: ragDiag
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Route chatbot speech through Centralized AudioArbiter (Priority 5)
      if (data.response) {
        const sentenceMatch = data.response.match(/^.*?[.!?](\s|$)/);
        const spokenSentence = data.spoken_text || (sentenceMatch ? sentenceMatch[0].trim() : data.response.slice(0, 150));
        audioArbiter.requestSpeech({
          text: spokenSentence,
          audio: data.spoken_audio || null,
          priority: AudioPriority.CHATBOT,
          source: 'chatbot'
        });
      }

      if (onVoiceSpeak && data.response) {
        const firstSentence = data.response.split('.')[0] + '.';
        onVoiceSpeak(firstSentence);
      }
    } catch (err) {
      console.error('Chatbot error:', err);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ Coaching agent error: ${err.message}. Please ensure the FastAPI backend is running on port 8000.`,
          tools_called: []
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={`chatbot-container ${isOpen ? 'expanded' : 'minimized'}`}>
      <div className="chatbot-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="chatbot-header-title">
          <span className="chatbot-logo">🧠</span>
          <div>
            <h4 className="chatbot-name">AI Coach Chatbot (LangChain + Gemini)</h4>
            <span className="chatbot-subtitle">Single Agent • 3 Tools (CV Analysis, RAG, Profile)</span>
          </div>
        </div>
        <div className="chatbot-header-controls">
          <button
            type="button"
            className={`btn-rag-toggle ${showRAGInspector ? 'active' : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              setShowRAGInspector(!showRAGInspector);
            }}
            title="Toggle RAG Retrieval Inspector"
          >
            {showRAGInspector ? 'Hide RAG Chunks' : 'RAG Inspector'}
          </button>
          <button
            type="button"
            className="chatbot-toggle-btn"
            aria-label={isOpen ? 'Minimize chatbot' : 'Expand chatbot'}
          >
            {isOpen ? '−' : '+'}
          </button>
        </div>
      </div>

      {isOpen && (
        <div className="chatbot-body">
          {/* Quick Prompts */}
          <div className="quick-prompts-row">
            {quickPrompts.map((prompt, idx) => (
              <button
                key={idx}
                type="button"
                className="quick-prompt-chip"
                onClick={() => handleSend(prompt)}
                disabled={isLoading}
              >
                {prompt}
              </button>
            ))}
          </div>

          {/* Messages Scroll Area */}
          <div className="chatbot-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🏋️‍♂️'}
                </div>
                <div className="message-content">
                  <div className="message-sender">
                    {msg.role === 'user' ? 'You' : 'AI Gym Coach'}
                  </div>
                  <div className="message-text">
                    {msg.content.split('\n').map((line, lidx) => (
                      <p key={lidx}>{line}</p>
                    ))}
                  </div>

                  {msg.tools_called && msg.tools_called.length > 0 && (
                    <div className="tools-badge-row">
                      <span className="tools-label">Tools Executed:</span>
                      {msg.tools_called.map((toolName, tidx) => (
                        <span key={tidx} className="tool-tag">
                          🔧 {toolName.replace(/_/g, ' ')}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* RAG Retrieval Diagnostic Inspector */}
                  {showRAGInspector && msg.rag_diagnostic && (
                    <div className="rag-inspector-drawer">
                      <div className="rag-drawer-title">
                        📚 RAG Retrieved Knowledge Chunks (Filter: {msg.rag_diagnostic.exercise_filter || 'None'})
                      </div>
                      <div className="rag-chunks-list">
                        {msg.rag_diagnostic.accepted_passages && msg.rag_diagnostic.accepted_passages.map((p, pidx) => (
                          <div key={pidx} className="rag-chunk-card">
                            <div className="chunk-header">
                              <span className="chunk-id">{p.chunk_id}</span>
                              <span className="chunk-score">Score: {p.score}</span>
                            </div>
                            <div className="chunk-section">Section: <strong>{p.section}</strong></div>
                            <div className="chunk-citation">Source: {p.citation}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="chat-message assistant loading">
                <div className="message-avatar">🏋️‍♂️</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span />
                    <span />
                    <span />
                  </div>
                  <span className="typing-label">Consulting tools & sports science literature...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Row */}
          <div className="chatbot-input-row">
            <input
              type="text"
              className="chatbot-input"
              placeholder={`Ask about ${currentExercise} form, science, or your profile...`}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
            <button
              type="button"
              className="chatbot-send-btn"
              onClick={() => handleSend()}
              disabled={isLoading || !inputMessage.trim()}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
