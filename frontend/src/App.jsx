import { useState, useCallback, useEffect } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL

// Helper functions for localStorage
const STORAGE_KEY = 'med_ai_chats'

const getChatsFromStorage = () => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

const saveChatsToStorage = (chats) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats))
  } catch (error) {
    console.error('Failed to save chats:', error)
  }
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sourceDocs, setSourceDocs] = useState([])
  const [chatId, setChatId] = useState(null)
  const [chats, setChats] = useState([])

  // Load chats from localStorage on mount
  useEffect(() => {
    const storedChats = getChatsFromStorage()
    setChats(storedChats)
  }, [])

  // Save chats to localStorage whenever chats change
  useEffect(() => {
    if (chats.length > 0) {
      saveChatsToStorage(chats)
    }
  }, [chats])

  const createNewChat = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/chat/new`, {
        method: 'POST',
      })
      if (response.ok) {
        const data = await response.json()
        const newChatId = data.chat_id
        setChatId(newChatId)
        setMessages([])
        setSourceDocs([])
        
        // Add new chat to list
        const newChat = {
          id: newChatId,
          title: 'New Chat',
          messages: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }
        setChats(prev => [newChat, ...prev])
      }
    } catch (error) {
      console.error("Failed to create new chat:", error)
      // Fallback: generate client-side chat_id
      const newChatId = crypto.randomUUID()
      setChatId(newChatId)
      setMessages([])
      setSourceDocs([])
      
      const newChat = {
        id: newChatId,
        title: 'New Chat',
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
      setChats(prev => [newChat, ...prev])
    }
  }, [])

  const switchToChat = useCallback((targetChatId) => {
    const chat = chats.find(c => c.id === targetChatId)
    if (chat) {
      setChatId(targetChatId)
      setMessages(chat.messages || [])
      setSourceDocs([])
      
      // Update updatedAt
      setChats(prev => prev.map(c => 
        c.id === targetChatId 
          ? { ...c, updatedAt: new Date().toISOString() }
          : c
      ))
    }
  }, [chats])

  const deleteChat = useCallback((targetChatId, e) => {
    e.stopPropagation()
    setChats(prev => prev.filter(c => c.id !== targetChatId))
    if (chatId === targetChatId) {
      setChatId(null)
      setMessages([])
      setSourceDocs([])
    }
    saveChatsToStorage(chats.filter(c => c.id !== targetChatId))
  }, [chatId, chats])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = { text: input, sender: 'user' }
    const currentInput = input
    setInput('')
    setIsLoading(true)
    setSourceDocs([])

    // Optimistically update UI
    setMessages(prev => [...prev, userMessage])

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: currentInput,
          chat_id: chatId 
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      // Attach sources to the bot message
      const botMessage = { text: data.llm_answer, sender: 'bot', sources: data.results }
      
      // Update messages with both user and bot messages
      setMessages(prev => {
        const updated = [...prev, botMessage]
        return updated
      })
      
      setSourceDocs(data.results)
      
      // Update chat_id if returned from server
      const currentChatId = data.chat_id || chatId
      if (data.chat_id) {
        setChatId(data.chat_id)
      }
      
      // After POST completes, make GET request to fetch all queries and answers
      try {
        const getResponse = await fetch(`${API_URL}/chat/${currentChatId}`, {
          method: 'GET',
        })
        
        if (getResponse.ok) {
          const chatHistory = await getResponse.json()
          console.log('All queries and answers for chat:', chatHistory)

        }
      } catch (getError) {
        console.error("Failed to fetch chat history:", getError)
      }
      
      // Update chat in list
      setChats(prev => {
        const existingChatIndex = prev.findIndex(c => c.id === currentChatId)
        const currentChat = prev.find(c => c.id === currentChatId)
        const isFirstMessage = !currentChat || currentChat.messages.length === 0
        const chatTitle = isFirstMessage 
          ? (currentInput.substring(0, 30) + (currentInput.length > 30 ? '...' : ''))
          : (currentChat?.title || 'New Chat')
        
        // Get updated messages for this chat
        const updatedMessages = currentChat 
          ? [...currentChat.messages, userMessage, botMessage]
          : [userMessage, botMessage]
        
        if (existingChatIndex >= 0) {
          // Update existing chat
          const updated = [...prev]
          updated[existingChatIndex] = {
            ...updated[existingChatIndex],
            messages: updatedMessages,
            title: updated[existingChatIndex].title === 'New Chat' ? chatTitle : updated[existingChatIndex].title,
            updatedAt: new Date().toISOString()
          }
          return updated
        } else {
          // Create new chat entry
          const newChat = {
            id: currentChatId,
            title: chatTitle,
            messages: updatedMessages,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
          }
          return [newChat, ...prev]
        }
      })
    } catch (error) {
      console.error("Failed to fetch:", error)
      // Remove the optimistically added user message and add error message
      setMessages(prev => {
        const withoutLastUser = prev.filter((msg, idx) => 
          !(idx === prev.length - 1 && msg.sender === 'user')
        )
        return [...withoutLastUser, { text: "Sorry, I couldn't connect to the server. Please try again.", sender: 'bot', isError: true }]
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <aside className="chat-sidebar">
        <div className="sidebar-header">
          <button 
            className="new-chat-button sidebar-new-chat" 
            onClick={createNewChat}
            disabled={isLoading}
            title="Start a new chat"
          >
            + New Chat
          </button>
        </div>
        <div className="chat-list">
          {chats.length === 0 ? (
            <div className="no-chats">No chats yet. Create a new chat to get started!</div>
          ) : (
            chats.map(chat => (
              <div
                key={chat.id}
                className={`chat-item ${chatId === chat.id ? 'active' : ''}`}
                onClick={() => switchToChat(chat.id)}
                title={chat.title}
              >
                <div className="chat-item-content">
                  <div className="chat-title">{chat.title}</div>
                  <div className="chat-preview">
                    {chat.messages.length > 0 
                      ? chat.messages[chat.messages.length - 1].text.substring(0, 50) + (chat.messages[chat.messages.length - 1].text.length > 50 ? '...' : '')
                      : 'Empty chat'}
                  </div>
                </div>
                <button
                  className="delete-chat-button"
                  onClick={(e) => deleteChat(chat.id, e)}
                  title="Delete chat"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>
      </aside>
      <div className="main-content">
        <header className="app-header">
          <div className="header-content">
            <div>
              <h1>Private Medical Chatbot</h1>
              <p>Powered by Mistral-7B & RAG</p>
            </div>
          </div>
        </header>
        <div className="chat-container">
        <div className="message-list">
          {messages.length === 0 && (
            <div className="welcome-message">
              <p>Hello! How can I assist you today? Start a conversation by asking a medical question.</p>
            </div>
          )}
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender} ${msg.isError ? 'error' : ''}`}>
              <p>{msg.text}</p>
              {msg.sender === 'bot' && msg.sources && msg.sources.length > 0 && (
                <div className="message-sources">
                  <details>
                    <summary className="sources-toggle">
                      <span className="sources-icon">📚</span>
                      View {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}
                    </summary>
                    <ul className="sources-list">
                      {msg.sources.map((source, srcIdx) => (
                        <li key={srcIdx} className="source-item">
                          <div className="source-header">
                            {source.metadata.type === 'pubmed' ? (
                              <>
                                <span className="source-badge pubmed">PubMed</span>
                                <span className="source-title">
                                  {source.metadata.journal || 'Medical Journal'}
                                  {source.metadata.year && ` (${source.metadata.year})`}
                                </span>
                              </>
                            ) : (
                              <>
                                <span className="source-badge medquad">MedQuAD</span>
                                <span className="source-title">
                                  {source.metadata.source || source.metadata.focus || 'Medical FAQ'}
                                </span>
                              </>
                            )}
                            <span className="source-score">
                              {(source.score * 100).toFixed(1)}% match
                            </span>
                          </div>
                          {source.metadata.url && (
                            <a 
                              href={source.metadata.url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="source-link"
                            >
                              View original source ↗
                            </a>
                          )}
                          {source.metadata.pmid && (
                            <a 
                              href={`https://pubmed.ncbi.nlm.nih.gov/${source.metadata.pmid}/`} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="source-link"
                            >
                              View on PubMed ↗
                            </a>
                          )}
                          <details className="source-content-details">
                            <summary>View excerpt</summary>
                            <p className="source-excerpt">{source.text}</p>
                          </details>
                        </li>
                      ))}
                    </ul>
                  </details>
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="loading-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
        </div>
        <form onSubmit={handleSubmit} className="message-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a medical question..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>Send</button>
        </form>
      </div>
      </div>
    </div>
  )
}

export default App
