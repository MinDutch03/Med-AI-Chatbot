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
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Load chats from localStorage on mount
  useEffect(() => {
    const storedChats = getChatsFromStorage()
    setChats(storedChats)
    // Restore the most recently updated chat if available
    if (storedChats.length > 0) {
      const mostRecent = storedChats.sort((a, b) => 
        new Date(b.updatedAt) - new Date(a.updatedAt)
      )[0]
      setChatId(mostRecent.id)
      setMessages(mostRecent.messages || [])
    }
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
      
      // Close sidebar on mobile after selection
      if (window.innerWidth <= 768) {
        setSidebarOpen(false)
      }
      
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

    // Ensure we have a chatId before sending
    let currentChatId = chatId
    if (!currentChatId) {
      // Generate a client-side chat_id if we don't have one
      currentChatId = crypto.randomUUID()
      setChatId(currentChatId)
    }

    // Optimistically update UI
    setMessages(prev => [...prev, userMessage])

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: currentInput,
          chat_id: currentChatId  // Use the ensured chatId
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      const botMessage = { text: data.llm_answer, sender: 'bot' }
      
      // Update messages with both user and bot messages
      setMessages(prev => {
        const updated = [...prev, botMessage]
        return updated
      })
      
      setSourceDocs(data.results)
      
      // Update chat_id if returned from server (should match, but ensure consistency)
      const finalChatId = data.chat_id || currentChatId
      if (data.chat_id && data.chat_id !== currentChatId) {
        setChatId(data.chat_id)
      }
      
      // Update chat in list
      setChats(prev => {
        const existingChatIndex = prev.findIndex(c => c.id === finalChatId)
        const currentChat = prev.find(c => c.id === finalChatId)
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
            id: finalChatId,
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
      <aside className={`chat-sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <button 
            className="new-chat-button sidebar-new-chat" 
            onClick={createNewChat}
            disabled={isLoading}
            title="Start a new chat"
          >
            + New Chat
          </button>
          <button 
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
          >
            {sidebarOpen ? '←' : '→'}
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
            <button 
              className="sidebar-toggle-mobile"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            >
              ☰
            </button>
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
      {sourceDocs.length > 0 && (
        <footer className="source-docs">
          <h2>Source Documents</h2>
          <ul>
            {sourceDocs.map(doc => (
              <li key={doc.id}>
                <p><strong>Source:</strong> {doc.metadata.source || 'N/A'}</p>
                <p><strong>Relevance:</strong> {doc.score.toFixed(4)}</p>
                <details>
                  <summary>View content</summary>
                  <p>{doc.metadata.text}</p>
                </details>
              </li>
            ))}
          </ul>
        </footer>
      )}
      </div>
    </div>
  )
}

export default App
