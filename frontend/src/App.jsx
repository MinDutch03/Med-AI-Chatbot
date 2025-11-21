import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sourceDocs, setSourceDocs] = useState([])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = { text: input, sender: 'user' }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setSourceDocs([])

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      const botMessage = { text: data.llm_answer, sender: 'bot' }
      setMessages(prev => [...prev, botMessage])
      setSourceDocs(data.results)
    } catch (error) {
      console.error("Failed to fetch:", error)
      const errorMessage = { text: "Sorry, I couldn't connect to the server. Please try again.", sender: 'bot', isError: true }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Private Medical Chatbot</h1>
        <p>Powered by Mistral-7B & RAG</p>
      </header>
      <div className="chat-container">
        <div className="message-list">
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
  )
}

export default App
