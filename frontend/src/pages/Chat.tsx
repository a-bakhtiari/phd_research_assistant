import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useProject } from '../contexts/ProjectContext'
import { chatApi } from '../services/api'
import type { ChatSession, ChatMessage } from '../services/types'
import { MessageSquare, Send, Plus, Trash2, FileText, Sparkles, ChevronDown, ChevronUp } from 'lucide-react'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import Modal from '../components/ui/Modal'

export default function Chat() {
  const { currentProject } = useProject()
  const queryClient = useQueryClient()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // State
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [messageInput, setMessageInput] = useState('')
  const [showNewSessionModal, setShowNewSessionModal] = useState(false)
  const [newSessionTitle, setNewSessionTitle] = useState('')
  const [useRag, setUseRag] = useState(true)
  const [isTyping, setIsTyping] = useState(false)
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set())
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Fetch sessions
  const { data: sessions, isLoading: sessionsLoading } = useQuery({
    queryKey: ['chat-sessions', currentProject?.id],
    queryFn: () => chatApi.listSessions(currentProject!.id),
    enabled: !!currentProject,
  })

  // Fetch messages for current session
  const { data: messages, isLoading: messagesLoading } = useQuery({
    queryKey: ['chat-messages', currentProject?.id, currentSessionId],
    queryFn: () => chatApi.getMessages(currentProject!.id, currentSessionId!),
    enabled: !!currentProject && !!currentSessionId,
  })

  // Create session mutation
  const createSessionMutation = useMutation({
    mutationFn: (title: string) =>
      chatApi.createSession({
        project_id: currentProject!.id,
        title,
      }),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({ queryKey: ['chat-sessions'] })
      setCurrentSessionId(newSession.session_id)
      setShowNewSessionModal(false)
      setNewSessionTitle('')
    },
    onError: (error: Error) => {
      console.error('Failed to create chat session:', error)
      alert(`Failed to create chat session: ${error.message}`)
    },
  })

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: (message: string) =>
      chatApi.sendMessage(currentProject!.id, currentSessionId!, {
        message,
        max_sources: 5,
        use_rag: useRag,
      }),
    onMutate: () => {
      setIsTyping(true)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-messages'] })
      setMessageInput('')
      setIsTyping(false)
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    },
    onError: () => {
      setIsTyping(false)
    },
  })

  // Delete session mutation
  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) =>
      chatApi.deleteSession(currentProject!.id, sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-sessions'] })
      if (currentSessionId === deleteSessionMutation.variables) {
        setCurrentSessionId(null)
      }
    },
  })

  // Auto-select first session
  useEffect(() => {
    if (sessions && sessions.length > 0 && !currentSessionId) {
      setCurrentSessionId(sessions[0].session_id)
    }
  }, [sessions, currentSessionId])

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Handle send message
  const handleSendMessage = () => {
    if (messageInput.trim() && currentSessionId) {
      sendMessageMutation.mutate(messageInput)
    }
  }

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // Auto-resize textarea
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessageInput(e.target.value)

    // Auto-resize
    const textarea = e.target
    textarea.style.height = 'auto'
    const newHeight = Math.min(textarea.scrollHeight, 150) // Max 150px (~6 rows)
    textarea.style.height = `${newHeight}px`
  }

  // Toggle source expansion
  const toggleSources = (messageId: string) => {
    setExpandedSources(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
  }

  if (!currentProject) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-gray-600 dark:text-gray-400">Please select a project</p>
      </div>
    )
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar - Sessions List */}
      <div className="w-80 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setShowNewSessionModal(true)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {sessionsLoading ? (
            <div className="flex justify-center py-8">
              <LoadingSpinner />
            </div>
          ) : sessions && sessions.length > 0 ? (
            sessions.map((session) => (
              <div
                key={session.session_id}
                className={`group p-3 rounded-lg cursor-pointer transition-colors ${
                  currentSessionId === session.session_id
                    ? 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                onClick={() => setCurrentSessionId(session.session_id)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {session.title}
                      </h3>
                      {session.message_count > 0 && (
                        <span className="flex-shrink-0 px-1.5 py-0.5 text-xs font-medium bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded">
                          {session.message_count}
                        </span>
                      )}
                    </div>
                    {session.last_message && (
                      <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-1 mb-1">
                        {session.last_message}
                      </p>
                    )}
                    <p className="text-xs text-gray-500 dark:text-gray-500">
                      {new Date(session.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      if (confirm('Delete this chat session?')) {
                        deleteSessionMutation.mutate(session.session_id)
                      }
                    }}
                    className="flex-shrink-0 opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded transition-opacity"
                  >
                    <Trash2 className="w-4 h-4 text-red-600 dark:text-red-400" />
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8">
              <MessageSquare className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-sm text-gray-600 dark:text-gray-400">
                No chat sessions yet
              </p>
            </div>
          )}
        </div>

        {/* RAG Mode Toggle */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
          <div className="flex items-center justify-between mb-2">
            <label htmlFor="rag-toggle" className="text-sm font-medium text-gray-900 dark:text-white">
              Search Mode
            </label>
            <button
              id="rag-toggle"
              role="switch"
              aria-checked={useRag}
              onClick={() => setUseRag(!useRag)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                useRag ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  useRag ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            {useRag ? (
              <>
                <FileText className="inline w-3 h-3 mr-1" />
                Searching papers for answers
              </>
            ) : (
              <>
                <Sparkles className="inline w-3 h-3 mr-1" />
                Direct LLM responses
              </>
            )}
          </p>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {currentSessionId ? (
          <>
            {/* Header */}
            <div className="p-6 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                {sessions?.find(s => s.session_id === currentSessionId)?.title || 'Chat'}
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Ask questions about your research papers
              </p>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 bg-gray-50 dark:bg-gray-900">
              {messagesLoading ? (
                <div className="flex justify-center py-12">
                  <LoadingSpinner size="lg" />
                </div>
              ) : messages && messages.length > 0 ? (
                <div className="space-y-4">
                  {messages.map((message) => (
                    <div
                      key={message.message_id}
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`rounded-2xl p-4 shadow-sm ${
                          message.role === 'user'
                            ? 'bg-blue-600 text-white max-w-[80%]'
                            : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 max-w-[85%]'
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          {message.role === 'assistant' && (
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                              <Sparkles className="w-4 h-4 text-white" />
                            </div>
                          )}
                          <div className="flex-1 min-w-0">
                            <p className={`text-sm leading-relaxed whitespace-pre-wrap ${
                              message.role === 'user' ? 'text-white' : 'text-gray-900 dark:text-white'
                            }`}>
                              {message.content}
                            </p>

                            {/* Sources - Collapsible */}
                            {message.sources && message.sources.length > 0 && (
                              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                                <button
                                  onClick={() => toggleSources(message.message_id)}
                                  className="flex items-center gap-2 text-xs font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
                                >
                                  {expandedSources.has(message.message_id) ? (
                                    <ChevronUp className="w-4 h-4" />
                                  ) : (
                                    <ChevronDown className="w-4 h-4" />
                                  )}
                                  <span>{message.sources.length} {message.sources.length === 1 ? 'source' : 'sources'}</span>
                                </button>

                                {expandedSources.has(message.message_id) && (
                                  <div className="space-y-2 mt-3">
                                    {message.sources.map((source, idx) => (
                                      <div
                                        key={idx}
                                        className="flex items-start gap-2 p-2 rounded bg-gray-50 dark:bg-gray-900/50 text-xs"
                                      >
                                        <FileText className="w-3 h-3 text-gray-500 dark:text-gray-400 flex-shrink-0 mt-0.5" />
                                        <div className="flex-1 min-w-0">
                                          <p className="text-gray-700 dark:text-gray-300 font-medium truncate">
                                            {source.paper_title}
                                          </p>
                                          <p className="text-gray-600 dark:text-gray-400 line-clamp-2 mt-1">
                                            {source.content}
                                          </p>
                                          <p className="text-gray-500 dark:text-gray-500 mt-1">
                                            Relevance: {(source.similarity_score * 100).toFixed(1)}%
                                          </p>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )}

                            {/* Metadata */}
                            <div className={`mt-3 flex items-center gap-3 text-xs ${
                              message.role === 'user' ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'
                            }`}>
                              <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
                              {message.tokens_used && typeof message.tokens_used === 'object' && (
                                <span>{message.tokens_used.total_tokens} tokens</span>
                              )}
                              {message.confidence_score !== undefined && (
                                <span>
                                  Confidence: {(message.confidence_score * 100).toFixed(0)}%
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Typing Indicator */}
                  {isTyping && (
                    <div className="flex justify-start">
                      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl p-4 shadow-sm max-w-[85%]">
                        <div className="flex items-start gap-3">
                          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                            <Sparkles className="w-4 h-4 text-white" />
                          </div>
                          <div className="flex items-center gap-1 py-1">
                            <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                            <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                            <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <MessageSquare className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 dark:text-gray-400">
                      Start a conversation by sending a message
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="p-6 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
              <div className="flex gap-3 items-end">
                <textarea
                  ref={textareaRef}
                  value={messageInput}
                  onChange={handleTextareaChange}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask a question about your papers..."
                  rows={1}
                  className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  style={{ minHeight: '48px', maxHeight: '150px' }}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!messageInput.trim() || sendMessageMutation.isPending}
                  className="flex-shrink-0 p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                  style={{ minHeight: '48px', minWidth: '48px' }}
                >
                  {sendMessageMutation.isPending ? (
                    <LoadingSpinner size="sm" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full bg-gray-50 dark:bg-gray-900">
            <div className="text-center">
              <MessageSquare className="w-20 h-20 text-gray-400 mx-auto mb-4" />
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                No Chat Selected
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                Select a chat or create a new one to get started
              </p>
              <button
                onClick={() => setShowNewSessionModal(true)}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Start New Chat
              </button>
            </div>
          </div>
        )}
      </div>

      {/* New Session Modal */}
      <Modal
        isOpen={showNewSessionModal}
        onClose={() => setShowNewSessionModal(false)}
        title="New Chat Session"
        size="sm"
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Session Title
            </label>
            <input
              type="text"
              value={newSessionTitle}
              onChange={(e) => setNewSessionTitle(e.target.value)}
              placeholder="e.g., Deep Learning Papers Discussion"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              autoFocus
            />
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => {
                if (newSessionTitle.trim()) {
                  createSessionMutation.mutate(newSessionTitle)
                }
              }}
              disabled={!newSessionTitle.trim() || createSessionMutation.isPending}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              Create
            </button>
            <button
              onClick={() => setShowNewSessionModal(false)}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </Modal>
    </div>
  )
}
