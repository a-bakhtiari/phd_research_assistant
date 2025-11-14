import { useState } from 'react'
import { useProject } from '../contexts/ProjectContext'
import {
  FileText,
  Plus,
  Trash2,
  Edit2,
  Save,
  BookOpen,
  MessageSquare,
  Send,
  PenTool,
  ListTree,
  Search,
  Clock,
  FileEdit,
  Sparkles
} from 'lucide-react'

// Types
interface Document {
  id: string
  name: string
  content: string
  wordCount: number
  lastModified: Date
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface Paper {
  id: number
  title: string
  authors?: string
  year?: number
}

type WritingMode = 'draft' | 'edit' | 'structure' | 'analysis'

export default function WritingAssistant() {
  const { currentProject } = useProject()

  // Document management state
  const [documents, setDocuments] = useState<Document[]>([
    {
      id: '1',
      name: 'Literature Review.txt',
      content: 'Literature Review\n\nIntroduction\n\nThis section explores recent advances in artificial intelligence and large language models...',
      wordCount: 145,
      lastModified: new Date()
    }
  ])
  const [selectedDocId, setSelectedDocId] = useState<string>('1')
  const [isEditingName, setIsEditingName] = useState<string | null>(null)
  const [editedName, setEditedName] = useState('')

  // Editor state
  const [editorContent, setEditorContent] = useState(documents[0].content)
  const [writingMode, setWritingMode] = useState<WritingMode>('draft')
  const [lastSaved, setLastSaved] = useState<Date>(new Date())

  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m your AI writing assistant. I can help you with drafting, editing, structuring your document, and analyzing your claims with references to your paper library. What would you like to work on?',
      timestamp: new Date()
    }
  ])
  const [chatInput, setChatInput] = useState('')

  // Mock papers from library (would come from API in production)
  const [referencePapers] = useState<Paper[]>([
    { id: 1, title: 'Why and how to embrace AI such as ChatGPT in your academic life', authors: 'Lin et al.', year: 2023 },
    { id: 2, title: 'Recent Advances in Generative AI and Large Language Models', authors: 'Hagos et al.', year: 2024 },
    { id: 3, title: 'ChatGPT: Bullshit spewer or the end of traditional assessments', authors: 'Rudolph', year: 2023 }
  ])

  if (!currentProject) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-gray-600 dark:text-gray-400">Please select a project</p>
      </div>
    )
  }

  const selectedDoc = documents.find(d => d.id === selectedDocId)
  const wordCount = editorContent.trim().split(/\s+/).filter(w => w.length > 0).length

  // Handlers
  const handleCreateDocument = () => {
    const newDoc: Document = {
      id: Date.now().toString(),
      name: 'Untitled Document.txt',
      content: '',
      wordCount: 0,
      lastModified: new Date()
    }
    setDocuments([...documents, newDoc])
    setSelectedDocId(newDoc.id)
    setEditorContent('')
  }

  const handleDeleteDocument = (docId: string) => {
    if (documents.length <= 1) {
      alert('Cannot delete the last document')
      return
    }
    const newDocs = documents.filter(d => d.id !== docId)
    setDocuments(newDocs)
    if (selectedDocId === docId) {
      setSelectedDocId(newDocs[0].id)
      setEditorContent(newDocs[0].content)
    }
  }

  const handleRenameDocument = (docId: string, newName: string) => {
    setDocuments(documents.map(d =>
      d.id === docId ? { ...d, name: newName } : d
    ))
    setIsEditingName(null)
  }

  const handleSelectDocument = (docId: string) => {
    // Save current document
    if (selectedDoc) {
      setDocuments(documents.map(d =>
        d.id === selectedDoc.id
          ? { ...d, content: editorContent, wordCount, lastModified: new Date() }
          : d
      ))
    }

    const doc = documents.find(d => d.id === docId)
    if (doc) {
      setSelectedDocId(docId)
      setEditorContent(doc.content)
    }
  }

  const handleSaveDocument = () => {
    if (selectedDoc) {
      setDocuments(documents.map(d =>
        d.id === selectedDoc.id
          ? { ...d, content: editorContent, wordCount, lastModified: new Date() }
          : d
      ))
      setLastSaved(new Date())
    }
  }

  const handleSendMessage = () => {
    if (!chatInput.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: chatInput,
      timestamp: new Date()
    }

    setMessages([...messages, userMessage])
    setChatInput('')

    // Simulate AI response (would call API in production)
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: getMockResponse(writingMode),
        timestamp: new Date()
      }
      setMessages(prev => [...prev, assistantMessage])
    }, 1000)
  }

  const getMockResponse = (mode: WritingMode): string => {
    const responses = {
      draft: 'Great! I can help you draft this section. Based on your paper library, I found relevant work by Lin et al. (2023) that discusses embracing AI in academic life. Would you like me to incorporate insights from this paper?',
      edit: 'I\'ve reviewed your text for clarity and grammar. Here are my suggestions: Consider breaking up long sentences and using more active voice. The argument in paragraph 2 could be strengthened with a citation.',
      structure: 'Looking at your document structure, I notice the introduction could benefit from a clearer thesis statement. The body paragraphs follow a logical flow, but consider adding transition sentences between sections 2 and 3.',
      analysis: 'I\'ve analyzed your claims against your paper library. Your statement about AI adoption in education (paragraph 1) aligns well with findings from Rudolph (2023) and Hagos et al. (2024). However, claim in paragraph 3 needs additional support.'
    }
    return responses[mode]
  }

  const getModeIcon = (mode: WritingMode) => {
    switch (mode) {
      case 'draft': return <PenTool className="w-4 h-4" />
      case 'edit': return <Edit2 className="w-4 h-4" />
      case 'structure': return <ListTree className="w-4 h-4" />
      case 'analysis': return <Search className="w-4 h-4" />
    }
  }

  const getModeDescription = (mode: WritingMode): string => {
    switch (mode) {
      case 'draft': return 'Exploratory writing with AI suggestions'
      case 'edit': return 'Refine clarity, grammar, and style'
      case 'structure': return 'Organize and outline your document'
      case 'analysis': return 'Check claims against your research library'
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-blue-600" />
              Writing Assistant
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              AI-powered academic writing with research integration
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>Saved {lastSaved.toLocaleTimeString()}</span>
            </div>
            <button
              onClick={handleSaveDocument}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Save className="w-4 h-4" />
              Save
            </button>
          </div>
        </div>
      </div>

      {/* Main 3-Column Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Column: Documents & Papers */}
        <div className="w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
          {/* Documents Section */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-3">
                <h2 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  Documents
                </h2>
                <button
                  onClick={handleCreateDocument}
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                  title="New Document"
                >
                  <Plus className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                </button>
              </div>

              <div className="space-y-1">
                {documents.map(doc => (
                  <div
                    key={doc.id}
                    className={`group p-3 rounded-lg cursor-pointer transition-colors ${
                      selectedDocId === doc.id
                        ? 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800'
                        : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
                    }`}
                    onClick={() => handleSelectDocument(doc.id)}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        {isEditingName === doc.id ? (
                          <input
                            type="text"
                            value={editedName}
                            onChange={(e) => setEditedName(e.target.value)}
                            onBlur={() => handleRenameDocument(doc.id, editedName)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') handleRenameDocument(doc.id, editedName)
                              if (e.key === 'Escape') setIsEditingName(null)
                            }}
                            className="w-full px-2 py-1 text-sm border border-blue-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                            autoFocus
                            onClick={(e) => e.stopPropagation()}
                          />
                        ) : (
                          <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
                            {doc.name}
                          </div>
                        )}
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          {doc.wordCount} words
                        </div>
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setIsEditingName(doc.id)
                            setEditedName(doc.name)
                          }}
                          className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded"
                          title="Rename"
                        >
                          <Edit2 className="w-3 h-3 text-gray-600 dark:text-gray-400" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteDocument(doc.id)
                          }}
                          className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded"
                          title="Delete"
                        >
                          <Trash2 className="w-3 h-3 text-red-600 dark:text-red-400" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Referenced Papers Section */}
            <div className="p-4">
              <h2 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <BookOpen className="w-4 h-4" />
                Reference Papers
              </h2>
              <div className="space-y-2">
                {referencePapers.map(paper => (
                  <div
                    key={paper.id}
                    className="p-3 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-700 transition-colors cursor-pointer"
                    title="Click to insert citation"
                  >
                    <div className="text-sm font-medium text-gray-900 dark:text-white line-clamp-2">
                      {paper.title}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {paper.authors} ({paper.year})
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Center Column: Editor */}
        <div className="flex-1 flex flex-col bg-white dark:bg-gray-800">
          {/* Mode Selector */}
          <div className="border-b border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Mode:</span>
              <div className="flex gap-2">
                {(['draft', 'edit', 'structure', 'analysis'] as WritingMode[]).map(mode => (
                  <button
                    key={mode}
                    onClick={() => setWritingMode(mode)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      writingMode === mode
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                    title={getModeDescription(mode)}
                  >
                    {getModeIcon(mode)}
                    <span className="capitalize">{mode}</span>
                  </button>
                ))}
              </div>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              {getModeDescription(writingMode)}
            </p>
          </div>

          {/* Editor */}
          <div className="flex-1 overflow-y-auto p-6">
            <textarea
              value={editorContent}
              onChange={(e) => setEditorContent(e.target.value)}
              placeholder="Start writing here..."
              className="w-full h-full min-h-full px-4 py-3 border-0 focus:outline-none focus:ring-0 bg-transparent text-gray-900 dark:text-white resize-none font-mono text-sm leading-relaxed"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            />
          </div>

          {/* Editor Footer */}
          <div className="border-t border-gray-200 dark:border-gray-700 px-6 py-3 flex items-center justify-between bg-gray-50 dark:bg-gray-800/50">
            <div className="flex items-center gap-4 text-xs text-gray-600 dark:text-gray-400">
              <span>{wordCount} words</span>
              <span>•</span>
              <span>{editorContent.split('\n').length} lines</span>
              <span>•</span>
              <span>{editorContent.length} characters</span>
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Auto-save enabled
            </div>
          </div>
        </div>

        {/* Right Column: AI Chat Assistant */}
        <div className="w-96 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 flex flex-col">
          {/* Chat Header */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
              <MessageSquare className="w-4 h-4" />
              AI Assistant
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Ask questions or get writing suggestions
            </p>
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex gap-3 ${
                  message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                }`}
              >
                <div
                  className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                  }`}
                >
                  {message.role === 'user' ? (
                    <FileEdit className="w-4 h-4" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                </div>
                <div
                  className={`flex-1 rounded-lg p-3 ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  <p className={`text-xs mt-2 ${
                    message.role === 'user' ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'
                  }`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Chat Input */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSendMessage()
                  }
                }}
                placeholder="Ask for help or suggestions..."
                className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={handleSendMessage}
                disabled={!chatInput.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Press Enter to send, Shift+Enter for new line
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
