/**
 * API Client for PhD Research Assistant Backend
 *
 * Provides typed functions for all backend endpoints
 */

import axios, { AxiosInstance, AxiosError } from 'axios'
import type {
  Project,
  ProjectCreate,
  ProjectStats,
  Paper,
  PaperUpdate,
  ChatSession,
  ChatSessionCreate,
  ChatMessage,
  ChatMessageRequest,
  RecommendationRequest,
  RecommendationResponse,
  EmbeddingModelInfo,
  LLMSettings,
  DocumentInfo,
  DocumentAnalysis,
  DocumentAnalysisRequest,
  DocumentRecommendationRequest,
  QueueStatus,
  BatchSelectRequest,
  ApiError
} from './types'

// Create axios instance with base configuration
const apiClient: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  timeout: 60000, // 60 seconds for AI-powered operations (recommendations use DeepSeek which can take 30-40s)
  headers: {
    'Content-Type': 'application/json',
  },
})

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiError>) => {
    const message = error.response?.data?.detail || error.message || 'An error occurred'
    return Promise.reject(new Error(message))
  }
)

// ============================================================================
// Projects API
// ============================================================================

export const projectsApi = {
  /**
   * List all projects
   */
  list: async (): Promise<Project[]> => {
    const { data } = await apiClient.get<Project[]>('/projects/')
    return data
  },

  /**
   * Create a new project
   */
  create: async (project: ProjectCreate): Promise<Project> => {
    const { data } = await apiClient.post<Project>('/projects/', project)
    return data
  },

  /**
   * Get project by ID
   */
  get: async (projectId: string): Promise<Project> => {
    const { data } = await apiClient.get<Project>(`/projects/${projectId}`)
    return data
  },

  /**
   * Get project statistics
   */
  getStats: async (projectId: string): Promise<ProjectStats> => {
    const { data } = await apiClient.get<ProjectStats>(`/projects/${projectId}/stats`)
    return data
  },

  /**
   * Delete a project
   */
  delete: async (projectId: string): Promise<void> => {
    await apiClient.delete(`/projects/${projectId}`)
  },
}

// ============================================================================
// Papers API
// ============================================================================

export const papersApi = {
  /**
   * List papers with optional filters
   */
  list: async (
    projectId: string,
    options?: {
      status?: string
      year?: number
      author?: string
      limit?: number
    }
  ): Promise<Paper[]> => {
    const params = new URLSearchParams({ project_id: projectId })
    if (options?.status) params.append('status', options.status)
    if (options?.year) params.append('year', options.year.toString())
    if (options?.author) params.append('author', options.author)
    if (options?.limit) params.append('limit', options.limit.toString())

    const { data } = await apiClient.get<Paper[]>(`/papers/?${params}`)
    return data
  },

  /**
   * Upload a new paper PDF
   */
  upload: async (projectId: string, file: File, onProgress?: (progress: number) => void): Promise<Paper> => {
    const formData = new FormData()
    formData.append('file', file)

    const { data } = await apiClient.post<Paper>(
      `/papers/upload?project_id=${projectId}`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100
            onProgress(progress)
          }
        },
      }
    )
    return data
  },

  /**
   * Analyze uploaded PDF without full processing
   * Returns confirmation requirement if >100 pages
   */
  analyzeUpload: async (projectId: string, file: File, onProgress?: (progress: number) => void): Promise<Paper> => {
    const formData = new FormData()
    formData.append('file', file)

    const { data } = await apiClient.post<Paper>(
      `/papers/analyze-upload?project_id=${projectId}`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100
            onProgress(progress)
          }
        },
      }
    )
    return data
  },

  /**
   * Upload multiple PDFs for batch analysis
   * Each file is analyzed quickly without full processing
   */
  uploadBatch: async (
    projectId: string,
    files: File[],
    onProgress?: (currentFile: number, totalFiles: number, fileName: string) => void
  ): Promise<Paper[]> => {
    const formData = new FormData()
    files.forEach(file => formData.append('files', file))

    const { data } = await apiClient.post<Paper[]>(
      `/papers/upload-batch?project_id=${projectId}`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            // Estimate which file we're on based on upload progress
            const overallProgress = progressEvent.loaded / progressEvent.total
            const currentFileIndex = Math.floor(overallProgress * files.length)
            const currentFile = Math.min(currentFileIndex + 1, files.length)
            const fileName = files[currentFileIndex]?.name || ''
            onProgress(currentFile, files.length, fileName)
          }
        },
      }
    )
    return data
  },

  /**
   * Complete processing after user confirmation for large PDFs
   */
  confirmProcessing: async (projectId: string, paperId: number, forceClean: boolean): Promise<Paper> => {
    const { data } = await apiClient.post<Paper>(
      `/papers/${paperId}/confirm-processing?project_id=${projectId}&force_clean=${forceClean}`
    )
    return data
  },

  /**
   * Get paper by ID
   */
  get: async (projectId: string, paperId: number): Promise<Paper> => {
    const { data } = await apiClient.get<Paper>(`/papers/${paperId}?project_id=${projectId}`)
    return data
  },

  /**
   * Download paper PDF
   */
  downloadPdf: async (projectId: string, paperId: number): Promise<Blob> => {
    const { data } = await apiClient.get<Blob>(
      `/papers/${paperId}/pdf?project_id=${projectId}`,
      { responseType: 'blob' }
    )
    return data
  },

  /**
   * Update paper metadata
   */
  update: async (projectId: string, paperId: number, updates: PaperUpdate): Promise<Paper> => {
    const { data} = await apiClient.put<Paper>(
      `/papers/${paperId}?project_id=${projectId}`,
      updates
    )
    return data
  },

  /**
   * Delete a paper
   */
  delete: async (projectId: string, paperId: number): Promise<void> => {
    await apiClient.delete(`/papers/${paperId}?project_id=${projectId}`)
  },

  /**
   * Reprocess a paper
   */
  reprocess: async (projectId: string, paperId: number): Promise<Paper> => {
    const { data } = await apiClient.post<Paper>(
      `/papers/${paperId}/reprocess?project_id=${projectId}`
    )
    return data
  },
}

// ============================================================================
// Chat API
// ============================================================================

export const chatApi = {
  /**
   * Create a new chat session
   */
  createSession: async (request: ChatSessionCreate): Promise<ChatSession> => {
    const { data } = await apiClient.post<ChatSession>(
      `/chat/sessions?project_id=${request.project_id}`,
      {
        project_id: request.project_id,
        title: request.title
      }
    )
    return data
  },

  /**
   * List all chat sessions
   */
  listSessions: async (projectId: string): Promise<ChatSession[]> => {
    const { data } = await apiClient.get<ChatSession[]>(`/chat/sessions?project_id=${projectId}`)
    return data
  },

  /**
   * Get session details
   */
  getSession: async (projectId: string, sessionId: string): Promise<ChatSession> => {
    const { data } = await apiClient.get<ChatSession>(
      `/chat/sessions/${sessionId}?project_id=${projectId}`
    )
    return data
  },

  /**
   * Send a message and get RAG response
   */
  sendMessage: async (
    projectId: string,
    sessionId: string,
    request: ChatMessageRequest
  ): Promise<ChatMessage> => {
    const { data } = await apiClient.post<ChatMessage>(
      `/chat/sessions/${sessionId}/messages?project_id=${projectId}`,
      request
    )
    return data
  },

  /**
   * Get conversation history
   */
  getMessages: async (projectId: string, sessionId: string): Promise<ChatMessage[]> => {
    const { data } = await apiClient.get<ChatMessage[]>(
      `/chat/sessions/${sessionId}/messages?project_id=${projectId}`
    )
    return data
  },

  /**
   * Delete a chat session
   */
  deleteSession: async (projectId: string, sessionId: string): Promise<void> => {
    await apiClient.delete(`/chat/sessions/${sessionId}?project_id=${projectId}`)
  },
}

// ============================================================================
// Recommendations API
// ============================================================================

export const recommendationsApi = {
  /**
   * Get paper recommendations based on query
   */
  query: async (request: RecommendationRequest): Promise<RecommendationResponse> => {
    const { data } = await apiClient.post<RecommendationResponse>('/recommendations/query', request)
    return data
  },
}

// ============================================================================
// Settings API
// ============================================================================

export const settingsApi = {
  /**
   * Get available embedding models
   */
  getEmbeddingModels: async (): Promise<EmbeddingModelInfo[]> => {
    const { data} = await apiClient.get<EmbeddingModelInfo[]>('/settings/embedding-models')
    return data
  },

  /**
   * Get LLM settings
   */
  getLLMSettings: async (): Promise<LLMSettings> => {
    const { data} = await apiClient.get<LLMSettings>('/settings/llm')
    return data
  },
}

// ============================================================================
// Documents API
// ============================================================================

export const documentsApi = {
  /**
   * List documents in project's documents folder
   */
  list: async (projectId: string): Promise<DocumentInfo[]> => {
    const { data } = await apiClient.get<DocumentInfo[]>(`/documents/${projectId}/list`)
    return data
  },

  /**
   * Analyze documents to extract research content
   */
  analyze: async (request: DocumentAnalysisRequest): Promise<DocumentAnalysis[]> => {
    const { data } = await apiClient.post<DocumentAnalysis[]>('/documents/analyze', request)
    return data
  },

  /**
   * Get recommendations based on document analysis
   */
  getRecommendations: async (request: DocumentRecommendationRequest): Promise<RecommendationResponse> => {
    const { data } = await apiClient.post<RecommendationResponse>('/documents/recommendations', request)
    return data
  },
}

// ============================================================================
// Queue API
// ============================================================================

export const queueApi = {
  /**
   * List pending papers awaiting user selection (status='analyzed')
   */
  listPending: async (projectId: string): Promise<Paper[]> => {
    const { data } = await apiClient.get<Paper[]>(`/queue/pending`, {
      params: { project_id: projectId }
    })
    return data
  },

  /**
   * List papers in processing queue (status='queued')
   */
  listQueued: async (projectId: string): Promise<Paper[]> => {
    const { data } = await apiClient.get<Paper[]>(`/queue/queued`, {
      params: { project_id: projectId }
    })
    return data
  },

  /**
   * Select papers for processing (analyzed → queued)
   */
  selectPapers: async (request: BatchSelectRequest): Promise<{ message: string }> => {
    const { data } = await apiClient.post<{ message: string }>('/queue/select', request)
    return data
  },

  /**
   * Reject papers from queue
   */
  rejectPapers: async (request: BatchSelectRequest): Promise<{ message: string }> => {
    const { data } = await apiClient.post<{ message: string }>('/queue/reject', request)
    return data
  },

  /**
   * Get current queue status
   */
  getStatus: async (projectId: string): Promise<QueueStatus> => {
    const { data } = await apiClient.get<QueueStatus>(`/queue/status`, {
      params: { project_id: projectId }
    })
    return data
  },
}

// Export default API object
export default {
  projects: projectsApi,
  papers: papersApi,
  chat: chatApi,
  recommendations: recommendationsApi,
  settings: settingsApi,
  documents: documentsApi,
  queue: queueApi,
}
