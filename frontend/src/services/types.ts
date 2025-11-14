/**
 * TypeScript interfaces matching backend Pydantic models
 */

// ============================================================================
// Project Types
// ============================================================================

export interface Project {
  id: string
  name: string
  description: string | null
  path: string
  created_at: string | null
  paper_count: number
}

export interface ProjectCreate {
  name: string
  description?: string
}

export interface ProjectStats {
  total_papers: number
  papers_by_status: Record<string, number>
  papers_by_year: Record<string, number>
  total_chat_sessions: number
  total_vector_embeddings: number
}

// ============================================================================
// Paper Types
// ============================================================================

export interface Paper {
  id: number
  filename: string
  original_filename: string
  title: string
  authors: string[]
  year: number | null
  summary: string | null
  abstract: string | null
  status: string
  tags: string[]
  date_added: string
  file_path: string
  doi: string | null
  journal: string | null
  page_count: number | null
  citation_count: number | null
  venue: string | null
  semantic_scholar_url: string | null

  // Large PDF handling fields
  needs_confirmation?: boolean
  confirmation_metadata?: {
    page_count: number
    page_count_after_refs: number
    estimated_time_minutes: number
    estimated_cost_usd: number
    references_removed: boolean
    threshold: number
  }
}

export interface PaperUpdate {
  status?: string
  tags?: string[]
  summary?: string
}

// ============================================================================
// Chat Types
// ============================================================================

export interface ChatSession {
  session_id: string
  project_id: string
  title: string
  created_at: string
  last_updated: string
  message_count: number
  total_tokens_used: number
}

export interface ChatSessionCreate {
  project_id: string
  title?: string
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface ChatMessage {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: SourceCitation[]
  tokens_used?: TokenUsage
  confidence_score?: number
}

export interface ChatMessageRequest {
  message: string
  max_sources?: number
  use_rag?: boolean
}

export interface SourceCitation {
  paper_id: number | null
  paper_title: string
  authors: string[]
  year: number | null
  content: string
  page_number: number | null
  similarity_score: number
}

// ============================================================================
// Recommendation Types
// ============================================================================

export interface RecommendationRequest {
  project_id: string
  query: string
  max_recommendations?: number
}

export interface PaperRecommendation {
  title: string
  authors: string[]
  year: number
  summary: string
  relevance_score: number
  reason: string
  doi: string | null
  url: string | null
  semantic_scholar_id: string | null
  citation_count: number | null
  venue: string | null
}

export interface RecommendationResponse {
  query: string
  recommendations: PaperRecommendation[]
  gap_analysis: string | null
  timestamp: string
}

// ============================================================================
// Settings Types
// ============================================================================

export interface EmbeddingModelInfo {
  name: string
  description: string
  dimension: number
  is_current: boolean
}

export interface LLMSettings {
  default_provider: string
  available_providers: string[]
  models: Record<string, string>
}

// ============================================================================
// Document Types
// ============================================================================

export interface DocumentInfo {
  filename: string
  file_path: string
  size_bytes: number
  last_modified: string
}

export interface DocumentAnalysis {
  title: string
  word_count: number
  research_questions: string[]
  key_concepts: string[]
  literature_gaps: string[]
  citations_present: string[]
  citations_needed: string[]
  document_type: string
  confidence_score: number
}

export interface DocumentAnalysisRequest {
  project_id: string
  document_paths: string[]
}

export interface DocumentRecommendationRequest {
  project_id: string
  document_paths: string[]
  max_recommendations?: number
}

// ============================================================================
// Queue Types
// ============================================================================

export interface QueueStatus {
  pending_count: number
  queued_count: number
  current_processing: {
    id: number
    title: string
    filename: string
  } | null
}

export interface BatchSelectRequest {
  project_id: string  // Project identifier like 'test_proj'
  paper_ids: number[]
}

// Paper status type union for better type safety
export type PaperStatus =
  | 'detected'              // File detected, awaiting quick analysis
  | 'analyzed'              // Quick analysis complete, awaiting user selection
  | 'queued'                // User selected for processing, in queue
  | 'processing'            // Currently being processed
  | 'pending_confirmation'  // Large PDF awaiting user confirmation
  | 'unread'                // Processed and ready to read
  | 'reading'               // Currently reading
  | 'read'                  // Finished reading

// ============================================================================
// API Response Types
// ============================================================================

export interface ApiError {
  detail: string
}
