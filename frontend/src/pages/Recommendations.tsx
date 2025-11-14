import { useState, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useProject } from '../contexts/ProjectContext'
import { recommendationsApi, documentsApi } from '../services/api'
import type { PaperRecommendation, DocumentInfo, DocumentAnalysis } from '../services/types'
import { Search, Lightbulb, ExternalLink, Star, Users, Calendar, FileText, CheckCircle2, XCircle, AlertCircle, RefreshCw } from 'lucide-react'
import LoadingSpinner from '../components/ui/LoadingSpinner'

export default function Recommendations() {
  const { currentProject } = useProject()
  const [query, setQuery] = useState('')
  const [recommendations, setRecommendations] = useState<PaperRecommendation[]>([])
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([])
  const [documentAnalyses, setDocumentAnalyses] = useState<DocumentAnalysis[]>([])
  const [showDocumentAnalysis, setShowDocumentAnalysis] = useState(false)
  const [lastQuery, setLastQuery] = useState('')

  // Load cached recommendations from localStorage on mount
  useEffect(() => {
    if (currentProject) {
      const cacheKey = `recommendations_${currentProject.id}`
      const cached = localStorage.getItem(cacheKey)
      console.log('Loading cache for project:', currentProject.id, 'Found:', !!cached)
      if (cached) {
        try {
          const { papers, query: savedQuery } = JSON.parse(cached)
          console.log('Loaded from cache:', papers.length, 'papers for query:', savedQuery)
          setRecommendations(papers)
          setLastQuery(savedQuery)
        } catch (e) {
          console.error('Failed to load cached recommendations:', e)
        }
      }
    }
  }, [currentProject])

  // Fetch documents
  const { data: documents = [], isLoading: documentsLoading } = useQuery({
    queryKey: ['documents', currentProject?.id],
    queryFn: () => documentsApi.list(currentProject!.id),
    enabled: !!currentProject,
  })

  // Query-based recommendations mutation
  const getRecommendationsMutation = useMutation({
    mutationFn: (searchQuery: string) =>
      recommendationsApi.query({
        project_id: currentProject!.id,
        query: searchQuery,
        max_recommendations: 10,
      }),
    onSuccess: (response, searchQuery) => {
      setRecommendations(response.recommendations)
      setLastQuery(searchQuery)
      setShowDocumentAnalysis(false)

      // Cache the results in localStorage (only if we got papers)
      if (currentProject && response.recommendations.length > 0) {
        const cacheKey = `recommendations_${currentProject.id}`
        localStorage.setItem(cacheKey, JSON.stringify({
          papers: response.recommendations,
          query: searchQuery,
          timestamp: new Date().toISOString()
        }))
        console.log('Cached', response.recommendations.length, 'papers for query:', searchQuery)
      } else if (response.recommendations.length === 0) {
        console.log('No papers found for query:', searchQuery, '- not caching empty results')
      }
    },
    onError: (error: any) => {
      console.error('Recommendation query error:', error)
    },
  })

  // Document analysis mutation
  const analyzeDocumentsMutation = useMutation({
    mutationFn: () =>
      documentsApi.analyze({
        project_id: currentProject!.id,
        document_paths: selectedDocuments,
      }),
    onSuccess: (analyses) => {
      setDocumentAnalyses(analyses)
      setShowDocumentAnalysis(true)
    },
    onError: (error: any) => {
      console.error('Document analysis error:', error)
    },
  })

  // Document-based recommendations mutation
  const getDocumentRecommendationsMutation = useMutation({
    mutationFn: () =>
      documentsApi.getRecommendations({
        project_id: currentProject!.id,
        document_paths: selectedDocuments,
        max_recommendations: 10,
      }),
    onSuccess: (response) => {
      setRecommendations(response.recommendations)
      setLastQuery('from documents')

      // Cache the results in localStorage (only if we got papers)
      if (currentProject && response.recommendations.length > 0) {
        const cacheKey = `recommendations_${currentProject.id}`
        localStorage.setItem(cacheKey, JSON.stringify({
          papers: response.recommendations,
          query: 'from documents',
          timestamp: new Date().toISOString()
        }))
        console.log('Cached', response.recommendations.length, 'papers from document analysis')
      }
    },
    onError: (error: any) => {
      console.error('Document recommendations error:', error)
    },
  })

  const handleSearch = () => {
    if (query.trim()) {
      getRecommendationsMutation.mutate(query)
    }
  }

  const handleAnalyzeDocuments = () => {
    if (selectedDocuments.length > 0) {
      analyzeDocumentsMutation.mutate()
    }
  }

  const handleGetRecommendationsFromDocuments = () => {
    if (selectedDocuments.length > 0) {
      getDocumentRecommendationsMutation.mutate()
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  const toggleDocument = (filePath: string) => {
    setSelectedDocuments((prev) =>
      prev.includes(filePath)
        ? prev.filter((p) => p !== filePath)
        : [...prev, filePath]
    )
  }

  // Removed auto-load to avoid timeout issues - user can search manually

  if (!currentProject) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-gray-600 dark:text-gray-400">Please select a project</p>
      </div>
    )
  }

  const isLoading =
    getRecommendationsMutation.isPending ||
    analyzeDocumentsMutation.isPending ||
    getDocumentRecommendationsMutation.isPending

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Paper Recommendations
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          AI-powered paper recommendations from Semantic Scholar
        </p>
      </div>

      {/* Simple Search Bar at Top */}
      <div className="mb-6">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Search for papers (e.g., 'machine learning', 'neural networks')..."
              className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={!query.trim() || isLoading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {getRecommendationsMutation.isPending ? (
              <>
                <LoadingSpinner size="sm" />
                Searching...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                Search
              </>
            )}
          </button>
        </div>
      </div>

      {/* Document Analysis Section */}
      {documents.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <FileText className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Document Analysis
            </h2>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              ({documents.length} documents available)
            </span>
          </div>

          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Select your research documents to get intelligent recommendations based on your work
          </p>

          {/* Document list */}
          <div className="space-y-2 mb-4 max-h-48 overflow-y-auto">
            {documents.map((doc) => (
              <label
                key={doc.file_path}
                className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors"
              >
                <input
                  type="checkbox"
                  checked={selectedDocuments.includes(doc.file_path)}
                  onChange={() => toggleDocument(doc.file_path)}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                />
                <FileText className="w-4 h-4 text-gray-400" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {doc.filename}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {(doc.size_bytes / 1024).toFixed(1)} KB • Modified{' '}
                    {new Date(doc.last_modified).toLocaleDateString()}
                  </p>
                </div>
              </label>
            ))}
          </div>

          {/* Action buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleAnalyzeDocuments}
              disabled={selectedDocuments.length === 0 || isLoading}
              className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 disabled:bg-gray-50 dark:disabled:bg-gray-800 disabled:cursor-not-allowed transition-colors text-sm flex items-center gap-2"
            >
              {analyzeDocumentsMutation.isPending ? (
                <>
                  <LoadingSpinner size="sm" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  Analyze Documents
                </>
              )}
            </button>
            <button
              onClick={handleGetRecommendationsFromDocuments}
              disabled={selectedDocuments.length === 0 || isLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm flex items-center gap-2"
            >
              {getDocumentRecommendationsMutation.isPending ? (
                <>
                  <LoadingSpinner size="sm" />
                  Generating...
                </>
              ) : (
                <>
                  <Lightbulb className="w-4 h-4" />
                  Get Recommendations
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Document Analysis Results */}
      {showDocumentAnalysis && documentAnalyses.length > 0 && (
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 p-6 mb-6">
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-200 mb-4">
            Document Analysis Results
          </h3>
          <div className="space-y-4">
            {documentAnalyses.map((analysis, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-lg p-4 space-y-3"
              >
                <div className="flex items-start justify-between">
                  <h4 className="font-medium text-gray-900 dark:text-white">{analysis.title}</h4>
                  <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded">
                    {analysis.document_type}
                  </span>
                </div>

                {analysis.research_questions.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Research Questions:
                    </p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      {analysis.research_questions.map((q, i) => (
                        <li key={i}>• {q}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {analysis.key_concepts.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Key Concepts:
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {analysis.key_concepts.map((concept, i) => (
                        <span
                          key={i}
                          className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded"
                        >
                          {concept}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {analysis.literature_gaps.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Literature Gaps:
                    </p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      {analysis.literature_gaps.map((gap, i) => (
                        <li key={i}>• {gap}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                  <span>{analysis.word_count.toLocaleString()} words</span>
                  <span>•</span>
                  <span>Confidence: {(analysis.confidence_score * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={handleGetRecommendationsFromDocuments}
            disabled={isLoading}
            className="mt-4 w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm flex items-center justify-center gap-2"
          >
            {getDocumentRecommendationsMutation.isPending ? (
              <>
                <LoadingSpinner size="sm" />
                Generating Recommendations...
              </>
            ) : (
              <>
                <Lightbulb className="w-4 h-4" />
                Generate Recommendations from Analysis
              </>
            )}
          </button>
        </div>
      )}


      {/* Error */}
      {(getRecommendationsMutation.isError ||
        analyzeDocumentsMutation.isError ||
        getDocumentRecommendationsMutation.isError) && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-8">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-600 dark:text-red-400 font-medium mb-1">
                An error occurred while fetching recommendations
              </p>
              <p className="text-sm text-red-500 dark:text-red-300">
                {getRecommendationsMutation.error?.message ||
                 analyzeDocumentsMutation.error?.message ||
                 getDocumentRecommendationsMutation.error?.message ||
                 'Please check your internet connection and try again.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {recommendations.length > 0 ? (
        <div>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Recommended Papers ({recommendations.length})
              </h2>
              {lastQuery && (
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Results for: "{lastQuery}"
                </p>
              )}
            </div>
            <button
              onClick={() => {
                if (lastQuery && lastQuery !== 'from documents') {
                  getRecommendationsMutation.mutate(lastQuery)
                }
              }}
              disabled={isLoading || !lastQuery || lastQuery === 'from documents'}
              className="flex items-center gap-2 px-4 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-gray-700 dark:text-gray-300"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
          </div>
          <div className="space-y-4">
            {recommendations.map((paper, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow"
              >
                {/* Title and Relevance */}
                <div className="flex items-start justify-between gap-4 mb-3">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex-1">
                    {paper.title}
                  </h3>
                  <div className="flex items-center gap-1 px-3 py-1 bg-blue-100 dark:bg-blue-900/30 rounded-full flex-shrink-0">
                    <Star className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    <span className="text-sm font-medium text-blue-800 dark:text-blue-300">
                      {(paper.relevance_score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Authors and Year */}
                <div className="flex flex-wrap gap-4 mb-3 text-sm text-gray-600 dark:text-gray-400">
                  {paper.authors.length > 0 && (
                    <div className="flex items-center gap-1">
                      <Users className="w-4 h-4" />
                      <span>
                        {paper.authors.slice(0, 3).join(', ')}
                        {paper.authors.length > 3 && ` +${paper.authors.length - 3}`}
                      </span>
                    </div>
                  )}
                  {paper.year && (
                    <div className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      <span>{paper.year}</span>
                    </div>
                  )}
                  {paper.citation_count !== null && (
                    <div className="flex items-center gap-1">
                      <span>{paper.citation_count} citations</span>
                    </div>
                  )}
                  {paper.venue && (
                    <div className="flex items-center gap-1">
                      <span className="italic">{paper.venue}</span>
                    </div>
                  )}
                </div>

                {/* Summary */}
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 leading-relaxed">
                  {paper.summary}
                </p>

                {/* Recommendation Reason */}
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 mb-4">
                  <p className="text-sm text-blue-900 dark:text-blue-200">
                    <span className="font-medium">Why recommended:</span> {paper.reason}
                  </p>
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  {paper.url && (
                    <a
                      href={paper.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
                    >
                      <ExternalLink className="w-4 h-4" />
                      View on Semantic Scholar
                    </a>
                  )}
                  {paper.doi && (
                    <a
                      href={`https://doi.org/${paper.doi}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm text-gray-700 dark:text-gray-300"
                    >
                      <ExternalLink className="w-4 h-4" />
                      View DOI
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : !isLoading && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
          <Lightbulb className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Discover Relevant Papers
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6 max-w-md mx-auto">
            Enter research topics or keywords above to get AI-powered paper recommendations from Semantic Scholar.
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mb-6">
            Note: Recommendations use AI analysis and may take 30-60 seconds
          </p>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            <p className="font-medium mb-2">Example queries:</p>
            <ul className="space-y-1">
              <li>"transformer models for natural language processing"</li>
              <li>"graph neural networks applications"</li>
              <li>"federated learning privacy"</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
