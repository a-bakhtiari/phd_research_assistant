import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useProject } from '../contexts/ProjectContext'
import { papersApi, queueApi } from '../services/api'
import type { Paper, PaperUpdate } from '../services/types'
import { Upload, Search, Download, Trash2, RefreshCw, ExternalLink, FileText } from 'lucide-react'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import PaperCard from '../components/ui/PaperCard'
import Modal from '../components/ui/Modal'
import LargePdfConfirmModal from '../components/ui/LargePdfConfirmModal'
import PendingPapersQueue from '../components/PendingPapersQueue'
import ProcessingPapersStatus from '../components/ProcessingPapersStatus'

export default function Papers() {
  const { currentProject } = useProject()
  const queryClient = useQueryClient()

  // State
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [showDetailsModal, setShowDetailsModal] = useState(false)
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadFiles, setUploadFiles] = useState<File[]>([])
  const [uploadProgress, setUploadProgress] = useState(0)
  const [currentFileIndex, setCurrentFileIndex] = useState(0)
  const [currentFileName, setCurrentFileName] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [confirmingPaper, setConfirmingPaper] = useState<Paper | null>(null)

  // Fetch papers
  const { data: papers, isLoading } = useQuery({
    queryKey: ['papers', currentProject?.id, statusFilter],
    queryFn: () => papersApi.list(currentProject!.id, { status: statusFilter || undefined }),
    enabled: !!currentProject,
    staleTime: 5000, // Consider data fresh for 5 seconds
    refetchInterval: false, // Disable automatic polling
    refetchOnWindowFocus: false, // Prevent refetch on window focus during batch uploads
  })

  // Analyze mutation (replaces upload mutation)
  const analyzeMutation = useMutation({
    mutationFn: (file: File) =>
      papersApi.analyzeUpload(currentProject!.id, file, setUploadProgress),
    onSuccess: (paper) => {
      if (paper.needs_confirmation) {
        // Large PDF - show confirmation modal
        setConfirmingPaper(paper)
        setShowUploadModal(false)
      } else {
        // Small PDF - already processed
        queryClient.invalidateQueries({ queryKey: ['papers'] })
        setShowUploadModal(false)
        setUploadFile(null)
        setUploadProgress(0)
      }
    },
  })

  // Batch upload mutation
  const batchUploadMutation = useMutation({
    mutationFn: (files: File[]) =>
      papersApi.uploadBatch(currentProject!.id, files, (current, total, fileName) => {
        setCurrentFileIndex(current)
        setCurrentFileName(fileName)
        setUploadProgress((current / total) * 100)
      }),
    onSuccess: (papers) => {
      // All papers analyzed and added to queue
      setShowUploadModal(false)
      setUploadFiles([])
      setUploadFile(null)
      setUploadProgress(0)
      setCurrentFileIndex(0)
      setCurrentFileName('')

      // Switch to pending tab first
      setStatusFilter('pending')

      // Invalidate after state updates complete
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['papers'] })
        // Show success message after invalidation
        alert(`Successfully analyzed ${papers.length} paper(s)! Check the Pending tab to review and process them.`)
      }, 100)
    },
    onError: (error) => {
      console.error('Batch upload failed:', error)
      // Don't clear files so user can retry
    },
  })

  // Confirm processing mutation
  const confirmMutation = useMutation({
    mutationFn: ({ paperId, forceClean }: { paperId: number; forceClean: boolean }) =>
      papersApi.confirmProcessing(currentProject!.id, paperId, forceClean),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['papers'] })
      setConfirmingPaper(null)
      setUploadFile(null)
      setUploadProgress(0)
    },
  })

  // Cancel confirmation mutation
  const cancelConfirmMutation = useMutation({
    mutationFn: (paperId: number) =>
      papersApi.delete(currentProject!.id, paperId),
    onSuccess: () => {
      setConfirmingPaper(null)
      setUploadFile(null)
      setUploadProgress(0)
    },
  })

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: ({ paperId, updates }: { paperId: number; updates: PaperUpdate }) =>
      papersApi.update(currentProject!.id, paperId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['papers'] })
      setShowDetailsModal(false)
    },
  })

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (paperId: number) => papersApi.delete(currentProject!.id, paperId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['papers'] })
      setShowDetailsModal(false)
    },
  })

  // Reprocess mutation
  const reprocessMutation = useMutation({
    mutationFn: (paperId: number) => papersApi.reprocess(currentProject!.id, paperId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['papers'] })
    },
  })

  // Filter papers by search query
  const filteredPapers = papers?.filter(paper =>
    paper.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    paper.authors.some(author => author.toLowerCase().includes(searchQuery.toLowerCase()))
  ) || []

  // Handle upload
  const handleUpload = () => {
    // If multiple files selected, use batch upload
    if (uploadFiles.length > 1) {
      batchUploadMutation.mutate(uploadFiles)
    } else if (uploadFile) {
      // Single file - use existing analyze flow
      analyzeMutation.mutate(uploadFile)
    }
  }

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 1) {
      // Multiple files selected
      setUploadFiles(files)
      setUploadFile(null)
    } else if (files.length === 1) {
      // Single file selected
      setUploadFile(files[0])
      setUploadFiles([])
    }
  }

  // Remove file from batch
  const removeFile = (index: number) => {
    setUploadFiles(prev => prev.filter((_, i) => i !== index))
  }

  // Handle download PDF
  const handleDownloadPdf = async (paper: Paper) => {
    try {
      const blob = await papersApi.downloadPdf(currentProject!.id, paper.id)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = paper.filename
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Download failed:', error)
      alert('Failed to download PDF')
    }
  }

  if (!currentProject) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-gray-600 dark:text-gray-400">Please select a project</p>
      </div>
    )
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Papers Library
        </h1>
        <button
          onClick={() => setShowUploadModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <Upload className="w-4 h-4" />
          Upload Paper
        </button>
      </div>

      {/* Processing Papers Status */}
      <ProcessingPapersStatus
        projectId={currentProject.id}
        onProcessingComplete={() => {
          queryClient.invalidateQueries({ queryKey: ['papers'] })
        }}
      />

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 mb-6">
        <div className="flex flex-col md:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search by title or author..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
            />
          </div>

        </div>

        {/* Status Tabs */}
        <div className="flex gap-2 border-b border-gray-200 dark:border-gray-700">
          {[
            { label: 'All Papers', value: '' },
            { label: 'Pending', value: 'pending' },
            { label: 'Unread', value: 'unread' },
            { label: 'Reading', value: 'reading' },
            { label: 'Read', value: 'read' },
          ].map((tab) => (
            <button
              key={tab.value}
              onClick={() => setStatusFilter(tab.value)}
              className={`px-6 py-3 font-medium text-sm transition-colors border-b-2 ${
                statusFilter === tab.value
                  ? 'border-blue-600 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Papers Grid or Pending Queue */}
      {statusFilter === 'pending' ? (
        <PendingPapersQueue
          projectId={currentProject.id}
          onRefresh={() => queryClient.invalidateQueries({ queryKey: ['papers'] })}
        />
      ) : isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : filteredPapers.length > 0 ? (
        <div className="grid grid-cols-1 gap-4">
          {filteredPapers.map((paper) => (
            <PaperCard
              key={paper.id}
              paper={paper}
              onClick={() => {
                setSelectedPaper(paper)
                setShowDetailsModal(true)
              }}
            />
          ))}
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
          <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            {searchQuery || statusFilter ? 'No papers match your filters' : 'No papers yet'}
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            {searchQuery || statusFilter
              ? 'Try adjusting your search or filters'
              : 'Upload your first paper to get started'}
          </p>
          {!searchQuery && !statusFilter && (
            <button
              onClick={() => setShowUploadModal(true)}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Upload Paper
            </button>
          )}
        </div>
      )}

      {/* Upload Modal */}
      <Modal
        isOpen={showUploadModal}
        onClose={() => {
          setShowUploadModal(false)
          setUploadFile(null)
          setUploadFiles([])
          setUploadProgress(0)
          setCurrentFileIndex(0)
          setCurrentFileName('')
        }}
        title={uploadFiles.length > 1 ? `Upload ${uploadFiles.length} Papers` : "Upload Paper"}
        size="md"
      >
        <div className="space-y-4">
          {/* File Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Select PDF Files (one or more)
            </label>
            <input
              type="file"
              accept=".pdf"
              multiple
              onChange={handleFileSelect}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Selected Files List (for batch upload) */}
          {uploadFiles.length > 0 && (
            <div className="border border-gray-300 dark:border-gray-600 rounded-lg p-3 max-h-48 overflow-y-auto">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Selected Files ({uploadFiles.length})
              </p>
              <ul className="space-y-2">
                {uploadFiles.map((file, index) => (
                  <li key={index} className="flex items-center justify-between text-sm">
                    <span className="text-gray-900 dark:text-white truncate flex-1">{file.name}</span>
                    <button
                      onClick={() => removeFile(index)}
                      className="ml-2 text-red-600 hover:text-red-800 dark:text-red-400"
                      disabled={batchUploadMutation.isPending}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Single File Display */}
          {uploadFile && !uploadFiles.length && (
            <div className="text-sm text-gray-700 dark:text-gray-300">
              <strong>Selected:</strong> {uploadFile.name}
            </div>
          )}

          {/* Upload Progress - Single File */}
          {analyzeMutation.isPending && (
            <div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                Analyzing... {uploadProgress.toFixed(0)}%
              </p>
            </div>
          )}

          {/* Upload Progress - Batch */}
          {batchUploadMutation.isPending && (
            <div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                Analyzing {currentFileIndex} of {uploadFiles.length} files...
              </p>
              {currentFileName && (
                <p className="text-xs text-gray-500 dark:text-gray-500 mt-1 truncate">
                  Current: {currentFileName}
                </p>
              )}
            </div>
          )}

          {/* Error */}
          {(analyzeMutation.isError || batchUploadMutation.isError) && (
            <p className="text-sm text-red-600 dark:text-red-400">
              Upload failed. Please try again.
            </p>
          )}

          {/* Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleUpload}
              disabled={(!uploadFile && uploadFiles.length === 0) || analyzeMutation.isPending || batchUploadMutation.isPending}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {batchUploadMutation.isPending
                ? `Analyzing ${currentFileIndex}/${uploadFiles.length}...`
                : analyzeMutation.isPending
                ? 'Analyzing...'
                : uploadFiles.length > 1
                ? `Analyze ${uploadFiles.length} PDFs`
                : 'Analyze PDF'}
            </button>
            <button
              onClick={() => {
                setShowUploadModal(false)
                setUploadFile(null)
                setUploadFiles([])
                setUploadProgress(0)
                setCurrentFileIndex(0)
                setCurrentFileName('')
              }}
              disabled={analyzeMutation.isPending || batchUploadMutation.isPending}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </Modal>

      {/* Paper Details Modal */}
      {selectedPaper && (
        <Modal
          isOpen={showDetailsModal}
          onClose={() => setShowDetailsModal(false)}
          title={selectedPaper.title}
          size="xl"
        >
          <div className="space-y-6">
            {/* Metadata */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Authors
                </label>
                <p className="text-gray-900 dark:text-white">
                  {selectedPaper.authors.join(', ') || 'Unknown'}
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Year
                </label>
                <p className="text-gray-900 dark:text-white">
                  {selectedPaper.year || 'Unknown'}
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Citations
                </label>
                <p className="text-gray-900 dark:text-white">
                  {selectedPaper.citation_count || 0}
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Status
                </label>
                <select
                  value={selectedPaper.status}
                  onChange={(e) =>
                    updateMutation.mutate({
                      paperId: selectedPaper.id,
                      updates: { status: e.target.value }
                    })
                  }
                  className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="unread">Unread</option>
                  <option value="reading">Reading</option>
                  <option value="read">Read</option>
                </select>
              </div>
            </div>

            {/* Abstract */}
            {selectedPaper.abstract && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Abstract
                </label>
                <p className="text-gray-900 dark:text-white text-sm leading-relaxed">
                  {selectedPaper.abstract}
                </p>
              </div>
            )}

            {/* Summary */}
            {selectedPaper.summary && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  AI Summary
                </label>
                <p className="text-gray-900 dark:text-white text-sm leading-relaxed">
                  {selectedPaper.summary}
                </p>
              </div>
            )}

            {/* Tags */}
            {selectedPaper.tags && selectedPaper.tags.length > 0 && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Tags
                </label>
                <div className="flex flex-wrap gap-2">
                  {selectedPaper.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-3 py-1 text-sm rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => handleDownloadPdf(selectedPaper)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Download className="w-4 h-4" />
                Download PDF
              </button>
              {selectedPaper.summary && (
                <a
                  href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'}/papers/${selectedPaper.id}/summary-pdf?project_id=${currentProject?.id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  <FileText className="w-4 h-4" />
                  Open Detailed Summary
                </a>
              )}
              <button
                onClick={() => reprocessMutation.mutate(selectedPaper.id)}
                disabled={reprocessMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <RefreshCw className={`w-4 h-4 ${reprocessMutation.isPending ? 'animate-spin' : ''}`} />
                Reprocess
              </button>
              {selectedPaper.semantic_scholar_url && (
                <a
                  href={selectedPaper.semantic_scholar_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  Semantic Scholar
                </a>
              )}
              <button
                onClick={() => {
                  if (confirm('Are you sure you want to delete this paper?')) {
                    deleteMutation.mutate(selectedPaper.id)
                  }
                }}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors ml-auto"
              >
                <Trash2 className="w-4 h-4" />
                Delete
              </button>
            </div>
          </div>
        </Modal>
      )}

      {/* Large PDF Confirmation Modal */}
      {confirmingPaper && (
        <LargePdfConfirmModal
          paper={confirmingPaper}
          onConfirm={(forceClean) => {
            confirmMutation.mutate({
              paperId: confirmingPaper.id,
              forceClean
            })
          }}
          onCancel={() => {
            cancelConfirmMutation.mutate(confirmingPaper.id)
          }}
          isProcessing={confirmMutation.isPending}
        />
      )}
    </div>
  )
}
