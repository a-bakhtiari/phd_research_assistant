/**
 * Pending Papers Queue Component
 *
 * Displays papers awaiting user selection for processing with batch selection UI.
 */

import { useState, useEffect } from 'react'
import { CheckCircle2, XCircle, Clock, FileText, Users, Calendar } from 'lucide-react'
import { queueApi } from '../services/api'
import type { Paper } from '../services/types'
import LoadingSpinner from './ui/LoadingSpinner'

interface PendingPapersQueueProps {
  projectId: string
  onRefresh?: () => void
}

export default function PendingPapersQueue({ projectId, onRefresh }: PendingPapersQueueProps) {
  const [pendingPapers, setPendingPapers] = useState<Paper[]>([])
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch pending papers on mount and when projectId changes
  useEffect(() => {
    fetchPendingPapers()
  }, [projectId])

  const fetchPendingPapers = async () => {
    try {
      setLoading(true)
      setError(null)
      const papers = await queueApi.listPending(projectId)
      setPendingPapers(papers)
      setSelectedIds(new Set()) // Clear selection on refresh
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load pending papers')
    } finally {
      setLoading(false)
    }
  }

  const handleToggleSelect = (paperId: number) => {
    setSelectedIds(prev => {
      const newSet = new Set(prev)
      if (newSet.has(paperId)) {
        newSet.delete(paperId)
      } else {
        newSet.add(paperId)
      }
      return newSet
    })
  }

  const handleSelectAll = () => {
    if (selectedIds.size === pendingPapers.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(pendingPapers.map(p => p.id)))
    }
  }

  const handleApprove = async () => {
    if (selectedIds.size === 0) return

    try {
      setSubmitting(true)
      setError(null)
      await queueApi.selectPapers({
        project_id: projectId,
        paper_ids: Array.from(selectedIds)
      })

      // Refresh the list
      await fetchPendingPapers()
      onRefresh?.()
    } catch (err: any) {
      console.error('Approve error:', err)
      const errorMsg = err?.response?.data?.detail || err?.message || 'Failed to approve papers'
      setError(typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg, null, 2))
    } finally {
      setSubmitting(false)
    }
  }

  const handleReject = async () => {
    if (selectedIds.size === 0) return

    if (!confirm(`Are you sure you want to reject ${selectedIds.size} paper(s)? They will be removed from the queue.`)) {
      return
    }

    try {
      setSubmitting(true)
      setError(null)
      await queueApi.rejectPapers({
        project_id: projectId,
        paper_ids: Array.from(selectedIds)
      })

      // Refresh the list
      await fetchPendingPapers()
      onRefresh?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reject papers')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-800 dark:text-red-300">{error}</p>
      </div>
    )
  }

  if (pendingPapers.length === 0) {
    return (
      <div className="text-center py-12">
        <FileText className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          No Pending Papers
        </h3>
        <p className="text-gray-500 dark:text-gray-400">
          Papers will appear here after quick analysis is complete
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header with bulk actions */}
      <div className="flex items-center justify-between bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={selectedIds.size === pendingPapers.length && pendingPapers.length > 0}
              onChange={handleSelectAll}
              className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Select All ({pendingPapers.length})
            </span>
          </label>

          {selectedIds.size > 0 && (
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {selectedIds.size} selected
            </span>
          )}
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleReject}
            disabled={selectedIds.size === 0 || submitting}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <XCircle size={16} />
            Reject
          </button>

          <button
            onClick={handleApprove}
            disabled={selectedIds.size === 0 || submitting}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 dark:bg-blue-500 rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <CheckCircle2 size={16} />
            {submitting ? 'Processing...' : 'Approve for Processing'}
          </button>
        </div>
      </div>

      {/* Papers list */}
      <div className="grid gap-4">
        {pendingPapers.map(paper => (
          <div
            key={paper.id}
            className={`bg-white dark:bg-gray-800 rounded-lg border-2 transition-all ${
              selectedIds.has(paper.id)
                ? 'border-blue-500 dark:border-blue-400 shadow-md'
                : 'border-gray-200 dark:border-gray-700 shadow-sm'
            }`}
          >
            <div className="p-4">
              <div className="flex items-start gap-3">
                {/* Checkbox */}
                <input
                  type="checkbox"
                  checked={selectedIds.has(paper.id)}
                  onChange={() => handleToggleSelect(paper.id)}
                  className="mt-1 w-5 h-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                />

                {/* Paper content */}
                <div className="flex-1 min-w-0">
                  {/* Title */}
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                    {paper.title || paper.filename}
                  </h3>

                  {/* Metadata */}
                  <div className="flex flex-wrap gap-3 mb-3 text-sm text-gray-600 dark:text-gray-400">
                    {paper.authors && paper.authors.length > 0 && (
                      <div className="flex items-center gap-1">
                        <Users size={14} />
                        <span className="line-clamp-1">{paper.authors.slice(0, 3).join(', ')}</span>
                        {paper.authors.length > 3 && <span> +{paper.authors.length - 3}</span>}
                      </div>
                    )}
                    {paper.year && (
                      <div className="flex items-center gap-1">
                        <Calendar size={14} />
                        <span>{paper.year}</span>
                      </div>
                    )}
                    {paper.page_count && (
                      <div className="flex items-center gap-1">
                        <FileText size={14} />
                        <span>{paper.page_count} pages</span>
                      </div>
                    )}
                    {paper.citation_count !== null && paper.citation_count !== undefined && (
                      <div className="flex items-center gap-1">
                        <span>{paper.citation_count} citations</span>
                      </div>
                    )}
                  </div>

                  {/* Abstract/Summary */}
                  {(paper.abstract || paper.summary) && (
                    <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-3 mb-2">
                      {paper.abstract || paper.summary}
                    </p>
                  )}

                  {/* Status badge */}
                  <div className="flex items-center gap-2">
                    <span className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300">
                      <Clock size={12} />
                      {paper.status}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      Added {new Date(paper.date_added).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
