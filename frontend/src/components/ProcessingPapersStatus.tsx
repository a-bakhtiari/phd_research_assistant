/**
 * Processing Papers Status Component
 *
 * Displays papers currently being processed with real-time WebSocket updates.
 */

import { useState, useEffect, useRef } from 'react'
import { Loader2, FileText, CheckCircle, XCircle } from 'lucide-react'

interface ProcessingPaper {
  paper_id: number
  title: string
  progress: number
  step: string
  status: 'processing' | 'complete' | 'failed'
  error?: string
}

interface ProcessingPapersStatusProps {
  projectId: string
  onProcessingComplete?: () => void
}

export default function ProcessingPapersStatus({ projectId, onProcessingComplete }: ProcessingPapersStatusProps) {
  const [processingPapers, setProcessingPapers] = useState<Map<number, ProcessingPaper>>(new Map())
  const wsRef = useRef<WebSocket | null>(null)

  // WebSocket connection for real-time updates
  useEffect(() => {
    // Fetch initial processing state
    const fetchInitialState = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/v1/processing/${projectId}`)
        const data = await response.json()

        if (data.processing_papers && data.processing_papers.length > 0) {
          const initialState = new Map<number, ProcessingPaper>()
          data.processing_papers.forEach((paper: ProcessingPaper) => {
            initialState.set(paper.paper_id, paper)
          })
          setProcessingPapers(initialState)
          console.log(`Restored ${data.processing_papers.length} processing paper(s) from server state`)
        }
      } catch (error) {
        console.error('Failed to fetch initial processing state:', error)
      }
    }

    fetchInitialState()

    const wsUrl = `ws://localhost:8000/api/v1/ws/${projectId}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected for processing updates')
    }

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data)

      if (message.type === 'processing_progress') {
        // Update progress for this paper
        setProcessingPapers(prev => {
          const updated = new Map(prev)
          updated.set(message.paper_id, {
            paper_id: message.paper_id,
            title: message.title,
            progress: message.progress,
            step: message.step,
            status: 'processing'
          })
          return updated
        })
      } else if (message.type === 'processing_complete') {
        // Mark as complete, then remove after 2 seconds
        setProcessingPapers(prev => {
          const updated = new Map(prev)
          updated.set(message.paper_id, {
            paper_id: message.paper_id,
            title: message.title,
            progress: 100,
            step: 'Complete!',
            status: 'complete'
          })
          return updated
        })

        // Remove after 2 seconds and trigger refresh
        setTimeout(() => {
          setProcessingPapers(prev => {
            const updated = new Map(prev)
            updated.delete(message.paper_id)
            return updated
          })
          onProcessingComplete?.()
        }, 2000)
      } else if (message.type === 'processing_failed') {
        // Mark as failed
        setProcessingPapers(prev => {
          const updated = new Map(prev)
          updated.set(message.paper_id, {
            paper_id: message.paper_id,
            title: message.title || 'Unknown',
            progress: 0,
            step: 'Failed',
            status: 'failed',
            error: message.error
          })
          return updated
        })

        // Remove after 5 seconds
        setTimeout(() => {
          setProcessingPapers(prev => {
            const updated = new Map(prev)
            updated.delete(message.paper_id)
            return updated
          })
        }, 5000)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }

    // Cleanup on unmount
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
    }
  }, [projectId, onProcessingComplete])

  if (processingPapers.size === 0) {
    return null
  }

  const papers = Array.from(processingPapers.values())

  return (
    <div className="mb-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
      <div className="flex items-center gap-3 mb-3">
        <Loader2 className="h-5 w-5 text-blue-600 dark:text-blue-400 animate-spin" />
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100">
          Processing Papers ({papers.length})
        </h3>
      </div>

      <div className="space-y-3">
        {papers.map(paper => (
          <div
            key={paper.paper_id}
            className={`bg-white dark:bg-gray-800 rounded-lg p-4 border-2 transition-all ${
              paper.status === 'complete'
                ? 'border-green-500 dark:border-green-400'
                : paper.status === 'failed'
                ? 'border-red-500 dark:border-red-400'
                : 'border-blue-200 dark:border-blue-700'
            }`}
          >
            {/* Header with icon and title */}
            <div className="flex items-start gap-3 mb-3">
              {paper.status === 'processing' && (
                <Loader2 className="h-5 w-5 text-blue-600 dark:text-blue-400 animate-spin mt-0.5 flex-shrink-0" />
              )}
              {paper.status === 'complete' && (
                <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0" />
              )}
              {paper.status === 'failed' && (
                <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
              )}

              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white line-clamp-1">
                  {paper.title}
                </h4>
                <p className={`text-xs mt-1 ${
                  paper.status === 'failed' ? 'text-red-600 dark:text-red-400' : 'text-gray-600 dark:text-gray-400'
                }`}>
                  {paper.step}
                  {paper.error && ` - ${paper.error}`}
                </p>
              </div>

              <div className="text-sm font-semibold text-gray-700 dark:text-gray-300 flex-shrink-0">
                {paper.progress}%
              </div>
            </div>

            {/* Progress bar */}
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${
                  paper.status === 'complete'
                    ? 'bg-green-600 dark:bg-green-500'
                    : paper.status === 'failed'
                    ? 'bg-red-600 dark:bg-red-500'
                    : 'bg-blue-600 dark:bg-blue-500'
                }`}
                style={{ width: `${paper.progress}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <p className="text-xs text-blue-700 dark:text-blue-300 mt-3">
        Papers will automatically appear in the Unread tab when processing completes.
      </p>
    </div>
  )
}
