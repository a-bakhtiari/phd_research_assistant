/**
 * Large PDF Confirmation Modal
 *
 * Shows confirmation dialog when uploading large PDFs (>100 pages)
 * Allows user to choose: clean anyway, skip cleaning, or cancel
 */

import { AlertTriangle, CheckCircle, XCircle, FileText, Clock, DollarSign } from 'lucide-react'
import type { Paper } from '../../services/types'

interface LargePdfConfirmModalProps {
  paper: Paper
  onConfirm: (forceClean: boolean) => void
  onCancel: () => void
  isProcessing: boolean
}

export default function LargePdfConfirmModal({
  paper,
  onConfirm,
  onCancel,
  isProcessing
}: LargePdfConfirmModalProps) {
  if (!paper.confirmation_metadata) return null

  const meta = paper.confirmation_metadata

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={!isProcessing ? onCancel : undefined}
      />

      {/* Modal */}
      <div className="relative bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-xl w-full mx-4">
        {/* Header */}
        <div className="flex items-center gap-3 p-6 border-b border-gray-200 dark:border-gray-700">
          <AlertTriangle className="text-yellow-500" size={24} />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Large PDF Detected
          </h2>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* PDF Info */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 space-y-2">
            <div className="flex items-center gap-2 text-gray-900 dark:text-white">
              <FileText size={18} />
              <span className="font-medium truncate" title={paper.original_filename}>
                {paper.original_filename}
              </span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <p>Original pages: {meta.page_count}</p>
              {meta.references_removed && (
                <p className="text-green-600 dark:text-green-400">
                  ✓ References removed: {meta.page_count - meta.page_count_after_refs} pages
                </p>
              )}
              <p className="font-medium">Processing pages: {meta.page_count_after_refs}</p>
            </div>
          </div>

          {/* Estimates */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-start gap-3">
              <Clock className="text-blue-500 mt-0.5" size={20} />
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Estimated Time</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  ~{meta.estimated_time_minutes} min
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <DollarSign className="text-green-500 mt-0.5" size={20} />
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Estimated Cost</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  ~${meta.estimated_cost_usd}
                </p>
              </div>
            </div>
          </div>

          {/* Info Message */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              This PDF exceeds the {meta.threshold}-page threshold.
            </p>
            <p className="text-sm text-gray-700 dark:text-gray-300 font-medium mb-2">
              Benefits of cleaning:
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1 list-disc list-inside">
              <li>Removes headers, footers, page numbers</li>
              <li>Removes tables and figures</li>
              <li>Better quality for AI chat and summaries</li>
            </ul>
          </div>

          {/* Processing Indicator */}
          {isProcessing && (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
              <p className="text-sm text-yellow-800 dark:text-yellow-300 font-medium">
                Processing paper... This may take several minutes.
              </p>
            </div>
          )}

          {/* Buttons */}
          <div className="space-y-3">
            <button
              onClick={() => onConfirm(true)}
              disabled={isProcessing}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <CheckCircle size={20} />
              Clean it anyway
            </button>

            <button
              onClick={() => onConfirm(false)}
              disabled={isProcessing}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <FileText size={20} />
              Skip cleaning, add to library
            </button>

            <button
              onClick={onCancel}
              disabled={isProcessing}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 disabled:bg-gray-100 disabled:cursor-not-allowed transition-colors"
            >
              <XCircle size={20} />
              Cancel upload
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
