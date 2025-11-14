/**
 * Paper Card Component
 */

import { FileText, Users, Calendar, ExternalLink } from 'lucide-react'
import type { Paper } from '../../services/types'

interface PaperCardProps {
  paper: Paper
  onClick?: () => void
}

export default function PaperCard({ paper, onClick }: PaperCardProps) {
  return (
    <div
      className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 hover:shadow-md transition-shadow cursor-pointer"
      onClick={onClick}
    >
      {/* Title */}
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2 line-clamp-2">
        {paper.title}
      </h3>

      {/* Metadata */}
      <div className="flex flex-wrap gap-3 mb-3 text-sm text-gray-600 dark:text-gray-400">
        {paper.authors.length > 0 && (
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
        {paper.citation_count !== null && (
          <div className="flex items-center gap-1">
            <FileText size={14} />
            <span>{paper.citation_count} citations</span>
          </div>
        )}
      </div>

      {/* Abstract/Summary */}
      {(paper.abstract || paper.summary) && (
        <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-2 mb-3">
          {paper.abstract || paper.summary}
        </p>
      )}

      {/* Tags */}
      {paper.tags && paper.tags.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {paper.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="px-2 py-1 text-xs rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300"
            >
              {tag}
            </span>
          ))}
          {paper.tags.length > 3 && (
            <span className="px-2 py-1 text-xs rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400">
              +{paper.tags.length - 3} more
            </span>
          )}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-500 dark:text-gray-400">
          Added {new Date(paper.date_added).toLocaleDateString()}
        </span>
        {paper.semantic_scholar_url && (
          <a
            href={paper.semantic_scholar_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline"
            onClick={(e) => e.stopPropagation()}
          >
            View on Semantic Scholar
            <ExternalLink size={14} />
          </a>
        )}
      </div>

      {/* Status Badge */}
      <div className="mt-2">
        <span className={`px-2 py-1 text-xs rounded-full ${
          paper.status === 'read'
            ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
            : paper.status === 'reading'
            ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300'
            : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
        }`}>
          {paper.status}
        </span>
      </div>
    </div>
  )
}
