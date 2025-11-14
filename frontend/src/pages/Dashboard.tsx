import { useQuery } from '@tanstack/react-query'
import { useProject } from '../contexts/ProjectContext'
import { projectsApi, papersApi, chatApi } from '../services/api'
import { BookOpen, MessageSquare, Lightbulb, Upload, Plus } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import PaperCard from '../components/ui/PaperCard'

export default function Dashboard() {
  const { currentProject } = useProject()
  const navigate = useNavigate()

  // Fetch project stats
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['project-stats', currentProject?.id],
    queryFn: () => projectsApi.getStats(currentProject!.id),
    enabled: !!currentProject,
  })

  // Fetch recent papers
  const { data: recentPapers, isLoading: papersLoading } = useQuery({
    queryKey: ['recent-papers', currentProject?.id],
    queryFn: () => papersApi.list(currentProject!.id, { limit: 5 }),
    enabled: !!currentProject,
  })

  // Fetch chat sessions count
  const { data: chatSessions, isLoading: chatsLoading } = useQuery({
    queryKey: ['chat-sessions', currentProject?.id],
    queryFn: () => chatApi.listSessions(currentProject!.id),
    enabled: !!currentProject,
  })

  if (!currentProject) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            No Project Selected
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Please select or create a project to get started
          </p>
        </div>
      </div>
    )
  }

  const isLoading = statsLoading || papersLoading || chatsLoading

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Welcome to {currentProject.name}
        </p>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : (
        <>
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Papers Card */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Total Papers
                </h3>
                <BookOpen className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <p className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                {stats?.total_papers || 0}
              </p>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                <span className="text-green-600 dark:text-green-400">
                  {stats?.papers_read || 0} read
                </span>
                {' · '}
                <span className="text-yellow-600 dark:text-yellow-400">
                  {stats?.papers_reading || 0} reading
                </span>
              </div>
            </div>

            {/* Chat Sessions Card */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Chat Sessions
                </h3>
                <MessageSquare className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <p className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
                {chatSessions?.length || 0}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Active conversations
              </p>
            </div>

            {/* Total Citations Card */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Total Citations
                </h3>
                <Lightbulb className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <p className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">
                {stats?.total_citations || 0}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Across all papers
              </p>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="mb-8">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Quick Actions
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button
                onClick={() => navigate('/papers')}
                className="flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
              >
                <Upload className="w-5 h-5" />
                <span className="font-medium">Upload Paper</span>
              </button>
              <button
                onClick={() => navigate('/chat')}
                className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors"
              >
                <Plus className="w-5 h-5" />
                <span className="font-medium">New Chat</span>
              </button>
              <button
                onClick={() => navigate('/recommendations')}
                className="flex items-center gap-3 p-4 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
              >
                <Lightbulb className="w-5 h-5" />
                <span className="font-medium">Get Recommendations</span>
              </button>
            </div>
          </div>

          {/* Recent Papers */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Recent Papers
              </h2>
              <button
                onClick={() => navigate('/papers')}
                className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
              >
                View all
              </button>
            </div>
            {recentPapers && recentPapers.length > 0 ? (
              <div className="grid grid-cols-1 gap-4">
                {recentPapers.map((paper) => (
                  <PaperCard
                    key={paper.id}
                    paper={paper}
                    onClick={() => navigate('/papers')}
                  />
                ))}
              </div>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-8 text-center">
                <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  No papers yet. Upload your first paper to get started!
                </p>
                <button
                  onClick={() => navigate('/papers')}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Upload Paper
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
