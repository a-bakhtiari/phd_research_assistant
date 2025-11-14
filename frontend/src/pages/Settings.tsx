import { useQuery } from '@tanstack/react-query'
import { settingsApi } from '../services/api'
import { Cpu, Database, Zap, Info } from 'lucide-react'
import LoadingSpinner from '../components/ui/LoadingSpinner'

export default function Settings() {
  // Fetch embedding models
  const { data: embeddingModels, isLoading: modelsLoading } = useQuery({
    queryKey: ['embedding-models'],
    queryFn: settingsApi.getEmbeddingModels,
  })

  // Fetch LLM settings
  const { data: llmSettings, isLoading: llmLoading } = useQuery({
    queryKey: ['llm-settings'],
    queryFn: settingsApi.getLLMSettings,
  })

  const isLoading = modelsLoading || llmLoading

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Settings
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          System configuration and model information
        </p>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : (
        <div className="space-y-6">
          {/* LLM Settings */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                <Zap className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                  Language Model
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  LLM configuration for chat and summarization
                </p>
              </div>
            </div>

            {llmSettings && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Provider
                  </label>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white mt-1">
                    {llmSettings.provider}
                  </p>
                </div>
                <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Model
                  </label>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white mt-1">
                    {llmSettings.model}
                  </p>
                </div>
                <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Temperature
                  </label>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white mt-1">
                    {llmSettings.temperature}
                  </p>
                </div>
                <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Max Tokens
                  </label>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white mt-1">
                    {llmSettings.max_tokens}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Embedding Models */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                <Database className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                  Embedding Models
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Available models for document embedding
                </p>
              </div>
            </div>

            {embeddingModels && embeddingModels.length > 0 ? (
              <div className="space-y-4">
                {embeddingModels.map((model, idx) => (
                  <div
                    key={idx}
                    className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                          {model.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {model.description}
                        </p>
                      </div>
                      {model.is_default && (
                        <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 text-xs font-medium rounded-full">
                          Default
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div>
                        <label className="text-xs font-medium text-gray-700 dark:text-gray-400">
                          Dimensions
                        </label>
                        <p className="text-sm font-semibold text-gray-900 dark:text-white">
                          {model.dimensions}
                        </p>
                      </div>
                      <div>
                        <label className="text-xs font-medium text-gray-700 dark:text-gray-400">
                          Max Tokens
                        </label>
                        <p className="text-sm font-semibold text-gray-900 dark:text-white">
                          {model.max_tokens}
                        </p>
                      </div>
                      <div>
                        <label className="text-xs font-medium text-gray-700 dark:text-gray-400">
                          Provider
                        </label>
                        <p className="text-sm font-semibold text-gray-900 dark:text-white">
                          {model.provider}
                        </p>
                      </div>
                      <div>
                        <label className="text-xs font-medium text-gray-700 dark:text-gray-400">
                          Model ID
                        </label>
                        <p className="text-sm font-mono text-gray-900 dark:text-white truncate">
                          {model.model_id}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-600 dark:text-gray-400 text-center py-4">
                No embedding models configured
              </p>
            )}
          </div>

          {/* System Info */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
                <Info className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                  System Information
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Application details and version
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Frontend Version
                </label>
                <p className="text-lg font-semibold text-gray-900 dark:text-white mt-1">
                  2.0.0
                </p>
              </div>
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  API URL
                </label>
                <p className="text-sm font-mono text-gray-900 dark:text-white mt-1 truncate">
                  {import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'}
                </p>
              </div>
            </div>
          </div>

          {/* Info Notice */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <div className="flex gap-3">
              <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-sm font-medium text-blue-900 dark:text-blue-200 mb-1">
                  Configuration Note
                </h3>
                <p className="text-sm text-blue-800 dark:text-blue-300">
                  API keys and sensitive settings are managed in the backend environment variables.
                  Contact your system administrator to modify these settings.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
