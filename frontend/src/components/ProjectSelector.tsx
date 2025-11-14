/**
 * Project Selector Component
 * Allows users to select and switch between projects
 */

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronDown, Plus, FolderOpen } from 'lucide-react'
import { projectsApi } from '../services/api'
import { useProject } from '../contexts/ProjectContext'
import Modal from './ui/Modal'
import LoadingSpinner from './ui/LoadingSpinner'

export default function ProjectSelector() {
  const { currentProject, setCurrentProject, setProjects } = useProject()
  const [isOpen, setIsOpen] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectDescription, setNewProjectDescription] = useState('')

  // Fetch projects
  const { data: projectList, isLoading, refetch } = useQuery({
    queryKey: ['projects'],
    queryFn: async () => {
      const projects = await projectsApi.list()
      setProjects(projects)
      // Auto-select first project if none selected
      if (!currentProject && projects.length > 0) {
        setCurrentProject(projects[0])
      }
      return projects
    },
  })

  // Create new project
  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newProjectName.trim()) return

    try {
      const project = await projectsApi.create({
        name: newProjectName,
        description: newProjectDescription || undefined,
      })
      setCurrentProject(project)
      setShowCreateModal(false)
      setNewProjectName('')
      setNewProjectDescription('')
      refetch()
    } catch (error: any) {
      console.error('Error creating project:', error)
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to create project'
      alert(errorMessage)
    }
  }

  if (isLoading) {
    return <LoadingSpinner size="sm" />
  }

  return (
    <>
      {/* Selector Dropdown */}
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors min-w-[200px]"
        >
          <FolderOpen size={18} className="text-gray-600 dark:text-gray-400" />
          <span className="flex-1 text-left text-gray-900 dark:text-white truncate">
            {currentProject?.name || 'Select Project'}
          </span>
          <ChevronDown size={18} className="text-gray-600 dark:text-gray-400" />
        </button>

        {/* Dropdown Menu */}
        {isOpen && (
          <div className="absolute top-full left-0 mt-2 w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
            {projectList && projectList.length > 0 ? (
              <>
                {projectList.map((project) => (
                  <button
                    key={project.id}
                    onClick={() => {
                      setCurrentProject(project)
                      setIsOpen(false)
                    }}
                    className={`w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors border-b border-gray-200 dark:border-gray-700 last:border-b-0 ${
                      currentProject?.id === project.id
                        ? 'bg-blue-50 dark:bg-blue-900/20'
                        : ''
                    }`}
                  >
                    <div className="font-medium text-gray-900 dark:text-white">
                      {project.name}
                    </div>
                    {project.description && (
                      <div className="text-sm text-gray-600 dark:text-gray-400 truncate">
                        {project.description}
                      </div>
                    )}
                    <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                      {project.paper_count} papers
                    </div>
                  </button>
                ))}
                <button
                  onClick={() => {
                    setShowCreateModal(true)
                    setIsOpen(false)
                  }}
                  className="w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2 text-blue-600 dark:text-blue-400 font-medium"
                >
                  <Plus size={18} />
                  Create New Project
                </button>
              </>
            ) : (
              <button
                onClick={() => {
                  setShowCreateModal(true)
                  setIsOpen(false)
                }}
                className="w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2 text-blue-600 dark:text-blue-400"
              >
                <Plus size={18} />
                Create Your First Project
              </button>
            )}
          </div>
        )}
      </div>

      {/* Create Project Modal */}
      <Modal
        isOpen={showCreateModal}
        onClose={() => {
          setShowCreateModal(false)
          setNewProjectName('')
          setNewProjectDescription('')
        }}
        title="Create New Project"
        size="sm"
      >
        <form onSubmit={handleCreateProject} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Project Name *
            </label>
            <input
              type="text"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="My Research Project"
              required
              autoFocus
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Description (optional)
            </label>
            <textarea
              value={newProjectDescription}
              onChange={(e) => setNewProjectDescription(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Brief description of your research..."
              rows={3}
            />
          </div>

          <div className="flex gap-3 justify-end">
            <button
              type="button"
              onClick={() => {
                setShowCreateModal(false)
                setNewProjectName('')
                setNewProjectDescription('')
              }}
              className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Create Project
            </button>
          </div>
        </form>
      </Modal>
    </>
  )
}
