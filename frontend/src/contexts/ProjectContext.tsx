/**
 * Project Context - Manages the currently selected project across the app
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import type { Project } from '../services/types'
import { projectsApi } from '../services/api'

interface ProjectContextType {
  currentProject: Project | null
  setCurrentProject: (project: Project | null) => void
  projects: Project[]
  setProjects: (projects: Project[]) => void
}

const ProjectContext = createContext<ProjectContextType | undefined>(undefined)

export function ProjectProvider({ children }: { children: ReactNode }) {
  const [currentProject, setCurrentProjectState] = useState<Project | null>(null)
  const [projects, setProjects] = useState<Project[]>([])

  // Load saved project from localStorage on mount
  useEffect(() => {
    const savedProjectId = localStorage.getItem('currentProjectId')
    if (savedProjectId && projects.length > 0) {
      const project = projects.find(p => p.id === savedProjectId)
      if (project) {
        setCurrentProjectState(project)
      }
    }
  }, [projects])

  // Save project to localStorage when it changes
  const setCurrentProject = async (project: Project | null) => {
    setCurrentProjectState(project)
    if (project) {
      localStorage.setItem('currentProjectId', project.id)

      // Activate project for automatic paper detection
      try {
        await projectsApi.activate(project.id)
        console.log(`Project ${project.name} activated for automatic paper detection`)
      } catch (error) {
        console.error('Failed to activate project for paper detection:', error)
        // Don't block - continue even if activation fails
      }
    } else {
      localStorage.removeItem('currentProjectId')
    }
  }

  return (
    <ProjectContext.Provider
      value={{
        currentProject,
        setCurrentProject,
        projects,
        setProjects,
      }}
    >
      {children}
    </ProjectContext.Provider>
  )
}

export function useProject() {
  const context = useContext(ProjectContext)
  if (context === undefined) {
    throw new Error('useProject must be used within a ProjectProvider')
  }
  return context
}
