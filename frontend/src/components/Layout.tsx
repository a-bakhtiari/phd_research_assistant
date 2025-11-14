import { Outlet, Link, useLocation } from 'react-router-dom'
import { BookOpen, MessageSquare, Lightbulb, Settings, LayoutDashboard, FileEdit } from 'lucide-react'
import ProjectSelector from './ProjectSelector'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Papers', href: '/papers', icon: BookOpen },
  { name: 'Chat', href: '/chat', icon: MessageSquare },
  { name: 'Recommendations', href: '/recommendations', icon: Lightbulb },
  { name: 'Writing Assistant', href: '/writing-assistant', icon: FileEdit },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export default function Layout() {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700">
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">
              PhD Research Assistant
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              AI-Powered Research
            </p>
          </div>

          {/* Project Selector */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <ProjectSelector />
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.href
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`
                    flex items-center gap-3 px-4 py-3 rounded-lg transition-colors
                    ${isActive
                      ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }
                  `}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{item.name}</span>
                </Link>
              )
            })}
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Version 2.0.0
            </p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <main className="min-h-screen">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
