import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ProjectProvider } from './contexts/ProjectContext'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Papers from './pages/Papers'
import Chat from './pages/Chat'
import Recommendations from './pages/Recommendations'
import WritingAssistant from './pages/WritingAssistant'
import Settings from './pages/Settings'

function App() {
  return (
    <ProjectProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="papers" element={<Papers />} />
            <Route path="chat" element={<Chat />} />
            <Route path="recommendations" element={<Recommendations />} />
            <Route path="writing-assistant" element={<WritingAssistant />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ProjectProvider>
  )
}

export default App
