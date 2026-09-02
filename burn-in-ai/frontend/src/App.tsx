import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, ShieldAlert, Cpu, Layers, BarChart2, PlayCircle, Settings, ChevronRight } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Simulation from './pages/Simulation'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')

  return (
    <div className="min-h-screen bg-background text-slate-100 flex overflow-hidden font-sans">
      {/* Sidebar */}
      <div className="w-64 border-r border-slate-800 bg-slate-900/50 backdrop-blur-md flex flex-col z-20">
        <div className="p-6 border-b border-slate-800">
          <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-blue-400 bg-clip-text text-transparent flex items-center gap-2">
            <ShieldAlert className="text-primary h-6 w-6" />
            VANGUARD
          </h1>
          <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest">Dynamic Burn-In AI</p>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
          {[
            { id: 'dashboard', icon: <Activity size={18} />, label: 'Overview' },
            { id: 'simulation', icon: <PlayCircle size={18} />, label: 'Live Burn-In Sim' },
            { id: 'mosfet', icon: <Cpu size={18} />, label: 'MOSFET Intelligence' },
            { id: 'igbt', icon: <Layers size={18} />, label: 'IGBT Intelligence' },
            { id: 'capacitor', icon: <Layers size={18} />, label: 'Capacitor Intel' },
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                activeTab === item.id 
                ? 'bg-primary/10 text-primary border border-primary/20 shadow-[0_0_15px_rgba(56,189,248,0.1)]' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              {item.icon}
              <span className="text-sm font-medium">{item.label}</span>
            </button>
          ))}
        </nav>
        
        <div className="p-4 border-t border-slate-800 text-xs text-slate-500 flex justify-between items-center">
          <span>v2.4.1 (Space Grade)</span>
          <Settings size={14} className="hover:text-slate-300 cursor-pointer" />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 relative overflow-y-auto">
        {/* Ambient background glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-primary/5 rounded-full blur-[120px] pointer-events-none" />
        
        <div className="p-8 max-w-7xl mx-auto relative z-10">
          <AnimatePresence mode="wait">
            {activeTab === 'dashboard' && <Dashboard key="dashboard" />}
            {activeTab === 'simulation' && <Simulation key="simulation" />}
            {['mosfet', 'igbt', 'capacitor'].includes(activeTab) && (
              <motion.div 
                key="placeholder"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex flex-col items-center justify-center h-[60vh] text-slate-400"
              >
                <Cpu size={48} className="mb-4 opacity-20" />
                <h2 className="text-2xl font-light mb-2">{activeTab.toUpperCase()} Intelligence</h2>
                <p>Component-specific module is ready for integration.</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}

export default App
