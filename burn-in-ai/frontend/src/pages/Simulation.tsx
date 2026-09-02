import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { Play, Pause, RotateCcw, AlertOctagon, CheckCircle2 } from 'lucide-react'

// Mock simulation data based on the prompt's synthetic example
const simulationSteps = [
  { time: '0h', rds: 10.0, limit: 50, anomaly: 0.05, drift: 'LOW', safety: 40.0, risk: 0.1, status: 'NORMAL' },
  { time: '24h', rds: 15.0, limit: 50, anomaly: 0.2, drift: 'MED', safety: 35.0, risk: 0.35, status: 'NORMAL' },
  { time: '96h', rds: 27.0, limit: 50, anomaly: 0.65, drift: 'HIGH', safety: 23.0, risk: 0.75, status: 'WATCH' },
  { time: '168h', rds: 46.0, limit: 50, anomaly: 0.95, drift: 'CRITICAL', safety: 4.0, risk: 0.98, status: 'REJECT' },
]

export default function Simulation() {
  const [step, setStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [selectedCase, setSelectedCase] = useState('DEFECT')

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying && step < 3) {
      interval = setInterval(() => {
        setStep(s => s + 1)
      }, 2000)
    } else if (step >= 3) {
      setIsPlaying(false)
    }
    return () => clearInterval(interval)
  }, [isPlaying, step])

  const currentData = simulationSteps.slice(0, step + 1)
  const currentStatus = simulationSteps[step]

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <header className="flex justify-between items-end mb-6">
        <div>
          <h1 className="text-3xl font-light tracking-tight text-white">Live Burn-In Simulation</h1>
          <p className="text-slate-400 mt-1">Accelerated trajectory analysis</p>
        </div>
        <div className="flex bg-surface p-1 rounded border border-slate-800">
          <button 
            onClick={() => {setSelectedCase('DEFECT'); setStep(0); setIsPlaying(false);}}
            className={`px-4 py-1.5 rounded text-sm font-medium ${selectedCase === 'DEFECT' ? 'bg-danger/20 text-danger' : 'text-slate-400'}`}
          >
            Defective Component
          </button>
          <button 
            onClick={() => {setSelectedCase('HEALTHY'); setStep(0); setIsPlaying(false);}}
            className={`px-4 py-1.5 rounded text-sm font-medium ${selectedCase === 'HEALTHY' ? 'bg-success/20 text-success' : 'text-slate-400'}`}
          >
            Healthy Component
          </button>
        </div>
      </header>

      <div className="flex gap-4 mb-6 bg-surface/50 p-4 rounded-xl border border-slate-700/50">
        <button onClick={() => setIsPlaying(!isPlaying)} disabled={step >= 3} className="flex items-center gap-2 bg-primary text-slate-900 px-6 py-2 rounded font-semibold hover:bg-primary/90 disabled:opacity-50">
          {isPlaying ? <Pause size={18} /> : <Play size={18} />} {isPlaying ? 'PAUSE' : 'START SIMULATION'}
        </button>
        <button onClick={() => {setStep(0); setIsPlaying(false)}} className="flex items-center gap-2 bg-slate-800 text-slate-300 px-6 py-2 rounded font-semibold hover:bg-slate-700">
          <RotateCcw size={18} /> RESET
        </button>
        <div className="ml-auto flex items-center text-sm text-slate-400">
          Checkpoint: <span className="text-white font-mono ml-2 text-lg">{currentStatus.time}</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Graph */}
        <div className="col-span-2 bg-surface/30 border border-slate-700/50 rounded-xl p-6 h-[400px]">
          <h3 className="text-sm font-semibold tracking-wider text-slate-300 mb-4">DEGRADATION TRAJECTORY (RDS_ON)</h3>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={currentData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#64748B" />
              <YAxis domain={[0, 60]} stroke="#64748B" />
              <Tooltip contentStyle={{ backgroundColor: '#1E293B', borderColor: '#334155' }} />
              <ReferenceLine y={50} stroke="#EF4444" strokeDasharray="3 3" label={{ position: 'top', value: 'Datasheet Limit', fill: '#EF4444', fontSize: 12 }} />
              
              <Line 
                type="monotone" 
                dataKey="rds" 
                stroke={currentStatus.status === 'REJECT' ? '#EF4444' : currentStatus.status === 'WATCH' ? '#F59E0B' : '#38BDF8'} 
                strokeWidth={3} 
                dot={{ r: 6, fill: '#0B0F19', strokeWidth: 2 }} 
                activeDot={{ r: 8 }} 
                isAnimationActive={true}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Real-time Intel */}
        <div className="col-span-1 space-y-4">
          <div className="bg-surface/30 border border-slate-700/50 rounded-xl p-6">
            <h3 className="text-sm font-semibold tracking-wider text-slate-400 mb-4">DYNAMIC AI DECISION</h3>
            
            <div className={`p-4 rounded-lg flex items-center gap-4 mb-6 border ${
              currentStatus.status === 'NORMAL' ? 'bg-success/10 border-success/30 text-success' :
              currentStatus.status === 'WATCH' ? 'bg-warning/10 border-warning/30 text-warning' :
              'bg-danger/10 border-danger/30 text-danger'
            }`}>
              {currentStatus.status === 'NORMAL' ? <CheckCircle2 size={32} /> : <AlertOctagon size={32} />}
              <div>
                <div className="text-xs font-semibold opacity-80 tracking-wider">STATUS</div>
                <div className="text-2xl font-bold tracking-tight">{currentStatus.status}</div>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Anomaly Score</span>
                  <span className="font-mono text-white">{currentStatus.anomaly.toFixed(2)}</span>
                </div>
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <motion.div className="h-full bg-blue-500" initial={{width:0}} animate={{width: `${currentStatus.anomaly * 100}%`}} />
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Unified Risk Score</span>
                  <span className="font-mono text-white">{currentStatus.risk.toFixed(2)}</span>
                </div>
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <motion.div 
                    className={`h-full ${currentStatus.risk > 0.7 ? 'bg-danger' : currentStatus.risk > 0.3 ? 'bg-warning' : 'bg-success'}`} 
                    initial={{width:0}} 
                    animate={{width: `${currentStatus.risk * 100}%`}} 
                  />
                </div>
              </div>

              <div className="pt-4 mt-4 border-t border-slate-700/50">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest">Predicted Drift</div>
                    <div className={`text-lg font-mono ${currentStatus.drift === 'CRITICAL' ? 'text-danger' : 'text-slate-200'}`}>{currentStatus.drift}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest">Safety Margin</div>
                    <div className="text-lg font-mono text-slate-200">{currentStatus.safety.toFixed(1)} µA</div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </motion.div>
  )
}
