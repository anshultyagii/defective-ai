import { motion } from 'framer-motion'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, CartesianGrid } from 'recharts'
import { Activity, AlertTriangle, ShieldCheck, Zap } from 'lucide-react'

const statCards = [
  { title: 'TOTAL SCREENED', value: '14,285', icon: <Activity className="text-blue-400" /> },
  { title: 'HIGH-RISK DETECTED', value: '412', icon: <AlertTriangle className="text-danger" /> },
  { title: 'DYNAMIC FLAGS', value: '156', icon: <Zap className="text-warning" /> },
  { title: 'FNR ESTIMATE', value: '0.01%', icon: <ShieldCheck className="text-success" /> },
]

const riskData = [
  { name: 'NORMAL', value: 13589, color: '#10B981' },
  { name: 'WATCH', value: 284, color: '#F59E0B' },
  { name: 'REJECT', value: 412, color: '#EF4444' },
]

const compData = [
  { name: 'MOSFET', anomalies: 210, normal: 5400 },
  { name: 'IGBT', anomalies: 104, normal: 3800 },
  { name: 'CAPACITOR', anomalies: 98, normal: 4389 },
]

export default function Dashboard() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-6"
    >
      <header className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-3xl font-light tracking-tight text-white">Fleet Overview</h1>
          <p className="text-slate-400 mt-1">Cross-component reliability screening</p>
        </div>
        <div className="flex gap-2">
          {['ALL', 'MOSFET', 'IGBT', 'CAPACITOR'].map(filter => (
            <button key={filter} className={`px-4 py-1.5 rounded text-xs font-semibold tracking-wider ${filter === 'ALL' ? 'bg-primary text-slate-900' : 'bg-surface border border-slate-700 text-slate-300 hover:bg-slate-700'}`}>
              {filter}
            </button>
          ))}
        </div>
      </header>

      {/* Stats Row */}
      <div className="grid grid-cols-4 gap-4">
        {statCards.map((card, i) => (
          <motion.div
            key={card.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="bg-surface/50 border border-slate-700/50 backdrop-blur-sm p-5 rounded-xl relative overflow-hidden group"
          >
            <div className="absolute top-0 right-0 p-4 opacity-20 group-hover:opacity-50 transition-opacity">
              {card.icon}
            </div>
            <p className="text-xs text-slate-400 font-medium tracking-wider mb-2">{card.title}</p>
            <p className="text-3xl font-light text-white">{card.value}</p>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-6 mt-6">
        {/* Risk Distribution */}
        <div className="col-span-1 bg-surface/30 border border-slate-700/50 rounded-xl p-6 backdrop-blur-sm">
          <h3 className="text-sm font-semibold tracking-wider text-slate-300 mb-6">UNIFIED RISK DISTRIBUTION</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskData}
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  {riskData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip 
                  contentStyle={{ backgroundColor: '#1E293B', borderColor: '#334155', color: '#fff' }}
                  itemStyle={{ color: '#fff' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-4 text-xs">
            {riskData.map(d => (
              <div key={d.name} className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
                <span className="text-slate-400">{d.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Component Anomalies */}
        <div className="col-span-2 bg-surface/30 border border-slate-700/50 rounded-xl p-6 backdrop-blur-sm">
          <h3 className="text-sm font-semibold tracking-wider text-slate-300 mb-6">DETECTIONS BY COMPONENT FAMILY</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={compData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="name" stroke="#64748B" tick={{fill: '#64748B', fontSize: 12}} />
                <YAxis stroke="#64748B" tick={{fill: '#64748B', fontSize: 12}} />
                <RechartsTooltip 
                  contentStyle={{ backgroundColor: '#1E293B', borderColor: '#334155', color: '#fff' }}
                  cursor={{fill: '#334155', opacity: 0.2}}
                />
                <Bar dataKey="anomalies" name="Anomalies" fill="#EF4444" radius={[4, 4, 0, 0]} barSize={40} />
                <Bar dataKey="normal" name="Normal" fill="#10B981" radius={[4, 4, 0, 0]} barSize={40} opacity={0.3} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      {/* Dynamic vs Static Comparison Teaser */}
      <div className="mt-8 bg-gradient-to-r from-blue-900/20 to-indigo-900/20 border border-blue-500/20 rounded-xl p-6 relative overflow-hidden">
        <div className="relative z-10 flex justify-between items-center">
          <div>
            <h3 className="text-lg font-medium text-white mb-1">Traditional ESS vs. Dynamic AI Screening</h3>
            <p className="text-sm text-blue-200/60 max-w-2xl">
              156 components in the current batch passed static datasheet limits but were flagged for unsafe degradation trajectories.
            </p>
          </div>
          <button className="px-5 py-2 bg-blue-500/20 text-blue-300 border border-blue-500/30 rounded shadow-lg hover:bg-blue-500/30 transition-all text-sm font-medium">
            View Analysis
          </button>
        </div>
      </div>
    </motion.div>
  )
}
