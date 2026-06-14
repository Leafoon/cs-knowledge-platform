'use client'

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, ArrowRight, Crown, FileText, CheckCircle2, Circle, AlertCircle } from 'lucide-react'

type NodeState = 'follower' | 'candidate' | 'leader' | 'dead'

interface RaftNode {
  id: number
  state: NodeState
  term: number
  log: { term: number; value: string; committed: boolean }[]
  votedFor: number | null
}

type Phase = 'idle' | 'election' | 'log-replication' | 'commit' | 'done'

interface LogEntry { term: number; value: string; committed: boolean }

const NODE_COLORS: Record<NodeState, string> = {
  follower: '#3b82f6', candidate: '#f59e0b', leader: '#22c55e', dead: '#94a3b8',
}
const NODE_LABELS: Record<NodeState, string> = {
  follower: 'Follower', candidate: 'Candidate', leader: 'Leader', dead: 'Dead',
}

function makeNode(id: number): RaftNode {
  return { id, state: 'follower', term: 0, log: [], votedFor: null }
}

export default function RaftConsensusAnimation() {
  const [nodes, setNodes] = useState<RaftNode[]>([makeNode(0), makeNode(1), makeNode(2), makeNode(3), makeNode(4)])
  const [phase, setPhase] = useState<Phase>('idle')
  const [step, setStep] = useState(0)
  const [autoPlay, setAutoPlay] = useState(false)
  const [speed, setSpeed] = useState(1500)
  const [logMessage, setLogMessage] = useState('点击"开始"观察 Raft 算法运行')
  const [leaderId, setLeaderId] = useState<number | null>(null)
  const [pendingEntry, setPendingEntry] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const reset = useCallback(() => {
    setNodes([makeNode(0), makeNode(1), makeNode(2), makeNode(3), makeNode(4)])
    setPhase('idle')
    setStep(0)
    setAutoPlay(false)
    setLogMessage('点击"开始"观察 Raft 算法运行')
    setLeaderId(null)
    setPendingEntry(null)
  }, [])

  const executeStep = useCallback((s: number) => {
    setNodes(prev => {
      const ns = prev.map(n => ({ ...n, log: [...n.log], votedFor: n.votedFor }))
      const electionSteps = [
        () => {
          ns[0].state = 'candidate'; ns[0].term = 1; ns[0].votedFor = 0
          setLogMessage('Node 0 选举超时，转为 Candidate，任期 term=1')
        },
        () => {
          ns[0].votedFor = 0; ns[1].votedFor = 0; ns[2].votedFor = 0
          ns[1].term = 1; ns[2].term = 1
          setLogMessage('Node 0 收到 Node 1、2 的投票（共 3/5 多数派）')
        },
        () => {
          ns[0].state = 'leader'; leaderId !== null || setLeaderId(0)
          setLogMessage('Node 0 获得多数票，成为 Leader (term=1)')
        },
      ]

      const replicationSteps = [
        () => {
          const entry: LogEntry = { term: 1, value: 'SET x=5', committed: false }
          ns[0].log.push(entry)
          setPendingEntry('SET x=5')
          setLogMessage('客户端请求 SET x=5，Leader 追加到本地日志')
        },
        () => {
          const entry: LogEntry = { term: 1, value: 'SET x=5', committed: false }
          ns[1].log.push({ ...entry }); ns[2].log.push({ ...entry }); ns[3].log.push({ ...entry })
          setLogMessage('Leader 将日志复制到 Followers（Node 1, 2, 3）')
        },
        () => {
          ns[1].term = 1; ns[2].term = 1; ns[3].term = 1
          setLogMessage('Followers 确认收到日志条目')
        },
      ]

      const commitSteps = [
        () => {
          ns[0].log[ns[0].log.length - 1].committed = true
          ns[1].log[ns[1].log.length - 1].committed = true
          ns[2].log[ns[2].log.length - 1].committed = true
          setLogMessage('多数派（3/5）确认，Leader 提交日志 (commitIndex++)')
        },
        () => {
          setLogMessage('Leader 通知 Followers 提交，状态机应用 SET x=5')
          setPendingEntry(null)
        },
        () => {
          setLogMessage('共识达成！x=5 已安全提交到集群')
          setPhase('done')
          setAutoPlay(false)
        },
      ]

      let allSteps: (() => void)[][]
      if (phase === 'election' || phase === 'idle') {
        allSteps = [electionSteps, replicationSteps, commitSteps]
        if (phase === 'idle') setPhase('election')
      } else if (phase === 'log-replication') {
        allSteps = [replicationSteps, commitSteps]
      } else if (phase === 'commit') {
        allSteps = [commitSteps]
      } else {
        return prev
      }

      let globalStep = s
      for (const group of allSteps) {
        if (globalStep < group.length) {
          group[globalStep]()
          break
        }
        globalStep -= group.length
      }

      return ns
    })

    setStep(s => s + 1)

    setPhase(prev => {
      if (prev === 'idle' || prev === 'election') {
        if (s >= 2 && s < 3) return 'election'
        if (s >= 3 && s < 6) return 'log-replication'
        if (s >= 6) return 'commit'
      }
      return prev
    })
  }, [phase])

  useEffect(() => {
    if (autoPlay) {
      timerRef.current = setInterval(() => setStep(s => {
        executeStep(s)
        return s + 1
      }), speed)
    } else if (timerRef.current) {
      clearInterval(timerRef.current)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [autoPlay, speed, executeStep])

  const startAnimation = () => { reset(); setTimeout(() => { setPhase('election'); setAutoPlay(true) }, 100) }
  const nextStep = () => executeStep(step)

  const cx = (i: number) => 100 + (i % 3) * 140 + (i > 2 ? 70 : 0)
  const cy = (i: number) => 80 + Math.floor(i / 3) * 120

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-4">
        <Crown className="w-6 h-6 text-amber-500" />
        <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100">Raft 共识算法动画</h3>
      </div>

      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        观察 Leader 选举 → 日志复制 → 提交的完整 Raft 共识流程。
      </p>

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <button onClick={startAnimation}
          className="px-4 py-1.5 rounded-lg bg-green-500 text-white text-sm font-medium hover:bg-green-600 flex items-center gap-1.5">
          <Play className="w-4 h-4" /> 开始
        </button>
        <button onClick={() => setAutoPlay(!autoPlay)} disabled={phase === 'idle'}
          className="px-3 py-1.5 rounded-lg bg-blue-500 text-white text-sm font-medium hover:bg-blue-600 disabled:opacity-40 flex items-center gap-1.5">
          {autoPlay ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {autoPlay ? '暂停' : '继续'}
        </button>
        <button onClick={nextStep} disabled={phase === 'idle' || phase === 'done'}
          className="px-3 py-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm font-medium hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-40 flex items-center gap-1.5">
          <ArrowRight className="w-4 h-4" /> 下一步
        </button>
        <button onClick={reset}
          className="px-3 py-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm font-medium hover:bg-slate-300 dark:hover:bg-slate-600 flex items-center gap-1.5">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
        <select value={speed} onChange={e => setSpeed(Number(e.target.value))}
          className="px-2 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-600">
          <option value={2500}>慢速</option>
          <option value={1500}>正常</option>
          <option value={700}>快速</option>
        </select>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1">
          <svg viewBox="0 0 440 260" className="w-full max-w-md mx-auto">
            {nodes.map((node, i) => {
              const x = cx(i), y = cy(i), color = NODE_COLORS[node.state]
              return (
                <g key={node.id}>
                  <motion.circle cx={x} cy={y} r={40} fill={color} opacity={node.state === 'dead' ? 0.3 : 0.15}
                    stroke={color} strokeWidth={2} animate={{ scale: node.state === 'leader' ? [1, 1.05, 1] : 1 }}
                    transition={{ repeat: node.state === 'leader' ? Infinity : 0, duration: 1.5 }} />
                  <circle cx={x} cy={y} r={28} fill={color} />
                  {node.state === 'leader' && <Crown x={x - 10} y={y - 40} className="w-5 h-5" />}
                  <text x={x} y={y - 6} textAnchor="middle" fontSize="14" fontWeight="bold" fill="white">N{node.id}</text>
                  <text x={x} y={y + 10} textAnchor="middle" fontSize="9" fill="rgba(255,255,255,0.8)">{NODE_LABELS[node.state]}</text>
                  <text x={x} y={y + 54} textAnchor="middle" fontSize="11" fill="#64748b" className="dark:fill-slate-400">term={node.term}</text>

                  {node.log.length > 0 && (
                    <g>
                      {node.log.map((entry, j) => (
                        <g key={j}>
                          <rect x={x - 24 + j * 26} y={y + 64} width={22} height={16} rx={3}
                            fill={entry.committed ? '#22c55e' : '#f59e0b'} opacity={0.9} />
                          <text x={x - 13 + j * 26} y={y + 75} textAnchor="middle" fontSize="7" fill="white" fontWeight="bold">
                            {entry.value.length > 4 ? entry.value.slice(0, 4) : entry.value}
                          </text>
                        </g>
                      ))}
                    </g>
                  )}
                </g>
              )
            })}

            {nodes.filter(n => n.state === 'leader').map(leader =>
              nodes.filter(n => n.id !== leader.id && n.state !== 'dead').map(follower => {
                const fromX = cx(leader.id), fromY = cy(leader.id)
                const toX = cx(follower.id), toY = cy(follower.id)
                return (
                  <motion.line key={`link-${leader.id}-${follower.id}`}
                    x1={fromX} y1={fromY + 30} x2={toX} y2={toY - 30}
                    stroke="#22c55e" strokeWidth={1.5} strokeDasharray="4 3" opacity={0.4}
                    initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.5 }} />
                )
              })
            )}
          </svg>
        </div>

        <div className="flex-1 min-w-0">
          <AnimatePresence mode="wait">
            <motion.div key={logMessage} initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }}
              className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 mb-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-blue-800 dark:text-blue-300">{logMessage}</p>
              </div>
            </motion.div>
          </AnimatePresence>

          <div className="space-y-2 mb-4">
            {nodes.map(node => (
              <div key={node.id} className="flex items-center gap-2 p-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                <div className="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold"
                  style={{ backgroundColor: NODE_COLORS[node.state] }}>N{node.id}</div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-slate-700 dark:text-slate-300 flex items-center gap-1">
                    {node.state === 'leader' && <Crown className="w-3 h-3 text-amber-500" />}
                    {NODE_LABELS[node.state]}
                    <span className="text-slate-400 dark:text-slate-500 ml-1">term={node.term}</span>
                    {node.votedFor !== null && <span className="text-slate-400 dark:text-slate-500 ml-1">voted=N{node.votedFor}</span>}
                  </div>
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {node.log.length === 0 && <span className="text-xs text-slate-400 dark:text-slate-500 italic">空日志</span>}
                    {node.log.map((entry, j) => (
                      <span key={j} className={`px-1.5 py-0.5 rounded text-xs font-medium ${entry.committed ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' : 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300'}`}>
                        {entry.value}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {pendingEntry && (
            <div className="p-2 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 mb-3">
              <p className="text-xs text-amber-700 dark:text-amber-300">待提交：<strong>{pendingEntry}</strong></p>
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 p-4 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Raft 三个子问题</h4>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="p-2 rounded bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
            <div className="font-medium text-blue-700 dark:text-blue-300 mb-1">Leader 选举</div>
            <p className="text-blue-600 dark:text-blue-400">Follower 超时后成为 Candidate，获得多数票成为 Leader。每个任期最多一个 Leader。</p>
          </div>
          <div className="p-2 rounded bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
            <div className="font-medium text-amber-700 dark:text-amber-300 mb-1">日志复制</div>
            <p className="text-amber-600 dark:text-amber-400">Leader 接收客户端请求，追加到本地日志后复制到 Followers。确保所有节点日志一致。</p>
          </div>
          <div className="p-2 rounded bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
            <div className="font-medium text-green-700 dark:text-green-300 mb-1">安全性</div>
            <p className="text-green-600 dark:text-green-400">多数派确认后提交。已提交的日志条目不会丢失，保证状态机安全。</p>
          </div>
        </div>
      </div>
    </div>
  )
}
