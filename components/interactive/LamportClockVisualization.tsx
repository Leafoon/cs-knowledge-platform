'use client'

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, Clock, MessageCircle, ArrowRight } from 'lucide-react'

interface ProcessEvent {
  id: string
  processIdx: number
  type: 'local' | 'send' | 'receive'
  lc: number
  label: string
  linkedTo?: string
  x: number
}

interface Message {
  id: string
  from: number
  to: number
  lc: number
  label: string
  sent: boolean
  received: boolean
}

const PROCESS_COLORS = ['#3b82f6', '#22c55e', '#a855f7']
const PROCESS_NAMES = ['P1', 'P2', 'P3']

export default function LamportClockVisualization() {
  const [events, setEvents] = useState<ProcessEvent[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [lc, setLc] = useState([0, 0, 0])
  const [step, setStep] = useState(0)
  const [autoPlay, setAutoPlay] = useState(false)
  const [speed, setSpeed] = useState(1200)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const timelineWidth = 340
  const processGap = 100
  const startX = 40

  const getEventX = (idx: number) => startX + idx * 45

  const scenario: { processIdx: number; type: 'local' | 'send' | 'receive'; label: string; targetMsg?: string }[] = [
    { processIdx: 0, type: 'local', label: 'a' },
    { processIdx: 1, type: 'local', label: 'b' },
    { processIdx: 0, type: 'send', label: 'send m1', targetMsg: 'm1' },
    { processIdx: 1, type: 'receive', label: 'recv m1', targetMsg: 'm1' },
    { processIdx: 1, type: 'send', label: 'send m2', targetMsg: 'm2' },
    { processIdx: 2, type: 'receive', label: 'recv m2', targetMsg: 'm2' },
    { processIdx: 0, type: 'local', label: 'c' },
    { processIdx: 2, type: 'local', label: 'd' },
    { processIdx: 2, type: 'send', label: 'send m3', targetMsg: 'm3' },
    { processIdx: 0, type: 'receive', label: 'recv m3', targetMsg: 'm3' },
  ]

  const reset = useCallback(() => {
    setEvents([])
    setMessages([])
    setLc([0, 0, 0])
    setStep(0)
    setAutoPlay(false)
  }, [])

  const executeStep = useCallback((currentStep: number) => {
    if (currentStep >= scenario.length) {
      setAutoPlay(false)
      return
    }

    const action = scenario[currentStep]
    setLc(prev => {
      const newLc = [...prev]
      if (action.type === 'local') {
        newLc[action.processIdx] += 1
      } else if (action.type === 'send') {
        newLc[action.processIdx] += 1
      } else if (action.type === 'receive') {
        const msg = messages.find(m => m.id === action.targetMsg)
        const msgLc = msg ? msg.lc : 0
        newLc[action.processIdx] = Math.max(newLc[action.processIdx], msgLc) + 1
      }

      const evt: ProcessEvent = {
        id: `e${currentStep}`,
        processIdx: action.processIdx,
        type: action.type,
        lc: newLc[action.processIdx],
        label: action.label,
        x: getEventX(currentStep),
        linkedTo: action.targetMsg,
      }
      setEvents(prev => [...prev, evt])

      if (action.type === 'send') {
        setMessages(prev => [...prev, {
          id: action.targetMsg!,
          from: action.processIdx,
          to: -1,
          lc: newLc[action.processIdx],
          label: action.targetMsg!,
          sent: true,
          received: false,
        }])
      } else if (action.type === 'receive') {
        setMessages(prev => prev.map(m => m.id === action.targetMsg ? { ...m, to: action.processIdx, received: true } : m))
      }

      return newLc
    })

    setStep(currentStep + 1)
  }, [messages])

  useEffect(() => {
    if (autoPlay) {
      timerRef.current = setInterval(() => {
        setStep(s => {
          if (s >= scenario.length) {
            setAutoPlay(false)
            return s
          }
          executeStep(s)
          return s
        })
      }, speed)
    } else if (timerRef.current) {
      clearInterval(timerRef.current)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [autoPlay, speed, executeStep])

  const nextStep = () => {
    if (step < scenario.length) executeStep(step)
  }

  const processY = (idx: number) => 60 + idx * processGap

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-4">
        <Clock className="w-6 h-6 text-blue-500" />
        <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100">Lamport 逻辑时钟可视化</h3>
      </div>

      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        观察多个进程间的事件与消息传递，理解 Lamport 逻辑时钟的更新规则。
      </p>

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <button onClick={() => setAutoPlay(!autoPlay)}
          className="px-3 py-1.5 rounded-lg bg-blue-500 text-white text-sm font-medium hover:bg-blue-600 flex items-center gap-1.5">
          {autoPlay ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {autoPlay ? '暂停' : '自动播放'}
        </button>
        <button onClick={nextStep} disabled={step >= scenario.length}
          className="px-3 py-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm font-medium hover:bg-slate-300 dark:hover:bg-slate-600 disabled:opacity-40 flex items-center gap-1.5">
          <ArrowRight className="w-4 h-4" /> 下一步
        </button>
        <button onClick={reset}
          className="px-3 py-1.5 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm font-medium hover:bg-slate-300 dark:hover:bg-slate-600 flex items-center gap-1.5">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
        <select value={speed} onChange={e => setSpeed(Number(e.target.value))}
          className="px-2 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-sm text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-600">
          <option value={2000}>慢速</option>
          <option value={1200}>正常</option>
          <option value={600}>快速</option>
        </select>
      </div>

      <div className="overflow-x-auto">
        <svg width={Math.max(500, startX + scenario.length * 45 + 60)} height={processY(2) + 80} className="mx-auto">
          {PROCESS_NAMES.map((name, i) => {
            const y = processY(i)
            return (
              <g key={name}>
                <line x1={20} y1={y} x2={startX + scenario.length * 45 + 40} y2={y}
                  stroke="#cbd5e1" strokeWidth="1" strokeDasharray="4 4" />
                <circle cx={20} cy={y} r={14} fill={PROCESS_COLORS[i]} />
                <text x={20} y={y + 1} textAnchor="middle" dominantBaseline="middle" fontSize="11" fontWeight="bold" fill="white">{name}</text>
              </g>
            )
          })}

          {messages.filter(m => m.received).map(msg => {
            const fromY = processY(msg.from)
            const toY = processY(msg.to)
            const fromEvt = events.find(e => e.type === 'send' && e.linkedTo === msg.id)
            const toEvt = events.find(e => e.type === 'receive' && e.linkedTo === msg.id)
            if (!fromEvt || !toEvt) return null
            return (
              <g key={msg.id}>
                <defs>
                  <marker id={`arrow-${msg.id}`} markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#f59e0b" />
                  </marker>
                </defs>
                <motion.line x1={fromEvt.x} y1={fromY} x2={toEvt.x} y2={toY}
                  stroke="#f59e0b" strokeWidth="2" markerEnd={`url(#arrow-${msg.id})`}
                  initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.4 }} />
                <text x={(fromEvt.x + toEvt.x) / 2} y={(fromY + toY) / 2 - 8}
                  textAnchor="middle" fontSize="10" fill="#f59e0b" fontWeight="bold">
                  {msg.label} (LC={msg.lc})
                </text>
              </g>
            )
          })}

          {events.map((evt, i) => {
            const y = processY(evt.processIdx)
            const color = PROCESS_COLORS[evt.processIdx]
            return (
              <motion.g key={evt.id} initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.3 }}>
                <circle cx={evt.x} cy={y} r={10} fill={evt.type === 'send' ? '#f59e0b' : evt.type === 'receive' ? '#22c55e' : color}
                  stroke="white" strokeWidth="2" />
                {evt.type === 'send' && <MessageCircle className="w-3 h-3" x={evt.x - 4} y={y - 4} />}
                <text x={evt.x} y={y + 1} textAnchor="middle" dominantBaseline="middle" fontSize="8" fill="white" fontWeight="bold">
                  {evt.lc}
                </text>
                <text x={evt.x} y={y - 18} textAnchor="middle" fontSize="9" fill={color} fontWeight="medium">
                  {evt.label}
                </text>
                <text x={evt.x} y={y + 24} textAnchor="middle" fontSize="10" fill="#64748b" className="dark:fill-slate-400">
                  LC={evt.lc}
                </text>
              </motion.g>
            )
          })}
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3">
        {lc.map((v, i) => (
          <div key={i} className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-center">
            <div className="text-xs text-slate-500 dark:text-slate-400">{PROCESS_NAMES[i]} 逻辑时钟</div>
            <motion.div key={v} initial={{ scale: 1.3 }} animate={{ scale: 1 }}
              className="text-2xl font-bold mt-1" style={{ color: PROCESS_COLORS[i] }}>{v}</motion.div>
          </div>
        ))}
      </div>

      <div className="mt-4 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
        <h4 className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-2">Lamport 时钟更新规则</h4>
        <ul className="text-xs text-blue-700 dark:text-blue-400 space-y-1">
          <li>1. <strong>本地事件</strong>：LC = LC + 1</li>
          <li>2. <strong>发送消息</strong>：LC = LC + 1，将 LC 附加到消息</li>
          <li>3. <strong>接收消息</strong>：LC = max(本地LC, 消息LC) + 1</li>
        </ul>
      </div>

      {step >= scenario.length && (
        <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
          <p className="text-sm text-green-800 dark:text-green-300 font-medium">
            场景完成！观察到因果关系：a → c → recv(m3)，LC 严格递增。但 LC(b) = 2 和 LC(a) = 1 不代表 b 发生在 a 之后——Lamport 时钟无法判断并发关系。
          </p>
        </motion.div>
      )}
    </div>
  )
}
