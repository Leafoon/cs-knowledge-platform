"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Cable } from "lucide-react"

type ArbMethod = "daisy" | "independent" | "polling"

const devices = [
  { id: "D0", name: "CPU", color: "#3b82f6" },
  { id: "D1", name: "DMA", color: "#10b981" },
  { id: "D2", name: "网卡", color: "#f59e0b" },
  { id: "D3", name: "磁盘", color: "#ec4899" },
]

const allSteps: Record<ArbMethod, { label: string; requesting: number[]; granting: number[]; winner: number | null }[]> = {
  daisy: [
    { label: "D0/D2 发起总线请求 (BR)", requesting: [0, 2], granting: [], winner: null },
    { label: "仲裁器检测到请求，发出 BG (D0优先)", requesting: [0, 2], granting: [0], winner: null },
    { label: "D0 获得总线授权，发出 BS", requesting: [0, 2], granting: [0], winner: 0 },
    { label: "D0 使用总线完成传输", requesting: [2], granting: [], winner: 0 },
  ],
  independent: [
    { label: "D0/D1/D2 同时发出独立请求线", requesting: [0, 1, 2], granting: [], winner: null },
    { label: "仲裁器比较优先级: D0 > D1 > D2", requesting: [0, 1, 2], granting: [], winner: null },
    { label: "仲裁器发送 D0 授权信号", requesting: [1, 2], granting: [0], winner: 0 },
    { label: "D0 释放总线后仲裁器处理 D1", requesting: [1, 2], granting: [], winner: 0 },
  ],
  polling: [
    { label: "仲裁器轮询 D0: 是否需要总线?", requesting: [2], granting: [], winner: null },
    { label: "仲裁器轮询 D1: 是否需要总线?", requesting: [2], granting: [], winner: null },
    { label: "仲裁器轮询 D2: 需要! 授予总线", requesting: [2], granting: [2], winner: 2 },
    { label: "D2 使用总线，仲裁器继续轮询", requesting: [], granting: [], winner: 2 },
  ],
}

const methodNames: Record<ArbMethod, string> = { daisy: "菊花链仲裁", independent: "独立请求仲裁", polling: "轮询仲裁" }
const methodDescs: Record<ArbMethod, string> = {
  daisy: "BG信号沿设备链依次传递，离仲裁器最近的设备优先级最高。简单但延迟大。",
  independent: "每个设备有独立的BR/BG线，仲裁器可并行处理，速度快但线多。",
  polling: "仲裁器按顺序查询各设备，公平但速度取决于轮询周期。",
}

export function BusArbiterDemo() {
  const [method, setMethod] = useState<ArbMethod>("daisy")
  const [step, setStep] = useState(0)
  const [autoPlay, setAutoPlay] = useState(false)

  const steps = allSteps[method]
  const current = steps[step]

  useEffect(() => {
    if (!autoPlay) return
    if (step >= steps.length - 1) { setAutoPlay(false); return }
    const timer = setTimeout(() => setStep((s) => s + 1), 1800)
    return () => clearTimeout(timer)
  }, [autoPlay, step, steps.length])

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Cable className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">总线仲裁演示</h3>
      </div>
      <div className="flex gap-2 mb-3">
        {(["daisy", "independent", "polling"] as const).map((m) => (
          <button key={m} onClick={() => { setMethod(m); setStep(0); setAutoPlay(false) }}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${method === m ? "bg-accent text-white" : "bg-bg-surface border border-border-subtle text-text-secondary hover:border-accent"}`}>
            {methodNames[m]}
          </button>
        ))}
        <button onClick={() => { setStep(0); setAutoPlay(true) }} disabled={autoPlay}
          className="ml-auto px-4 py-1.5 rounded text-sm font-medium bg-green-600 text-white hover:bg-green-700 disabled:opacity-50">
          {autoPlay ? "演示中..." : "自动演示"}
        </button>
      </div>
      <p className="text-xs text-text-secondary mb-4">{methodDescs[method]}</p>
      <div className="flex gap-6">
        <svg viewBox="0 0 500 200" className="flex-1 min-h-[180px]">
          <rect x={220} y={10} width={60} height={30} rx={4} fill="#6366f1" fillOpacity={0.2} stroke="#6366f1" strokeWidth={1.5} />
          <text x={250} y={30} textAnchor="middle" fill="#6366f1" fontSize={11} fontWeight="600">仲裁器</text>
          <rect x={120} y={85} width={260} height={16} rx={4} fill="#374151" stroke="#4b5563" strokeWidth={1} />
          <text x={250} y={97} textAnchor="middle" fill="#9ca3af" fontSize={10}>共享总线</text>
          {devices.map((dev, i) => {
            const cx = 80 + i * 120
            const isReq = current.requesting.includes(i)
            const isGrant = current.granting.includes(i)
            const isWinner = current.winner === i
            return (
              <g key={dev.id}>
                <line x1={cx + 25} y1={45} x2={cx + 25} y2={85} stroke={isGrant ? dev.color : "#4b5563"} strokeWidth={isGrant ? 2 : 1} />
                <motion.rect x={cx} y={50} width={50} height={35} rx={6}
                  fill={isWinner ? dev.color : "transparent"} fillOpacity={isWinner ? 0.25 : 0.05}
                  stroke={dev.color} strokeWidth={isWinner ? 2.5 : isReq ? 2 : 1}
                  animate={isReq && !isWinner ? { strokeOpacity: [0.5, 1, 0.5] } : {}}
                  transition={{ duration: 1, repeat: Infinity }} />
                <text x={cx + 25} y={62} textAnchor="middle" fill={dev.color} fontSize={10} fontWeight="600">{dev.id}</text>
                <text x={cx + 25} y={76} textAnchor="middle" fill="#9ca3af" fontSize={9}>{dev.name}</text>
                <rect x={cx + 10} y={110} width={30} height={10} rx={2} fill={isReq ? dev.color : "#374151"} fillOpacity={isReq ? 0.4 : 0.2} />
                <text x={cx + 25} y={118} textAnchor="middle" fill="#6b7280" fontSize={7}>BR</text>
                <rect x={cx + 10} y={125} width={30} height={10} rx={2} fill={isGrant ? dev.color : "#374151"} fillOpacity={isGrant ? 0.4 : 0.2} />
                <text x={cx + 25} y={133} textAnchor="middle" fill="#6b7280" fontSize={7}>BG</text>
                {isWinner && <motion.g initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 300 }}>
                  <circle cx={cx + 25} cy={155} r={10} fill={dev.color} fillOpacity={0.2} stroke={dev.color} strokeWidth={1.5} />
                  <text x={cx + 25} y={159} textAnchor="middle" fill={dev.color} fontSize={10}>✓</text>
                </motion.g>}
              </g>
            )
          })}
          {method === "daisy" && <>
            <line x1={250} y1={42} x2={105} y2={55} stroke="#6366f1" strokeWidth={1} strokeDasharray="4 2" />
            <line x1={105} y1={70} x2={225} y2={55} stroke="#6366f1" strokeWidth={1} strokeDasharray="4 2" />
          </>}
          {method === "independent" && devices.map((_, i) => (
            <line key={i} x1={250} y1={42} x2={105 + i * 120} y2={55} stroke="#6366f1" strokeWidth={1} strokeDasharray="4 2" />
          ))}
        </svg>
        <div className="w-52 flex flex-col gap-2">
          <AnimatePresence mode="wait">
            <motion.div key={`${method}-${step}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
              className="p-3 rounded-lg border border-border-subtle bg-bg-surface">
              <div className="text-xs text-text-secondary mb-1">步骤 {step + 1}/{steps.length}</div>
              <div className="text-sm text-text-primary">{current.label}</div>
            </motion.div>
          </AnimatePresence>
          <div className="flex gap-2">
            <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0 || autoPlay}
              className="flex-1 px-3 py-1.5 rounded text-sm border border-border-subtle text-text-secondary hover:border-accent disabled:opacity-40">上一步</button>
            <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step >= steps.length - 1 || autoPlay}
              className="flex-1 px-3 py-1.5 rounded text-sm border border-border-subtle text-text-secondary hover:border-accent disabled:opacity-40">下一步</button>
          </div>
          <button onClick={() => { setStep(0); setAutoPlay(false) }}
            className="px-3 py-1.5 rounded text-sm border border-border-subtle text-text-secondary hover:border-accent">重置</button>
        </div>
      </div>
    </div>
  )
}
