"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { HardDrive, MemoryStick, Info } from "lucide-react";

export default function SevenStateModel() {
  const [hoveredState, setHoveredState] = useState<string | null>(null);

  const states = [
    {
      id: "new",
      name: "新建",
      color: "bg-blue-500",
      position: { x: 50, y: 50 },
      inMemory: false,
      description: "进程正在创建，PCB 已分配"
    },
    {
      id: "ready",
      name: "就绪",
      color: "bg-green-500",
      position: { x: 200, y: 50 },
      inMemory: true,
      description: "在内存中，等待 CPU 调度"
    },
    {
      id: "running",
      name: "运行",
      color: "bg-yellow-500",
      position: { x: 350, y: 50 },
      inMemory: true,
      description: "正在 CPU 上执行"
    },
    {
      id: "blocked",
      name: "阻塞",
      color: "bg-red-500",
      position: { x: 200, y: 150 },
      inMemory: true,
      description: "在内存中，等待 I/O 或事件"
    },
    {
      id: "suspended-ready",
      name: "挂起就绪",
      color: "bg-purple-500",
      position: { x: 200, y: 250 },
      inMemory: false,
      description: "在磁盘中，一旦换入可立即运行"
    },
    {
      id: "suspended-blocked",
      name: "挂起阻塞",
      color: "bg-orange-500",
      position: { x: 350, y: 250 },
      inMemory: false,
      description: "在磁盘中，等待事件 + 换入"
    },
    {
      id: "terminated",
      name: "终止",
      color: "bg-gray-500",
      position: { x: 500, y: 50 },
      inMemory: false,
      description: "进程已结束"
    }
  ];

  const transitions = [
    { from: "new", to: "ready", label: "初始化完成", color: "stroke-slate-600" },
    { from: "ready", to: "running", label: "调度", color: "stroke-green-600" },
    { from: "running", to: "ready", label: "时间片用完", color: "stroke-orange-600" },
    { from: "running", to: "blocked", label: "等待 I/O", color: "stroke-red-600" },
    { from: "blocked", to: "ready", label: "事件发生", color: "stroke-blue-600" },
    { from: "ready", to: "suspended-ready", label: "swap out", color: "stroke-purple-600" },
    { from: "suspended-ready", to: "ready", label: "swap in", color: "stroke-green-600" },
    { from: "blocked", to: "suspended-blocked", label: "swap out", color: "stroke-purple-600" },
    { from: "suspended-blocked", to: "suspended-ready", label: "事件发生", color: "stroke-blue-600" },
    { from: "suspended-blocked", to: "blocked", label: "swap in", color: "stroke-green-600" },
    { from: "running", to: "terminated", label: "exit()", color: "stroke-gray-600" }
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center">
        七状态模型：增加挂起状态
      </h3>

      {/* Memory vs Disk Legend */}
      <div className="flex justify-center gap-6 mb-6">
        <div className="flex items-center gap-2">
          <MemoryStick className="w-5 h-5 text-blue-600" />
          <span className="text-sm font-semibold text-slate-700">在内存中</span>
        </div>
        <div className="flex items-center gap-2">
          <HardDrive className="w-5 h-5 text-purple-600" />
          <span className="text-sm font-semibold text-slate-700">在磁盘中（挂起）</span>
        </div>
      </div>

      {/* State Diagram */}
      <div className="bg-white rounded-lg shadow-md p-6 relative" style={{ height: "400px" }}>
        {/* States */}
        {states.map(state => (
          <motion.div
            key={state.id}
            onMouseEnter={() => setHoveredState(state.id)}
            onMouseLeave={() => setHoveredState(null)}
            whileHover={{ scale: 1.1 }}
            className={`absolute ${state.color} text-white p-4 rounded-lg shadow-lg cursor-pointer transition-all ${
              hoveredState === state.id ? "z-10 ring-4 ring-blue-300" : ""
            }`}
            style={{
              left: `${state.position.x}px`,
              top: `${state.position.y}px`,
              width: "120px"
            }}
          >
            <div className="text-center">
              <div className="text-xs mb-1">
                {state.inMemory ? (
                  <MemoryStick className="inline w-3 h-3" />
                ) : (
                  <HardDrive className="inline w-3 h-3" />
                )}
              </div>
              <div className="font-bold text-sm">{state.name}</div>
            </div>
          </motion.div>
        ))}

        {/* Transition Arrows (simplified) */}
        <svg className="absolute inset-0" width="100%" height="100%" style={{ pointerEvents: "none" }}>
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
              <polygon points="0 0, 10 3, 0 6" fill="#64748b" />
            </marker>
          </defs>
          {/* Simple arrows representation */}
          {transitions.slice(0, 6).map((trans, idx) => (
            <line
              key={idx}
              x1={states.find(s => s.id === trans.from)!.position.x + 60}
              y1={states.find(s => s.id === trans.from)!.position.y + 30}
              x2={states.find(s => s.id === trans.to)!.position.x + 60}
              y2={states.find(s => s.id === trans.to)!.position.y + 30}
              className={trans.color}
              strokeWidth="2"
              markerEnd="url(#arrow)"
            />
          ))}
        </svg>
      </div>

      {/* Detail Panel */}
      {hoveredState && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 bg-blue-50 border-2 border-blue-400 rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Info className="w-5 h-5 text-blue-600" />
            <h4 className="font-bold text-slate-800">
              {states.find(s => s.id === hoveredState)?.name}
            </h4>
          </div>
          <p className="text-sm text-slate-700">
            {states.find(s => s.id === hoveredState)?.description}
          </p>
          <div className="mt-2 text-xs text-slate-600">
            位置：{states.find(s => s.id === hoveredState)?.inMemory ? "内存" : "磁盘（交换区）"}
          </div>
        </motion.div>
      )}

      {/* Transition Table */}
      <div className="mt-6 bg-white rounded-lg shadow-md p-4">
        <h4 className="font-bold text-slate-800 mb-3">状态转换表</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-100">
                <th className="px-3 py-2 text-left">当前状态</th>
                <th className="px-3 py-2 text-left">事件</th>
                <th className="px-3 py-2 text-left">新状态</th>
              </tr>
            </thead>
            <tbody>
              {transitions.map((trans, idx) => (
                <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="px-3 py-2 font-semibold">
                    {states.find(s => s.id === trans.from)?.name}
                  </td>
                  <td className="px-3 py-2 text-slate-600">{trans.label}</td>
                  <td className="px-3 py-2 font-semibold">
                    {states.find(s => s.id === trans.to)?.name}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Info Box */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <p className="text-sm text-slate-700">
          <strong>挂起（Suspend）</strong>：操作系统将进程暂时移出内存（swap out）到磁盘交换区，
          释放内存给其他进程。挂起进程不占用物理内存，但保留 PCB。需要时可换入（swap in）内存。
        </p>
      </div>
    </div>
  );
}
