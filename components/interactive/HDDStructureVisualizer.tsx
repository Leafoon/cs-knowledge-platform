"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { HardDrive, Clock, Zap, Info, Gauge } from "lucide-react";

interface LatencyBreakdown {
  seekTime: number;
  rotationalLatency: number;
  transferTime: number;
  total: number;
}

export function HDDStructureVisualizer() {
  const [rpm, setRpm] = useState(7200);
  const [selectedPart, setSelectedPart] = useState<string | null>(null);
  const [headAngle, setHeadAngle] = useState(0);
  const [isSpinning, setIsSpinning] = useState(true);
  const animRef = useRef<number>();

  const rotationPerSec = rpm / 60;
  const avgRotationalLatency = (60 / rpm / 2) * 1000; // ms
  const seekTime = 8; // ms average
  const transferRate = 150; // MB/s typical
  const sectorSize = 512; // bytes

  const transferTime = (sectorSize / (transferRate * 1024 * 1024)) * 1000; // ms per sector

  const latency: LatencyBreakdown = {
    seekTime,
    rotationalLatency: avgRotationalLatency,
    transferTime,
    total: seekTime + avgRotationalLatency + transferTime,
  };

  useEffect(() => {
    if (!isSpinning) return;
    const spin = () => {
      setHeadAngle(prev => (prev + rotationPerSec * 2) % 360);
      animRef.current = requestAnimationFrame(spin);
    };
    animRef.current = requestAnimationFrame(spin);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [isSpinning, rotationPerSec]);

  const parts = [
    { id: "platter", label: "盘片 (Platter)", desc: "铝/玻璃基板涂覆磁性材料，数据存储表面。通常多个盘片堆叠在同一主轴上。", color: "blue" },
    { id: "track", label: "磁道 (Track)", desc: "盘片上的同心圆，每个磁道被划分为多个扇区。外圈磁道存储更多扇区。", color: "green" },
    { id: "sector", label: "扇区 (Sector)", desc: "磁道上的弧段，通常 512 字节或 4KB。是磁盘读写的最小单位。", color: "purple" },
    { id: "cylinder", label: "柱面 (Cylinder)", desc: "所有盘片上同一半径位置的磁道集合。切换柱面需要寻道，同一柱面内切换只需切换磁头。", color: "orange" },
    { id: "head", label: "磁头 (R/W Head)", desc: "悬浮在盘片表面约 10nm 处进行读写。每个盘片表面有一个磁头，安装在磁臂上。", color: "red" },
    { id: "arm", label: "磁臂 (Actuator Arm)", desc: "由音圈电机驱动，精确定位磁头到目标磁道。寻道时间取决于移动距离。", color: "cyan" },
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string; dark: string }> = {
    blue: { bg: "bg-blue-50 dark:bg-blue-900/20", border: "border-blue-400 dark:border-blue-600", text: "text-blue-700 dark:text-blue-300", dark: "dark:border-blue-800" },
    green: { bg: "bg-green-50 dark:bg-green-900/20", border: "border-green-400 dark:border-green-600", text: "text-green-700 dark:text-green-300", dark: "dark:border-green-800" },
    purple: { bg: "bg-purple-50 dark:bg-purple-900/20", border: "border-purple-400 dark:border-purple-600", text: "text-purple-700 dark:text-purple-300", dark: "dark:border-purple-800" },
    orange: { bg: "bg-orange-50 dark:bg-orange-900/20", border: "border-orange-400 dark:border-orange-600", text: "text-orange-700 dark:text-orange-300", dark: "dark:border-orange-800" },
    red: { bg: "bg-red-50 dark:bg-red-900/20", border: "border-red-400 dark:border-red-600", text: "text-red-700 dark:text-red-300", dark: "dark:border-red-800" },
    cyan: { bg: "bg-cyan-50 dark:bg-cyan-900/20", border: "border-cyan-400 dark:border-cyan-600", text: "text-cyan-700 dark:text-cyan-300", dark: "dark:border-cyan-800" },
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 mb-2">
        <HardDrive className="w-6 h-6 text-blue-600 dark:text-blue-400" />
        <h3 className="text-lg font-bold text-text-primary">HDD 结构可视化</h3>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium text-text-secondary">RPM</label>
            <input
              type="range"
              min={5400}
              max={15000}
              step={7200}
              value={rpm}
              onChange={e => setRpm(parseInt(e.target.value))}
              className="flex-1"
            />
            <span className="font-mono text-sm text-text-primary w-16">{rpm}</span>
          </div>
          <div className="flex gap-2">
            {[5400, 7200, 10000, 15000].map(r => (
              <button
                key={r}
                onClick={() => setRpm(r)}
                className={`px-3 py-1 rounded text-sm ${rpm === r ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}
              >
                {r}
              </button>
            ))}
          </div>

          <div className="relative w-full aspect-square max-w-[320px] mx-auto">
            <svg viewBox="0 0 300 300" className="w-full h-full">
              <circle cx="150" cy="150" r="130" fill="none" stroke="currentColor" strokeWidth="2" className="text-gray-300 dark:text-gray-600" />
              {[100, 80, 60, 40].map((r, i) => (
                <circle key={i} cx="150" cy="150" r={r} fill="none" stroke="currentColor" strokeWidth="0.5" className="text-gray-200 dark:text-gray-700"
                  strokeDasharray="4 4" />
              ))}
              <g style={{ transformOrigin: "150px 150px", transform: `rotate(${headAngle}deg)` }}>
                {[0, 60, 120, 180, 240, 300].map((angle, i) => (
                  <line key={i} x1="150" y1="150" x2={150 + 125 * Math.cos(angle * Math.PI / 180)} y2={150 + 125 * Math.sin(angle * Math.PI / 180)}
                    stroke="currentColor" strokeWidth="0.5" className="text-gray-300 dark:text-gray-600" />
                ))}
              </g>
              <line x1="150" y1="150" x2="280" y2="150" stroke="currentColor" strokeWidth="3" className="text-red-500"
                style={{ transformOrigin: "150px 150px", transform: `rotate(${headAngle * 0.1 + 20}deg)` }} />
              <circle cx="150" cy="150" r="8" fill="currentColor" className="text-gray-400 dark:text-gray-500" />
              <circle cx="150" cy="150" r="3" fill="currentColor" className="text-gray-600 dark:text-gray-300" />
              <circle cx={150 + 120 * Math.cos((headAngle * 0.1 + 20) * Math.PI / 180)} cy={150 + 120 * Math.sin((headAngle * 0.1 + 20) * Math.PI / 180)}
                r="5" fill="currentColor" className="text-red-500" />
            </svg>
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-2">
              <button
                onClick={() => setIsSpinning(!isSpinning)}
                className="px-3 py-1 rounded-full text-xs bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700"
              >
                {isSpinning ? "暂停" : "旋转"}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {parts.map(part => {
              const c = colorMap[part.color];
              return (
                <button
                  key={part.id}
                  onClick={() => setSelectedPart(selectedPart === part.id ? null : part.id)}
                  className={`p-2 rounded-lg border text-left text-sm transition-all ${
                    selectedPart === part.id
                      ? `${c.bg} ${c.border} ${c.text}`
                      : "border-gray-200 dark:border-gray-700 text-text-secondary hover:border-gray-400"
                  }`}
                >
                  {part.label}
                </button>
              );
            })}
          </div>

          <AnimatePresence mode="wait">
            {selectedPart && (
              <motion.div
                key={selectedPart}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className={`p-4 rounded-lg border ${colorMap[parts.find(p => p.id === selectedPart)!.color].bg} ${colorMap[parts.find(p => p.id === selectedPart)!.color].border}`}
              >
                <p className="text-sm text-text-secondary">{parts.find(p => p.id === selectedPart)!.desc}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="space-y-4">
          <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Clock className="w-5 h-5 text-orange-500" />
              <h4 className="font-semibold text-text-primary">延迟分解</h4>
            </div>
            <div className="space-y-3">
              {[
                { label: "寻道时间", value: seekTime, unit: "ms", color: "bg-red-500", pct: seekTime / latency.total },
                { label: "旋转延迟", value: avgRotationalLatency, unit: "ms", color: "bg-orange-500", pct: avgRotationalLatency / latency.total },
                { label: "传输时间", value: transferTime, unit: "ms", color: "bg-green-500", pct: transferTime / latency.total },
              ].map(item => (
                <div key={item.label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-text-secondary">{item.label}</span>
                    <span className="font-mono text-text-primary">{item.value.toFixed(3)} {item.unit}</span>
                  </div>
                  <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className={`h-full ${item.color} rounded-full`}
                      initial={{ width: 0 }}
                      animate={{ width: `${item.pct * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                </div>
              ))}
              <div className="pt-2 border-t border-gray-200 dark:border-gray-700 flex justify-between">
                <span className="font-semibold text-text-primary">总访问时间</span>
                <span className="font-mono font-semibold text-blue-600 dark:text-blue-400">{latency.total.toFixed(3)} ms</span>
              </div>
            </div>
          </div>

          <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Gauge className="w-5 h-5 text-blue-500" />
              <h4 className="font-semibold text-text-primary">关键参数</h4>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "转速", value: `${rpm} RPM` },
                { label: "旋转周期", value: `${(60000 / rpm).toFixed(1)} ms` },
                { label: "平均旋转延迟", value: `${avgRotationalLatency.toFixed(2)} ms` },
                { label: "传输速率", value: `~${transferRate} MB/s` },
                { label: "扇区大小", value: "512B / 4KB" },
                { label: "磁头悬浮高度", value: "~10 nm" },
              ].map(item => (
                <div key={item.label} className="p-2 bg-white dark:bg-gray-800 rounded">
                  <div className="text-xs text-text-secondary">{item.label}</div>
                  <div className="font-mono text-sm font-semibold text-text-primary">{item.value}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-text-secondary">
                <p><strong className="text-text-primary">磁盘访问时间</strong> = 寻道时间 + 旋转延迟 + 传输时间。寻道时间占主导，因此磁盘调度算法对性能至关重要。</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
