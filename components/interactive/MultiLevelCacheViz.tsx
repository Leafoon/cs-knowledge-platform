"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Layers, Clock, Database, ArrowDown, ArrowUp } from "lucide-react";

interface CacheLevel {
  name: string;
  size: string;
  latency: number;
  color: string;
}

const LEVELS: CacheLevel[] = [
  { name: "L1 Cache", size: "64 KB", latency: 1, color: "green" },
  { name: "L2 Cache", size: "512 KB", latency: 4, color: "blue" },
  { name: "L3 Cache", size: "16 MB", latency: 12, color: "purple" },
  { name: "主存", size: "16 GB", latency: 100, color: "orange" },
];

export function MultiLevelCacheViz() {
  const [hitLevel, setHitLevel] = useState<number | null>(null);
  const [animating, setAnimating] = useState(false);
  const [currentLevel, setCurrentLevel] = useState(0);

  const runAccess = () => {
    setAnimating(true);
    setCurrentLevel(0);
    setHitLevel(null);
    const hitIdx = Math.floor(Math.random() * 4);
    let i = 0;
    const interval = setInterval(() => {
      setCurrentLevel(i);
      if (i === hitIdx) {
        setHitLevel(i);
        setAnimating(false);
        clearInterval(interval);
      }
      i++;
    }, 600);
  };

  const totalTime = hitLevel !== null
    ? LEVELS.slice(0, hitLevel + 1).reduce((s, l) => s + l.latency, 0)
    : 0;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Layers className="w-5 h-5 text-blue-500" />
        多级Cache可视化
      </h3>
      <button onClick={runAccess} disabled={animating}
        className="mb-4 px-4 py-2 bg-blue-500 text-white rounded text-sm disabled:opacity-50">
        {animating ? "访问中..." : "模拟一次访存"}
      </button>
      <div className="space-y-3">
        {LEVELS.map((level, i) => {
          const isActive = animating && currentLevel === i;
          const isHit = hitLevel === i;
          const isPast = hitLevel !== null && i < hitLevel;
          return (
            <motion.div key={level.name}
              animate={{ scale: isActive || isHit ? 1.02 : 1, opacity: isPast ? 0.5 : 1 }}
              className={`flex items-center gap-4 p-4 rounded border ${isHit ? `border-${level.color}-500 bg-${level.color}-500/20` : isActive ? `border-${level.color}-500/50 bg-${level.color}-500/10` : "border-border-subtle bg-bg-subtle"}`}>
              <div className="flex items-center gap-2 w-32">
                <Database className={`w-5 h-5 ${isHit ? `text-${level.color}-500` : "text-text-secondary"}`} />
                <div>
                  <div className="text-sm font-medium">{level.name}</div>
                  <div className="text-xs text-text-secondary">{level.size}</div>
                </div>
              </div>
              <div className="flex-1 h-8 bg-bg-elevated rounded overflow-hidden">
                <motion.div initial={{ width: 0 }}
                  animate={{ width: `${(level.latency / 100) * 100}%` }}
                  className={`h-full ${isHit ? `bg-${level.color}-500` : `bg-${level.color}-500/30`} rounded`}
                />
              </div>
              <div className="flex items-center gap-1 text-sm w-24">
                <Clock className="w-3 h-3" />
                {level.latency} 周期
              </div>
              {isHit && <span className="text-xs font-medium text-green-500">HIT</span>}
              {isPast && <span className="text-xs text-text-secondary">MISS</span>}
              {i < LEVELS.length - 1 && !isHit && (
                <ArrowDown className="w-4 h-4 text-text-secondary" />
              )}
            </motion.div>
          );
        })}
      </div>
      {hitLevel !== null && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="mt-4 p-3 bg-bg-subtle rounded text-sm">
          <div className="flex items-center gap-2">
            <ArrowUp className="w-4 h-4 text-green-500" />
            <span>在 <strong>{LEVELS[hitLevel].name}</strong> 命中，总访问时间: <strong>{totalTime} 周期</strong></span>
          </div>
        </motion.div>
      )}
    </div>
  );
}
