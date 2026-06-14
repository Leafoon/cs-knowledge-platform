"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cable, Award } from "lucide-react";

const devices = [
  { id: 0, name: "设备0", priority: "最高", color: "red" },
  { id: 1, name: "设备1", priority: "高", color: "orange" },
  { id: 2, name: "设备2", priority: "中", color: "yellow" },
  { id: 3, name: "设备3", priority: "低", color: "blue" },
];

const colorMap: Record<string, string> = {
  red: "border-red-500 bg-red-500/10 text-red-300",
  orange: "border-orange-500 bg-orange-500/10 text-orange-300",
  yellow: "border-yellow-500 bg-yellow-500/10 text-yellow-300",
  blue: "border-blue-500 bg-blue-500/10 text-blue-300",
};

export function PriorityArbitrationCircuit() {
  const [requests, setRequests] = useState([false, false, false, false]);
  const [granted, setGranted] = useState<number | null>(null);

  const toggleRequest = (id: number) => {
    const newReqs = [...requests];
    newReqs[id] = !newReqs[id];
    setRequests(newReqs);
    setGranted(null);
  };

  const resolveArbitration = () => {
    for (let i = 0; i < requests.length; i++) {
      if (requests[i]) { setGranted(i); return; }
    }
    setGranted(null);
  };

  const hasRequest = requests.some(Boolean);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Cable className="w-5 h-5 text-amber-400" />
        <h3 className="text-lg font-semibold">优先级判优电路</h3>
      </div>

      <div className="text-xs text-gray-400 mb-3">菊链仲裁 (Daisy-Chain) - 点击设备发送中断请求</div>

      <div className="flex items-center gap-3 mb-4">
        <div className="px-3 py-2 bg-gray-700 rounded text-xs text-gray-300">仲裁器</div>
        <div className="flex-1 h-0.5 bg-gray-600 relative">
          <motion.div
            className="absolute top-0 left-0 h-full bg-amber-400"
            animate={{ width: hasRequest && granted !== null ? "100%" : "0%" }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-4">
        {devices.map((d) => (
          <motion.button
            key={d.id}
            onClick={() => toggleRequest(d.id)}
            className={`p-3 rounded-lg border-2 transition-all ${
              granted === d.id
                ? "ring-2 ring-amber-400 " + colorMap[d.color]
                : requests[d.id]
                ? colorMap[d.color]
                : "border-gray-600 bg-gray-800/30 text-gray-400"
            }`}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
          >
            <div className="text-sm font-medium">{d.name}</div>
            <div className="text-[10px] mt-1">优先级: {d.priority}</div>
            <div className={`mt-2 w-3 h-3 rounded-full mx-auto ${
              requests[d.id] ? "bg-amber-400 animate-pulse" : "bg-gray-600"
            }`} />
            <div className="text-[10px] mt-1">{requests[d.id] ? "请求中" : "空闲"}</div>
          </motion.button>
        ))}
      </div>

      <div className="flex gap-2 mb-4">
        <button
          onClick={resolveArbitration}
          disabled={!hasRequest}
          className="px-4 py-1.5 bg-amber-600 rounded text-sm text-white hover:bg-amber-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          仲裁
        </button>
        <button
          onClick={() => { setRequests([false, false, false, false]); setGranted(null); }}
          className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600"
        >
          清除
        </button>
      </div>

      <AnimatePresence>
        {granted !== null && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg text-sm flex items-center gap-2"
          >
            <Award className="w-4 h-4 text-amber-400" />
            <span className="text-amber-300">{devices[granted].name}</span>
            <span className="text-gray-400">获得总线（优先级最高且有请求）</span>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-4 p-3 bg-gray-800/30 rounded-lg text-xs text-gray-400">
        <div className="font-medium text-gray-300 mb-1">菊链仲裁原理:</div>
        <div>• IGR（中断授权）信号从高优先级设备串行传递到低优先级</div>
        <div>• 有中断请求且收到IGR的设备截获授权，不再向下传递</div>
        <div>• 优先级由物理位置决定（离仲裁器越近优先级越高）</div>
      </div>
    </div>
  );
}
