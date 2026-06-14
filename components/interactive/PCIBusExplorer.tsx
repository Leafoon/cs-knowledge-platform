"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CircuitBoard, Zap, ArrowRight } from "lucide-react";

const signalGroups = [
  {
    name: "地址/数据线",
    signals: [
      { name: "AD[31:0]", desc: "地址/数据复用线，先传地址后传数据" },
      { name: "C/BE[3:0]#", desc: "命令/字节使能复用线" },
      { name: "PAR", desc: "AD和C/BE的偶校验位" },
    ],
  },
  {
    name: "接口控制",
    signals: [
      { name: "FRAME#", desc: "帧信号，表示总线事务开始" },
      { name: "IRDY#", desc: "主设备就绪，可以传输数据" },
      { name: "TRDY#", desc: "从设备就绪，可以传输数据" },
      { name: "STOP#", desc: "从设备请求停止当前事务" },
      { name: "DEVSEL#", desc: "设备选择，从设备已认领事务" },
      { name: "IDSEL#", desc: "初始化设备选择，配置空间访问" },
    ],
  },
  {
    name: "仲裁信号",
    signals: [
      { name: "REQ#", desc: "总线请求，设备向仲裁器请求总线" },
      { name: "GNT#", desc: "总线授权，仲裁器允许使用总线" },
    ],
  },
  {
    name: "错误报告",
    signals: [
      { name: "PERR#", desc: "数据奇偶校验错误" },
      { name: "SERR#", desc: "系统错误（地址奇偶校验等）" },
    ],
  },
];

const readTiming = [
  { phase: "地址期", ad: "地址", cbe: "读命令", frame: 0, devsel: 1, irdy: 1, trdy: 1 },
  { phase: "转向期", ad: "高阻", cbe: "-", frame: 0, devsel: 1, irdy: 0, trdy: 1 },
  { phase: "数据期1", ad: "数据1", cbe: "字节使能", frame: 0, devsel: 0, irdy: 0, trdy: 0 },
  { phase: "数据期2", ad: "数据2", cbe: "字节使能", frame: 1, devsel: 0, irdy: 0, trdy: 0 },
  { phase: "空闲", ad: "-", cbe: "-", frame: 1, devsel: 1, irdy: 1, trdy: 1 },
];

function SignalLine({ active, label }: { active: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2 h-8">
      <span className="w-16 text-xs font-mono text-right">{label}</span>
      <div className="flex-1 h-6 relative bg-gray-800 rounded">
        <motion.div
          className="absolute inset-0 rounded"
          animate={{ backgroundColor: active ? "#3b82f6" : "#1f2937" }}
          transition={{ duration: 0.3 }}
        />
        <motion.div
          className="absolute top-0 left-0 h-full w-1 rounded-full"
          animate={{ x: active ? 200 : 0, backgroundColor: active ? "#60a5fa" : "#4b5563" }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
        />
      </div>
    </div>
  );
}

export function PCIBusExplorer() {
  const [tab, setTab] = useState<"signals" | "timing">("signals");
  const [selectedGroup, setSelectedGroup] = useState(0);
  const [timingStep, setTimingStep] = useState(0);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <CircuitBoard className="w-5 h-5 text-blue-400" />
        <h3 className="text-lg font-semibold">PCI 总线探索器</h3>
      </div>

      <div className="flex gap-2 mb-4">
        {(["signals", "timing"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              tab === t ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            {t === "signals" ? "信号定义" : "读写时序"}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {tab === "signals" ? (
          <motion.div
            key="signals"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <div className="flex gap-2 mb-4">
              {signalGroups.map((g, i) => (
                <button
                  key={g.name}
                  onClick={() => setSelectedGroup(i)}
                  className={`px-3 py-1 rounded text-xs ${
                    selectedGroup === i ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300"
                  }`}
                >
                  {g.name}
                </button>
              ))}
            </div>
            <div className="space-y-2">
              {signalGroups[selectedGroup].signals.map((s) => (
                <motion.div
                  key={s.name}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex gap-4 p-3 bg-gray-800/50 rounded"
                >
                  <span className="font-mono text-blue-300 w-28 shrink-0">{s.name}</span>
                  <span className="text-sm text-gray-300">{s.desc}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="timing"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <div className="flex items-center gap-2 mb-4">
              <button
                onClick={() => setTimingStep((s) => (s + 1) % readTiming.length)}
                className="px-3 py-1 bg-blue-600 rounded text-sm text-white hover:bg-blue-500"
              >
                下一步
              </button>
              <span className="text-sm text-gray-400">
                阶段: <span className="text-white">{readTiming[timingStep].phase}</span>
              </span>
            </div>
            <div className="space-y-1 mb-4">
              <SignalLine active={readTiming[timingStep].frame === 0} label="FRAME#" />
              <SignalLine active={readTiming[timingStep].irdy === 0} label="IRDY#" />
              <SignalLine active={readTiming[timingStep].trdy === 0} label="TRDY#" />
              <SignalLine active={readTiming[timingStep].devsel === 0} label="DEVSEL#" />
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="p-2 bg-gray-800/50 rounded">
                <span className="text-gray-400">AD: </span>
                <span className="text-blue-300">{readTiming[timingStep].ad}</span>
              </div>
              <div className="p-2 bg-gray-800/50 rounded">
                <span className="text-gray-400">C/BE: </span>
                <span className="text-green-300">{readTiming[timingStep].cbe}</span>
              </div>
              <div className="p-2 bg-gray-800/50 rounded">
                <span className="text-gray-400">操作: </span>
                <span className="text-yellow-300">
                  {readTiming[timingStep].phase === "地址期" ? "主设备发出地址" : "数据传输"}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
