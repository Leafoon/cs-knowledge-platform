"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface ComponentInfo {
  id: string;
  name: string;
  desc: string;
  details: string[];
  color: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

const components: ComponentInfo[] = [
  {
    id: "cpu",
    name: "CPU (中央处理器)",
    desc: "计算机的核心部件，负责指令的执行和数据的处理",
    details: ["运算器 (ALU)：执行算术和逻辑运算", "控制器 (CU)：指挥协调各部件工作", "寄存器组：暂存指令和数据", "程序计数器 (PC)：存放下条指令地址"],
    color: "#667eea",
    x: 300, y: 60, w: 280, h: 140,
  },
  {
    id: "alu",
    name: "运算器 (ALU)",
    desc: "执行算术运算和逻辑运算的核心部件",
    details: ["加减乘除运算", "与或非异或逻辑运算", "移位操作", "状态标志位（零/溢出/进位/符号）"],
    color: "#f59e0b",
    x: 310, y: 80, w: 120, h: 50,
  },
  {
    id: "cu",
    name: "控制器 (CU)",
    desc: "产生控制信号，协调各部件按照指令要求工作",
    details: ["取指令", "分析指令", "执行指令", "产生控制信号序列"],
    color: "#10b981",
    x: 450, y: 80, w: 120, h: 50,
  },
  {
    id: "memory",
    name: "主存储器",
    desc: "存放程序和数据的部件，CPU可直接访问",
    details: ["RAM：随机存取存储器", "ROM：只读存储器", "MAR：存储器地址寄存器", "MDR：存储器数据寄存器"],
    color: "#ef4444",
    x: 50, y: 60, w: 180, h: 140,
  },
  {
    id: "input",
    name: "输入设备",
    desc: "将外部信息转换为计算机能识别的数据",
    details: ["键盘、鼠标", "扫描仪", "摄像头", "传感器"],
    color: "#8b5cf6",
    x: 50, y: 280, w: 180, h: 80,
  },
  {
    id: "output",
    name: "输出设备",
    desc: "将计算机处理结果转换为人能理解的形式",
    details: ["显示器", "打印机", "音响", "投影仪"],
    color: "#ec4899",
    x: 650, y: 280, w: 180, h: 80,
  },
  {
    id: "bus",
    name: "系统总线",
    desc: "连接各部件的信息传输通道",
    details: ["数据总线 (DB)：传输数据", "地址总线 (AB)：传输地址", "控制总线 (CB)：传输控制信号"],
    color: "#06b6d4",
    x: 200, y: 240, w: 480, h: 30,
  },
];

export function HardwareBlockDiagram() {
  const [selected, setSelected] = useState<string | null>(null);
  const selInfo = components.find((c) => c.id === selected);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        计算机硬件组成框图
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        点击各部件查看详细信息
      </p>
      <div className="flex flex-col lg:flex-row gap-4">
        <div className="flex-1 bg-bg-secondary rounded-lg p-4 overflow-x-auto">
          <svg viewBox="0 0 880 380" className="w-full min-w-[500px]">
            {/* connections */}
            <line x1="230" y1="130" x2="300" y2="130" stroke="#94a3b8" strokeWidth="2" strokeDasharray="6 3" />
            <line x1="440" y1="200" x2="440" y2="240" stroke="#94a3b8" strokeWidth="2" strokeDasharray="6 3" />
            <line x1="140" y1="200" x2="140" y2="240" stroke="#94a3b8" strokeWidth="2" strokeDasharray="6 3" />
            <line x1="140" y1="270" x2="140" y2="280" stroke="#94a3b8" strokeWidth="2" strokeDasharray="6 3" />
            <line x1="740" y1="270" x2="740" y2="280" stroke="#94a3b8" strokeWidth="2" strokeDasharray="6 3" />
            <line x1="740" y1="320" x2="680" y2="320" stroke="#94a3b8" strokeWidth="2" strokeDasharray="6 3" />
            <line x1="230" y1="320" x2="200" y2="320" stroke="#94a3b8" strokeWidth="2" strokeDasharray="6 3" />

            {/* Memory */}
            <motion.rect
              x={50} y={60} width={180} height={140} rx={12}
              fill={selected === "memory" ? "#ef444430" : "#ef444410"}
              stroke={selected === "memory" ? "#ef4444" : "#ef444480"}
              strokeWidth={selected === "memory" ? 3 : 1.5}
              className="cursor-pointer"
              onClick={() => setSelected(selected === "memory" ? null : "memory")}
              whileHover={{ scale: 1.02 }}
            />
            <text x={140} y={125} textAnchor="middle" className="fill-text-primary text-sm font-semibold pointer-events-none">主存储器</text>
            <text x={140} y={145} textAnchor="middle" className="fill-text-secondary text-xs pointer-events-none">Memory</text>

            {/* CPU box */}
            <motion.rect
              x={300} y={60} width={280} height={140} rx={12}
              fill={selected === "cpu" ? "#667eea30" : "#667eea10"}
              stroke={selected === "cpu" ? "#667eea" : "#667eea80"}
              strokeWidth={selected === "cpu" ? 3 : 1.5}
              className="cursor-pointer"
              onClick={() => setSelected(selected === "cpu" ? null : "cpu")}
              whileHover={{ scale: 1.02 }}
            />
            <text x={440} y={75} textAnchor="middle" className="fill-text-primary text-xs font-semibold pointer-events-none">CPU</text>

            {/* ALU */}
            <motion.rect
              x={310} y={90} width={120} height={50} rx={8}
              fill={selected === "alu" ? "#f59e0b30" : "#f59e0b10"}
              stroke={selected === "alu" ? "#f59e0b" : "#f59e0b80"}
              strokeWidth={selected === "alu" ? 3 : 1.5}
              className="cursor-pointer"
              onClick={(e) => { e.stopPropagation(); setSelected(selected === "alu" ? null : "alu"); }}
              whileHover={{ scale: 1.02 }}
            />
            <text x={370} y={120} textAnchor="middle" className="fill-text-primary text-xs font-semibold pointer-events-none">ALU</text>

            {/* CU */}
            <motion.rect
              x={450} y={90} width={120} height={50} rx={8}
              fill={selected === "cu" ? "#10b98130" : "#10b98110"}
              stroke={selected === "cu" ? "#10b981" : "#10b98180"}
              strokeWidth={selected === "cu" ? 3 : 1.5}
              className="cursor-pointer"
              onClick={(e) => { e.stopPropagation(); setSelected(selected === "cu" ? null : "cu"); }}
              whileHover={{ scale: 1.02 }}
            />
            <text x={510} y={120} textAnchor="middle" className="fill-text-primary text-xs font-semibold pointer-events-none">CU</text>

            {/* Registers */}
            <rect x={310} y={150} width={260} height={40} rx={6} fill="none" stroke="#94a3b880" strokeWidth={1} strokeDasharray="4 2" />
            <text x={440} y={175} textAnchor="middle" className="fill-text-secondary text-[10px] pointer-events-none">寄存器组 / PC / IR / MAR / MDR</text>

            {/* Bus */}
            <motion.rect
              x={200} y={240} width={480} height={30} rx={6}
              fill={selected === "bus" ? "#06b6d430" : "#06b6d410"}
              stroke={selected === "bus" ? "#06b6d4" : "#06b6d480"}
              strokeWidth={selected === "bus" ? 3 : 1.5}
              className="cursor-pointer"
              onClick={() => setSelected(selected === "bus" ? null : "bus")}
              whileHover={{ scale: 1.01 }}
            />
            <text x={440} y={260} textAnchor="middle" className="fill-text-primary text-xs font-semibold pointer-events-none">系统总线 (数据 / 地址 / 控制)</text>

            {/* Input */}
            <motion.rect
              x={50} y={280} width={180} height={80} rx={12}
              fill={selected === "input" ? "#8b5cf630" : "#8b5cf610"}
              stroke={selected === "input" ? "#8b5cf6" : "#8b5cf680"}
              strokeWidth={selected === "input" ? 3 : 1.5}
              className="cursor-pointer"
              onClick={() => setSelected(selected === "input" ? null : "input")}
              whileHover={{ scale: 1.02 }}
            />
            <text x={140} y={325} textAnchor="middle" className="fill-text-primary text-sm font-semibold pointer-events-none">输入设备</text>

            {/* Output */}
            <motion.rect
              x={650} y={280} width={180} height={80} rx={12}
              fill={selected === "output" ? "#ec489930" : "#ec489910"}
              stroke={selected === "output" ? "#ec4899" : "#ec489980"}
              strokeWidth={selected === "output" ? 3 : 1.5}
              className="cursor-pointer"
              onClick={() => setSelected(selected === "output" ? null : "output")}
              whileHover={{ scale: 1.02 }}
            />
            <text x={740} y={325} textAnchor="middle" className="fill-text-primary text-sm font-semibold pointer-events-none">输出设备</text>
          </svg>
        </div>

        {/* detail panel */}
        <div className="lg:w-72">
          <AnimatePresence mode="wait">
            {selInfo ? (
              <motion.div
                key={selInfo.id}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="rounded-lg border border-border-subtle bg-bg-secondary p-4"
              >
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: selInfo.color }} />
                  <h4 className="font-semibold text-sm text-text-primary">{selInfo.name}</h4>
                </div>
                <p className="text-xs text-text-secondary mb-3">{selInfo.desc}</p>
                <ul className="space-y-1.5">
                  {selInfo.details.map((d, i) => (
                    <motion.li
                      key={i}
                      initial={{ opacity: 0, x: -5 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="text-xs text-text-secondary flex items-start gap-1.5"
                    >
                      <span className="mt-1 w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: selInfo.color }} />
                      {d}
                    </motion.li>
                  ))}
                </ul>
              </motion.div>
            ) : (
              <motion.div
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="rounded-lg border border-dashed border-border-subtle bg-bg-secondary p-4 text-center"
              >
                <p className="text-sm text-text-secondary">点击左侧部件查看详情</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
