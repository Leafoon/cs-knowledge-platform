"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Server, Cpu, Zap, Shield, Check, X, ChevronRight } from "lucide-react";

interface VType {
  id: string;
  name: string;
  nameEn: string;
  icon: React.ReactNode;
  color: string;
  borderColor: string;
  description: string;
  architecture: { label: string; color: string }[];
  performance: number;
  compatibility: number;
  complexity: number;
  pros: string[];
  cons: string[];
  examples: string[];
}

const vtypes: VType[] = [
  {
    id: "full",
    name: "全虚拟化",
    nameEn: "Full Virtualization",
    icon: <Server className="w-5 h-5" />,
    color: "from-red-500 to-orange-500",
    borderColor: "border-red-500/50",
    description: "通过二进制翻译捕获敏感指令，Guest OS 完全不需要修改，以为自己运行在真实硬件上。",
    architecture: [
      { label: "Guest App", color: "bg-blue-400" },
      { label: "Guest OS (未修改)", color: "bg-blue-500" },
      { label: "二进制翻译器", color: "bg-red-400" },
      { label: "VMM (Ring 0)", color: "bg-red-600" },
      { label: "物理硬件", color: "bg-gray-600" },
    ],
    performance: 60,
    compatibility: 95,
    complexity: 80,
    pros: [
      "Guest OS 无需任何修改",
      "可运行任意操作系统（包括闭源）",
      "兼容性最好",
    ],
    cons: [
      "二进制翻译带来显著性能开销",
      "敏感指令捕获和模拟代价高",
      "系统调用密集型负载影响大",
    ],
    examples: ["早期 VMware Workstation", "VirtualBox (部分)", "QEMU (纯软件模式)"],
  },
  {
    id: "para",
    name: "半虚拟化",
    nameEn: "Paravirtualization",
    icon: <Zap className="w-5 h-5" />,
    color: "from-green-500 to-emerald-500",
    borderColor: "border-green-500/50",
    description: "修改 Guest OS 内核，将敏感指令替换为 Hypercall 直接请求 VMM，消除翻译开销。",
    architecture: [
      { label: "Guest App", color: "bg-blue-400" },
      { label: "Guest OS (已修改)", color: "bg-green-500" },
      { label: "Hypercall 接口", color: "bg-green-400" },
      { label: "VMM", color: "bg-green-600" },
      { label: "物理硬件", color: "bg-gray-600" },
    ],
    performance: 85,
    compatibility: 50,
    complexity: 60,
    pros: [
      "性能接近原生",
      "消除了二进制翻译开销",
      "上下文切换开销更小",
    ],
    cons: [
      "需要修改 Guest OS 内核源码",
      "不支持闭源操作系统（如 Windows）",
      "维护成本高（需跟踪内核更新）",
    ],
    examples: ["Xen (早期)", "VMware Guest Tools (部分优化)", "virtio 驱动"],
  },
  {
    id: "hw",
    name: "硬件辅助虚拟化",
    nameEn: "Hardware-Assisted Virtualization",
    icon: <Cpu className="w-5 h-5" />,
    color: "from-purple-500 to-indigo-500",
    borderColor: "border-purple-500/50",
    description: "CPU 提供 VMX 模式和 VMCS 结构，硬件自动处理敏感指令的 VM Exit，VMM 运行在 Root 模式。",
    architecture: [
      { label: "Guest App", color: "bg-blue-400" },
      { label: "Guest OS (未修改)", color: "bg-blue-500" },
      { label: "VMX Non-Root 模式", color: "bg-purple-400" },
      { label: "VMM (VMX Root 模式)", color: "bg-purple-600" },
      { label: "VT-x / AMD-V 硬件", color: "bg-gray-600" },
    ],
    performance: 95,
    compatibility: 95,
    complexity: 40,
    pros: [
      "Guest OS 无需修改",
      "性能接近原生",
      "VMM 设计简化（硬件处理）",
    ],
    cons: [
      "需要 CPU 支持虚拟化扩展",
      "VM Entry/Exit 仍有开销（~数百周期）",
      "旧硬件不支持",
    ],
    examples: ["KVM + QEMU", "VMware ESXi", "Microsoft Hyper-V", "Xen HVM"],
  },
];

export default function VirtualizationTypes() {
  const [selected, setSelected] = useState("full");
  const current = vtypes.find((v) => v.id === selected)!;

  return (
    <div className="w-full space-y-6">
      <div className="flex gap-2 flex-wrap">
        {vtypes.map((v) => (
          <button
            key={v.id}
            onClick={() => setSelected(v.id)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border transition-all text-sm font-medium ${
              selected === v.id
                ? `${v.borderColor} bg-gradient-to-r ${v.color} text-white shadow-lg`
                : "border-gray-600 bg-gray-800/50 text-gray-300 hover:border-gray-500 hover:bg-gray-700/50"
            }`}
          >
            {v.icon}
            {v.name}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selected}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.2 }}
          className="space-y-5"
        >
          <div className="bg-gray-800/60 rounded-xl p-5 border border-gray-700">
            <div className="flex items-center gap-3 mb-3">
              <div className={`p-2 rounded-lg bg-gradient-to-br ${current.color} text-white`}>
                {current.icon}
              </div>
              <div>
                <h3 className="text-lg font-bold text-white">{current.name}</h3>
                <p className="text-xs text-gray-400">{current.nameEn}</p>
              </div>
            </div>
            <p className="text-gray-300 text-sm leading-relaxed">{current.description}</p>
          </div>

          <div className="bg-gray-800/60 rounded-xl p-5 border border-gray-700">
            <h4 className="text-sm font-semibold text-gray-200 mb-3">架构层次</h4>
            <div className="flex flex-col items-center gap-1">
              {current.architecture.map((layer, i) => (
                <motion.div
                  key={layer.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.08 }}
                  className="flex items-center gap-2 w-full max-w-xs"
                >
                  <div className={`w-full text-center py-2 px-3 rounded-md text-white text-xs font-medium ${layer.color}`}>
                    {layer.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "性能", value: current.performance, color: "bg-green-500" },
              { label: "兼容性", value: current.compatibility, color: "bg-blue-500" },
              { label: "实现复杂度", value: current.complexity, color: "bg-orange-500" },
            ].map((m) => (
              <div key={m.label} className="bg-gray-800/60 rounded-xl p-4 border border-gray-700 text-center">
                <p className="text-xs text-gray-400 mb-2">{m.label}</p>
                <p className="text-2xl font-bold text-white">{m.value}%</p>
                <div className="w-full bg-gray-700 rounded-full h-1.5 mt-2">
                  <motion.div
                    className={`h-1.5 rounded-full ${m.color}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${m.value}%` }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700">
              <h4 className="text-sm font-semibold text-green-400 mb-2 flex items-center gap-1">
                <Check className="w-4 h-4" /> 优点
              </h4>
              <ul className="space-y-1.5">
                {current.pros.map((p) => (
                  <li key={p} className="text-xs text-gray-300 flex items-start gap-2">
                    <ChevronRight className="w-3 h-3 text-green-500 mt-0.5 flex-shrink-0" />
                    {p}
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700">
              <h4 className="text-sm font-semibold text-red-400 mb-2 flex items-center gap-1">
                <X className="w-4 h-4" /> 缺点
              </h4>
              <ul className="space-y-1.5">
                {current.cons.map((c) => (
                  <li key={c} className="text-xs text-gray-300 flex items-start gap-2">
                    <ChevronRight className="w-3 h-3 text-red-500 mt-0.5 flex-shrink-0" />
                    {c}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700">
            <h4 className="text-sm font-semibold text-gray-200 mb-2">代表产品</h4>
            <div className="flex flex-wrap gap-2">
              {current.examples.map((e) => (
                <span key={e} className="px-3 py-1 bg-gray-700 text-gray-300 text-xs rounded-full border border-gray-600">
                  {e}
                </span>
              ))}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
