"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shield,
  CheckCircle2,
  XCircle,
  Layers,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Code2,
  Bug,
  Lock,
  FileCheck,
} from "lucide-react";

interface OSInfo {
  name: string;
  tcbSize: string;
  tcbLines: number;
  verificationMethod: string;
  syscalls: number;
  properties: string[];
  color: string;
  darkColor: string;
  borderColor: string;
  darkBorderColor: string;
  bgColor: string;
  darkBgColor: string;
  badgeColor: string;
  darkBadgeColor: string;
}

interface VerificationLayer {
  name: string;
  description: string;
  detail: string;
  icon: React.ReactNode;
}

const osData: OSInfo[] = [
  {
    name: "seL4",
    tcbSize: "~10K 行",
    tcbLines: 10000,
    verificationMethod: "Isabelle/HOL 形式化验证",
    syscalls: 13,
    properties: [
      "功能正确性 (Functional Correctness)",
      "完整性 (Integrity)",
      "机密性 (Confidentiality)",
      "信息流安全性 (Information Flow)",
      "能力安全 (Capability Safety)",
    ],
    color: "bg-emerald-100",
    darkColor: "dark:bg-emerald-900/40",
    borderColor: "border-emerald-400",
    darkBorderColor: "dark:border-emerald-600",
    bgColor: "bg-emerald-50",
    darkBgColor: "dark:bg-emerald-950/30",
    badgeColor: "bg-emerald-500",
    darkBadgeColor: "dark:bg-emerald-600",
  },
  {
    name: "Linux",
    tcbSize: "~27M 行",
    tcbLines: 27000000,
    verificationMethod: "测试驱动 (Testing-based)",
    syscalls: 300,
    properties: [],
    color: "bg-orange-100",
    darkColor: "dark:bg-orange-900/40",
    borderColor: "border-orange-400",
    darkBorderColor: "dark:border-orange-600",
    bgColor: "bg-orange-50",
    darkBgColor: "dark:bg-orange-950/30",
    badgeColor: "bg-orange-500",
    darkBadgeColor: "dark:bg-orange-600",
  },
];

const verificationLayers: VerificationLayer[] = [
  {
    name: "Abstract Spec",
    description: "抽象规范",
    detail: "定义系统行为的数学模型，不涉及实现细节",
    icon: <FileCheck className="w-5 h-5" />,
  },
  {
    name: "Executable Spec",
    description: "可执行规范",
    detail: "将抽象规范细化为可执行的形式化描述",
    icon: <Code2 className="w-5 h-5" />,
  },
  {
    name: "C Implementation",
    description: "C 语言实现",
    detail: "生产代码，证明与可执行规范的行为等价",
    icon: <Bug className="w-5 h-5" />,
  },
  {
    name: "Binary",
    description: "二进制代码",
    detail: "编译产物，证明与 C 实现语义一致",
    icon: <Lock className="w-5 h-5" />,
  },
];

const tcbComparison = [
  { label: "代码行数", sel4: "~10,000", linux: "~27,000,000" },
  { label: "系统调用数", sel4: "13", linux: "300+" },
  { label: "验证方法", sel4: "Isabelle/HOL", linux: "测试覆盖" },
  { label: "形式化证明", sel4: "✓ 有", linux: "✗ 无" },
  { label: "安全保证级别", sel4: "数学证明", linux: "经验估计" },
];

export default function FormalVerificationComparison() {
  const [selectedOS, setSelectedOS] = useState<number>(0);
  const [showLayers, setShowLayers] = useState(true);
  const [expandedLayer, setExpandedLayer] = useState<number | null>(null);

  const current = osData[selectedOS];

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-gray-900 rounded-2xl space-y-8">
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 flex items-center justify-center gap-2">
          <Shield className="w-7 h-7 text-emerald-500" />
          形式化验证对比
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          seL4 微内核 vs 传统操作系统 — 安全验证深度解析
        </p>
      </div>

      <div className="flex justify-center gap-3">
        {osData.map((os, idx) => (
          <button
            key={os.name}
            onClick={() => setSelectedOS(idx)}
            className={`px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
              selectedOS === idx
                ? `${os.color} ${os.darkColor} ${os.borderColor} ${os.darkBorderColor} border-2 shadow-md`
                : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border-2 border-transparent hover:border-gray-300 dark:hover:border-gray-600"
            }`}
          >
            {os.name}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selectedOS}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.25 }}
          className={`p-6 rounded-2xl border-2 ${current.borderColor} ${current.darkBorderColor} ${current.bgColor} ${current.darkBgColor}`}
        >
          <div className="flex items-center gap-3 mb-5">
            <span
              className={`px-3 py-1 rounded-lg text-white text-sm font-bold ${current.badgeColor} ${current.darkBadgeColor}`}
            >
              {current.name}
            </span>
            <span className="text-gray-700 dark:text-gray-300 font-medium">
              {current.verificationMethod}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-5">
            <div className="text-center p-3 rounded-xl bg-white/70 dark:bg-gray-800/50">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                TCB 大小
              </div>
              <div className="text-lg font-bold text-gray-800 dark:text-gray-100">
                {current.tcbSize}
              </div>
            </div>
            <div className="text-center p-3 rounded-xl bg-white/70 dark:bg-gray-800/50">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                系统调用
              </div>
              <div className="text-lg font-bold text-gray-800 dark:text-gray-100">
                {current.syscalls}
                {current.syscalls > 100 ? "+" : ""}
              </div>
            </div>
            <div className="text-center p-3 rounded-xl bg-white/70 dark:bg-gray-800/50">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                验证工具
              </div>
              <div className="text-lg font-bold text-gray-800 dark:text-gray-100">
                {selectedOS === 0 ? "HOL" : "N/A"}
              </div>
            </div>
          </div>

          {current.properties.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                已证明的安全属性：
              </h4>
              <div className="flex flex-wrap gap-2">
                {current.properties.map((p) => (
                  <span
                    key={p}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs rounded-full bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 font-medium"
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    {p}
                  </span>
                ))}
              </div>
            </div>
          )}

          {current.properties.length === 0 && (
            <div className="flex items-center gap-2 text-sm text-orange-600 dark:text-orange-400">
              <XCircle className="w-4 h-4" />
              无形式化安全证明，依赖测试和代码审查
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      <div className="bg-gray-50 dark:bg-gray-800/50 rounded-2xl p-5">
        <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-4 text-center">
          TCB 关键指标对比
        </h3>
        <div className="space-y-2">
          {tcbComparison.map((row) => (
            <div
              key={row.label}
              className="grid grid-cols-3 gap-3 items-center text-sm"
            >
              <span className="text-right font-medium text-gray-600 dark:text-gray-400">
                {row.sel4}
              </span>
              <span className="text-center text-xs text-gray-500 dark:text-gray-500 font-semibold bg-gray-200 dark:bg-gray-700 rounded-full py-0.5">
                {row.label}
              </span>
              <span className="text-left font-medium text-gray-600 dark:text-gray-400">
                {row.linux}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <button
          onClick={() => setShowLayers(!showLayers)}
          className="flex items-center justify-center gap-2 w-full text-sm font-bold text-gray-700 dark:text-gray-300 mb-4 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors"
        >
          <Layers className="w-5 h-5" />
          seL4 验证层次结构
          {showLayers ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        <AnimatePresence>
          {showLayers && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="space-y-3">
                {verificationLayers.map((layer, idx) => (
                  <motion.div
                    key={layer.name}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                  >
                    <button
                      onClick={() =>
                        setExpandedLayer(expandedLayer === idx ? null : idx)
                      }
                      className="w-full flex items-center gap-4 p-4 rounded-xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-emerald-400 dark:hover:border-emerald-600 transition-all group"
                    >
                      <div className="flex items-center justify-center w-10 h-10 rounded-full bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400 group-hover:bg-emerald-200 dark:group-hover:bg-emerald-800/50 transition-colors">
                        {layer.icon}
                      </div>
                      <div className="flex-1 text-left">
                        <div className="font-bold text-gray-800 dark:text-gray-100 text-sm">
                          {layer.name}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {layer.description}
                        </div>
                      </div>
                      {idx < verificationLayers.length - 1 && (
                        <ArrowRight className="w-4 h-4 text-gray-400 mr-2" />
                      )}
                      {expandedLayer === idx ? (
                        <ChevronUp className="w-4 h-4 text-gray-400" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-400" />
                      )}
                    </button>

                    <AnimatePresence>
                      {expandedLayer === idx && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div className="px-4 py-3 ml-14 text-sm text-gray-600 dark:text-gray-400 bg-emerald-50/50 dark:bg-emerald-950/20 rounded-b-xl border-x border-b border-emerald-200 dark:border-emerald-800">
                            {layer.detail}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                ))}

                <div className="flex items-center justify-center gap-2 pt-2">
                  {verificationLayers.map((_, idx) => (
                    <React.Fragment key={idx}>
                      <div className="w-3 h-3 rounded-full bg-emerald-400 dark:bg-emerald-500" />
                      {idx < verificationLayers.length - 1 && (
                        <div className="w-8 h-0.5 bg-emerald-300 dark:bg-emerald-600" />
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
