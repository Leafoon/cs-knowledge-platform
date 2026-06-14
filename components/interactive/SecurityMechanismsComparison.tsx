"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shield,
  Lock,
  Key,
  Eye,
  Layers,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Info,
} from "lucide-react";

interface Mechanism {
  id: string;
  name: string;
  fullName: string;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
  borderColor: string;
  layer: string;
  protection: string;
  howItWorks: string;
  limitations: string[];
  effectiveness: number;
  performance: number;
  compatibility: number;
}

const mechanisms: Mechanism[] = [
  {
    id: "aslr",
    name: "ASLR",
    fullName: "地址空间布局随机化",
    icon: <Layers className="w-5 h-5" />,
    color: "text-blue-600 dark:text-blue-400",
    bgColor: "bg-blue-50 dark:bg-blue-900/30",
    borderColor: "border-blue-300 dark:border-blue-700",
    layer: "操作系统",
    protection: "防止攻击者预测内存地址",
    howItWorks:
      "每次程序启动时随机化栈、堆、共享库、代码段的基址。攻击者无法预先计算目标地址，增加漏洞利用难度。",
    limitations: [
      "信息泄露漏洞可绕过",
      "32 位系统熵不足（~16 bits）",
      "不防止相对地址攻击",
      "需要 PIE 编译选项支持代码段随机化",
    ],
    effectiveness: 70,
    performance: 95,
    compatibility: 90,
  },
  {
    id: "dep",
    name: "DEP/NX",
    fullName: "数据执行保护",
    icon: <Lock className="w-5 h-5" />,
    color: "text-red-600 dark:text-red-400",
    bgColor: "bg-red-50 dark:bg-red-900/30",
    borderColor: "border-red-300 dark:border-red-700",
    layer: "CPU 硬件",
    protection: "阻止数据区域执行代码",
    howItWorks:
      "CPU 的 NX（No-Execute）位标记内存页为不可执行。栈和堆上的数据无法直接执行 shellcode。",
    limitations: [
      "ROP 攻击可绕过（复用已有代码）",
      "需要 CPU 硬件支持",
      "旧程序可能不兼容",
      "JIT 编译需要特殊处理",
    ],
    effectiveness: 75,
    performance: 98,
    compatibility: 85,
  },
  {
    id: "canary",
    name: "Stack Canary",
    fullName: "栈保护哨兵",
    icon: <Key className="w-5 h-5" />,
    color: "text-emerald-600 dark:text-emerald-400",
    bgColor: "bg-emerald-50 dark:bg-emerald-900/30",
    borderColor: "border-emerald-300 dark:border-emerald-700",
    layer: "编译器",
    protection: "检测栈缓冲区溢出",
    howItWorks:
      "在返回地址前放置随机哨兵值。函数返回前检查哨兵是否被修改，若被覆盖则调用 __stack_chk_fail 终止程序。",
    limitations: [
      "只能检测栈溢出，不能防护堆溢出",
      "信息泄露可获取 canary 值",
      "格式化字符串漏洞可泄露",
      "需要编译器支持（-fstack-protector）",
    ],
    effectiveness: 65,
    performance: 92,
    compatibility: 95,
  },
  {
    id: "selinux",
    name: "SELinux",
    fullName: "安全增强型 Linux",
    icon: <Shield className="w-5 h-5" />,
    color: "text-purple-600 dark:text-purple-400",
    bgColor: "bg-purple-50 dark:bg-purple-900/30",
    borderColor: "border-purple-300 dark:border-purple-700",
    layer: "内核 LSM",
    protection: "强制访问控制，限制进程权限",
    howItWorks:
      "为每个进程和资源分配安全上下文标签，基于全局策略决定访问权限。即使 root 进程也受策略约束。",
    limitations: [
      "配置复杂，学习曲线陡峭",
      "策略维护成本高",
      "误配置可能导致服务不可用",
      "性能开销（AVC 缓存可缓解）",
    ],
    effectiveness: 90,
    performance: 85,
    compatibility: 70,
  },
  {
    id: "seccomp",
    name: "seccomp",
    fullName: "安全计算模式",
    icon: <Eye className="w-5 h-5" />,
    color: "text-amber-600 dark:text-amber-400",
    bgColor: "bg-amber-50 dark:bg-amber-900/30",
    borderColor: "border-amber-300 dark:border-amber-700",
    layer: "内核",
    protection: "限制进程可用的系统调用",
    howItWorks:
      "通过 BPF 过滤器定义进程允许使用的系统调用白名单。禁止不必要的系统调用，减少攻击面。",
    limitations: [
      "粒度限于系统调用级别",
      "无法控制系统调用的参数",
      "BPF 过滤器本身可能有漏洞",
      "调试困难",
    ],
    effectiveness: 75,
    performance: 97,
    compatibility: 80,
  },
];

function BarChart({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 dark:text-slate-400 w-16">
        {label}
      </span>
      <div className="flex-1 h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className={`h-full rounded-full ${color}`}
        />
      </div>
      <span className="text-xs font-mono text-slate-600 dark:text-slate-300 w-8">
        {value}%
      </span>
    </div>
  );
}

export default function SecurityMechanismsComparison() {
  const [selected, setSelected] = useState<string>("aslr");
  const [viewMode, setViewMode] = useState<"detail" | "table">("detail");

  const current = mechanisms.find((m) => m.id === selected)!;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-4">
        <Shield className="w-8 h-8 text-slate-700 dark:text-slate-300" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          安全防护机制对比
        </h3>
      </div>

      {/* View toggle */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setViewMode("detail")}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
            viewMode === "detail"
              ? "bg-slate-700 dark:bg-slate-600 text-white"
              : "bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600"
          }`}
        >
          详情视图
        </button>
        <button
          onClick={() => setViewMode("table")}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
            viewMode === "table"
              ? "bg-slate-700 dark:bg-slate-600 text-white"
              : "bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600"
          }`}
        >
          表格对比
        </button>
      </div>

      <AnimatePresence mode="wait">
        {viewMode === "table" ? (
          <motion.div
            key="table"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="overflow-x-auto bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-100 dark:bg-slate-700">
                    <th className="p-3 text-left font-semibold text-slate-700 dark:text-slate-200">
                      机制
                    </th>
                    <th className="p-3 text-left font-semibold text-slate-700 dark:text-slate-200">
                      防护层
                    </th>
                    <th className="p-3 text-left font-semibold text-slate-700 dark:text-slate-200">
                      防护目标
                    </th>
                    <th className="p-3 text-center font-semibold text-slate-700 dark:text-slate-200">
                      有效性
                    </th>
                    <th className="p-3 text-center font-semibold text-slate-700 dark:text-slate-200">
                      性能
                    </th>
                    <th className="p-3 text-center font-semibold text-slate-700 dark:text-slate-200">
                      兼容性
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {mechanisms.map((m, i) => (
                    <tr
                      key={m.id}
                      onClick={() => {
                        setSelected(m.id);
                        setViewMode("detail");
                      }}
                      className={`cursor-pointer transition-colors ${
                        i % 2 === 0
                          ? "bg-white dark:bg-slate-800"
                          : "bg-slate-50 dark:bg-slate-750"
                      } hover:bg-slate-100 dark:hover:bg-slate-700`}
                    >
                      <td className="p-3">
                        <div className="flex items-center gap-2">
                          <span className={m.color}>{m.icon}</span>
                          <span className="font-medium text-slate-700 dark:text-slate-200">
                            {m.name}
                          </span>
                        </div>
                      </td>
                      <td className="p-3 text-slate-600 dark:text-slate-300">
                        {m.layer}
                      </td>
                      <td className="p-3 text-slate-600 dark:text-slate-300">
                        {m.protection}
                      </td>
                      <td className="p-3">
                        <div className="flex items-center justify-center">
                          <div className="w-16 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                m.effectiveness >= 80
                                  ? "bg-emerald-500"
                                  : m.effectiveness >= 60
                                  ? "bg-amber-500"
                                  : "bg-red-500"
                              }`}
                              style={{ width: `${m.effectiveness}%` }}
                            />
                          </div>
                          <span className="ml-1.5 text-xs font-mono text-slate-500">
                            {m.effectiveness}
                          </span>
                        </div>
                      </td>
                      <td className="p-3">
                        <div className="flex items-center justify-center">
                          <div className="w-16 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full bg-blue-500"
                              style={{ width: `${m.performance}%` }}
                            />
                          </div>
                          <span className="ml-1.5 text-xs font-mono text-slate-500">
                            {m.performance}
                          </span>
                        </div>
                      </td>
                      <td className="p-3">
                        <div className="flex items-center justify-center">
                          <div className="w-16 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full bg-purple-500"
                              style={{ width: `${m.compatibility}%` }}
                            />
                          </div>
                          <span className="ml-1.5 text-xs font-mono text-slate-500">
                            {m.compatibility}
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="detail"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Mechanism selector */}
            <div className="flex flex-wrap gap-2 mb-4">
              {mechanisms.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setSelected(m.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    selected === m.id
                      ? `${m.bgColor} ${m.color} ${m.borderColor} border shadow-md`
                      : "bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-600"
                  }`}
                >
                  {m.icon}
                  {m.name}
                </button>
              ))}
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={selected}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="grid lg:grid-cols-2 gap-4"
              >
                {/* Left: Info */}
                <div className="space-y-4">
                  <div
                    className={`${current.bgColor} ${current.borderColor} border rounded-lg p-4`}
                  >
                    <h4 className={`text-lg font-bold ${current.color} mb-1`}>
                      {current.fullName}
                    </h4>
                    <p className="text-sm text-slate-600 dark:text-slate-300">
                      {current.protection}
                    </p>
                  </div>

                  <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                    <h5 className="font-semibold text-slate-700 dark:text-slate-200 mb-2 flex items-center gap-2">
                      <Info className="w-4 h-4" />
                      工作原理
                    </h5>
                    <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
                      {current.howItWorks}
                    </p>
                  </div>

                  <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                    <h5 className="font-semibold text-slate-700 dark:text-slate-200 mb-2">
                      所在层次：{current.layer}
                    </h5>
                  </div>
                </div>

                {/* Right: Stats & Limitations */}
                <div className="space-y-4">
                  <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                    <h5 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">
                      评估指标
                    </h5>
                    <div className="space-y-2">
                      <BarChart
                        label="有效性"
                        value={current.effectiveness}
                        color="bg-emerald-500"
                      />
                      <BarChart
                        label="性能"
                        value={current.performance}
                        color="bg-blue-500"
                      />
                      <BarChart
                        label="兼容性"
                        value={current.compatibility}
                        color="bg-purple-500"
                      />
                    </div>
                  </div>

                  <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border border-red-200 dark:border-red-800">
                    <h5 className="font-semibold text-red-700 dark:text-red-300 mb-2 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" />
                      局限性
                    </h5>
                    <ul className="space-y-1">
                      {current.limitations.map((l, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-2 text-sm text-red-700 dark:text-red-300"
                        >
                          <XCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                          {l}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Best practice note */}
      <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
        <p className="text-xs text-amber-700 dark:text-amber-300">
          <strong>最佳实践：</strong>
          没有单一机制能提供完全防护。应采用<strong>纵深防御</strong>（Defense in Depth）策略，同时启用 ASLR + DEP/NX + Stack Canary + SELinux/seccomp，形成多层安全屏障。
        </p>
      </div>
    </div>
  );
}
