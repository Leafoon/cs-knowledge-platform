"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shield,
  Lock,
  Users,
  Key,
  User,
  UserCheck,
  Settings,
  CheckCircle,
  XCircle,
  ArrowRight,
} from "lucide-react";

type ModelId = "dac" | "mac" | "rbac" | "capabilities";

interface ModelInfo {
  id: ModelId;
  name: string;
  fullName: string;
  icon: React.ReactNode;
  color: string;
  bg: string;
  border: string;
  description: string;
  principles: string[];
  pros: string[];
  cons: string[];
  examples: string[];
}

const models: ModelInfo[] = [
  {
    id: "dac",
    name: "DAC",
    fullName: "自主访问控制 (Discretionary Access Control)",
    icon: <User className="w-5 h-5" />,
    color: "text-blue-600 dark:text-blue-400",
    bg: "bg-blue-50 dark:bg-blue-900/30",
    border: "border-blue-300 dark:border-blue-700",
    description:
      "资源的所有者可以自主决定谁可以访问该资源。最灵活但安全性最低的模型。",
    principles: [
      "所有者控制权限分配",
      "权限可以传递给其他用户",
      "基于用户身份进行访问决策",
      "操作系统强制执行所有者的决定",
    ],
    pros: ["灵活，易于使用", "实现简单，开销小", "符合日常使用直觉"],
    cons: ["权限传播难以控制", "易受提权攻击", "无法强制组织级安全策略"],
    examples: ["Unix 文件权限 (rwx)", "Windows NTFS 权限", "文件共享 ACL"],
  },
  {
    id: "mac",
    name: "MAC",
    fullName: "强制访问控制 (Mandatory Access Control)",
    icon: <Shield className="w-5 h-5" />,
    color: "text-red-600 dark:text-red-400",
    bg: "bg-red-50 dark:bg-red-900/30",
    border: "border-red-300 dark:border-red-700",
    description:
      "由系统管理员定义全局安全策略，用户无法自行修改。每个主体和客体都有安全标签。",
    principles: [
      "系统强制执行全局策略",
      "安全标签分级 (Top Secret → Unclassified)",
      "Bell-LaPadula: 不上读、不下写",
      "Biba: 不上写、不下读",
    ],
    pros: ["安全性最高", "可强制执行组织策略", "防止权限扩散"],
    cons: ["配置复杂", "灵活性低", "学习曲线陡峭"],
    examples: ["SELinux (RHEL/Fedora)", "AppArmor (Ubuntu)", "Windows Mandatory Integrity"],
  },
  {
    id: "rbac",
    name: "RBAC",
    fullName: "基于角色的访问控制 (Role-Based Access Control)",
    icon: <Users className="w-5 h-5" />,
    color: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-50 dark:bg-emerald-900/30",
    border: "border-emerald-300 dark:border-emerald-700",
    description:
      "将权限分配给角色，再将角色分配给用户。用户通过角色间接获得权限。",
    principles: [
      "用户 → 角色 → 权限",
      "角色继承层次结构",
      "互斥角色约束",
      "最小权限 + 职责分离",
    ],
    pros: ["简化权限管理", "支持角色层次", "适合企业环境"],
    cons: ["角色爆炸问题", "不适合细粒度控制", "需要定期审查角色"],
    examples: ["企业 IAM 系统", "数据库角色", "Web 应用权限框架"],
  },
  {
    id: "capabilities",
    name: "Capabilities",
    fullName: "能力机制 (Capability-Based Access Control)",
    icon: <Key className="w-5 h-5" />,
    color: "text-purple-600 dark:text-purple-400",
    bg: "bg-purple-50 dark:bg-purple-900/30",
    border: "border-purple-300 dark:border-purple-700",
    description:
      "将权限封装为不可伪造的令牌（capability），持有令牌即拥有对应权限。",
    principles: [
      "权限是不可伪造的令牌",
      "持有令牌 = 拥有权限",
      "支持权限委托和传递",
      "细粒度权限控制",
    ],
    pros: ["细粒度控制", "支持权限委托", "最小权限原则"],
    cons: ["实现复杂", "权限回收困难", "生态支持有限"],
    examples: ["Linux Capabilities", "seccomp-bpf", "WebAssembly Capabilities"],
  },
];

export default function AccessControlModels() {
  const [selected, setSelected] = useState<ModelId>("dac");
  const [showComparison, setShowComparison] = useState(false);

  const current = models.find((m) => m.id === selected)!;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <Shield className="w-8 h-8 text-blue-600 dark:text-blue-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          访问控制模型对比
        </h3>
      </div>

      {/* Model Selector */}
      <div className="flex flex-wrap gap-2 mb-6">
        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => {
              setSelected(m.id);
              setShowComparison(false);
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
              selected === m.id && !showComparison
                ? `${m.bg} ${m.color} ${m.border} border shadow-md`
                : "bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-600"
            }`}
          >
            {m.icon}
            {m.name}
          </button>
        ))}
        <button
          onClick={() => setShowComparison(!showComparison)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
            showComparison
              ? "bg-amber-50 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 border border-amber-300 dark:border-amber-700 shadow-md"
              : "bg-white dark:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-600"
          }`}
        >
          <Settings className="w-5 h-5" />
          对比表
        </button>
      </div>

      <AnimatePresence mode="wait">
        {showComparison ? (
          <motion.div
            key="comparison"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-200 dark:bg-slate-700">
                    <th className="p-3 text-left font-semibold text-slate-700 dark:text-slate-200 rounded-tl-lg">
                      特性
                    </th>
                    {models.map((m) => (
                      <th
                        key={m.id}
                        className={`p-3 text-center font-semibold ${m.color}`}
                      >
                        {m.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    {
                      label: "权限控制者",
                      values: ["资源所有者", "系统管理员", "角色管理员", "令牌持有者"],
                    },
                    {
                      label: "灵活性",
                      values: ["高", "低", "中", "高"],
                    },
                    {
                      label: "安全性",
                      values: ["低", "高", "中", "高"],
                    },
                    {
                      label: "实现复杂度",
                      values: ["低", "高", "中", "高"],
                    },
                    {
                      label: "粒度",
                      values: ["文件级", "系统级", "角色级", "对象级"],
                    },
                    {
                      label: "典型应用",
                      values: ["Unix", "SELinux", "企业IAM", "Linux Caps"],
                    },
                  ].map((row, i) => (
                    <tr
                      key={row.label}
                      className={
                        i % 2 === 0
                          ? "bg-white dark:bg-slate-800"
                          : "bg-slate-50 dark:bg-slate-750"
                      }
                    >
                      <td className="p-3 font-medium text-slate-700 dark:text-slate-300">
                        {row.label}
                      </td>
                      {row.values.map((v, j) => (
                        <td
                          key={j}
                          className="p-3 text-center text-slate-600 dark:text-slate-400"
                        >
                          {v}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key={selected}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="grid lg:grid-cols-2 gap-6"
          >
            {/* Left: Info */}
            <div className="space-y-4">
              <div className={`${current.bg} ${current.border} border rounded-lg p-4`}>
                <h4 className={`text-lg font-bold ${current.color} mb-2`}>
                  {current.fullName}
                </h4>
                <p className="text-slate-600 dark:text-slate-300 text-sm">
                  {current.description}
                </p>
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                <h5 className="font-semibold text-slate-700 dark:text-slate-200 mb-2">
                  核心原则
                </h5>
                <ul className="space-y-1">
                  {current.principles.map((p, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-sm text-slate-600 dark:text-slate-300"
                    >
                      <ArrowRight className="w-4 h-4 mt-0.5 flex-shrink-0 text-slate-400" />
                      {p}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                <h5 className="font-semibold text-slate-700 dark:text-slate-200 mb-2">
                  应用实例
                </h5>
                <div className="flex flex-wrap gap-2">
                  {current.examples.map((e, i) => (
                    <span
                      key={i}
                      className={`px-2 py-1 rounded text-xs font-medium ${current.bg} ${current.color}`}
                    >
                      {e}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: Pros/Cons */}
            <div className="space-y-4">
              <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
                <h5 className="font-semibold text-emerald-700 dark:text-emerald-300 mb-2 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4" />
                  优点
                </h5>
                <ul className="space-y-1">
                  {current.pros.map((p, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-sm text-emerald-700 dark:text-emerald-300"
                    >
                      <span className="text-emerald-500">+</span>
                      {p}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border border-red-200 dark:border-red-800">
                <h5 className="font-semibold text-red-700 dark:text-red-300 mb-2 flex items-center gap-2">
                  <XCircle className="w-4 h-4" />
                  缺点
                </h5>
                <ul className="space-y-1">
                  {current.cons.map((c, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-sm text-red-700 dark:text-red-300"
                    >
                      <span className="text-red-500">-</span>
                      {c}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Visual diagram */}
              <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
                <h5 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">
                  权限流向
                </h5>
                <div className="flex items-center justify-center gap-2 text-sm">
                  {selected === "dac" && (
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/40 rounded text-blue-700 dark:text-blue-300">
                        用户
                      </span>
                      <ArrowRight className="w-4 h-4 text-slate-400" />
                      <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/40 rounded text-blue-700 dark:text-blue-300">
                        资源
                      </span>
                      <span className="text-xs text-slate-500 ml-2">
                        (所有者决定)
                      </span>
                    </div>
                  )}
                  {selected === "mac" && (
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-red-100 dark:bg-red-900/40 rounded text-red-700 dark:text-red-300">
                        主体标签
                      </span>
                      <ArrowRight className="w-4 h-4 text-slate-400" />
                      <span className="px-2 py-1 bg-red-100 dark:bg-red-900/40 rounded text-red-700 dark:text-red-300">
                        策略检查
                      </span>
                      <ArrowRight className="w-4 h-4 text-slate-400" />
                      <span className="px-2 py-1 bg-red-100 dark:bg-red-900/40 rounded text-red-700 dark:text-red-300">
                        客体标签
                      </span>
                    </div>
                  )}
                  {selected === "rbac" && (
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-emerald-100 dark:bg-emerald-900/40 rounded text-emerald-700 dark:text-emerald-300">
                        用户
                      </span>
                      <ArrowRight className="w-4 h-4 text-slate-400" />
                      <span className="px-2 py-1 bg-emerald-100 dark:bg-emerald-900/40 rounded text-emerald-700 dark:text-emerald-300">
                        角色
                      </span>
                      <ArrowRight className="w-4 h-4 text-slate-400" />
                      <span className="px-2 py-1 bg-emerald-100 dark:bg-emerald-900/40 rounded text-emerald-700 dark:text-emerald-300">
                        权限
                      </span>
                    </div>
                  )}
                  {selected === "capabilities" && (
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/40 rounded text-purple-700 dark:text-purple-300">
                        进程
                      </span>
                      <ArrowRight className="w-4 h-4 text-slate-400" />
                      <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/40 rounded text-purple-700 dark:text-purple-300">
                        Capability 令牌
                      </span>
                      <ArrowRight className="w-4 h-4 text-slate-400" />
                      <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/40 rounded text-purple-700 dark:text-purple-300">
                        对象
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
