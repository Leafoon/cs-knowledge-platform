"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Shield, Lock, User, Settings } from "lucide-react";

interface Ring {
  level: number;
  name: string;
  color: string;
  privileges: string[];
  examples: string[];
}

const rings: Ring[] = [
  {
    level: 0,
    name: "Ring 0 (内核态)",
    color: "#dc2626",
    privileges: [
      "执行特权指令 (cli, sti, hlt, lgdt, lidt)",
      "访问所有内存地址",
      "访问 I/O 端口",
      "修改控制寄存器 (CR0, CR3, CR4)",
      "管理中断和异常",
    ],
    examples: ["操作系统内核", "设备驱动程序", "虚拟机监视器 (Hypervisor)"],
  },
  {
    level: 1,
    name: "Ring 1 (保留)",
    color: "#f59e0b",
    privileges: [
      "有限的特权指令访问",
      "部分系统资源控制",
      "中等安全级别",
    ],
    examples: ["设备驱动 (理论)", "OS 服务 (很少使用)"],
  },
  {
    level: 2,
    name: "Ring 2 (保留)",
    color: "#10b981",
    privileges: [
      "更受限的特权指令",
      "较少的系统资源访问",
      "低特权级别",
    ],
    examples: ["OS 服务 (理论)", "系统库 (很少使用)"],
  },
  {
    level: 3,
    name: "Ring 3 (用户态)",
    color: "#3b82f6",
    privileges: [
      "只能执行非特权指令",
      "通过系统调用访问内核服务",
      "受限的内存访问 (用户空间)",
      "无法直接访问硬件",
    ],
    examples: ["应用程序", "浏览器", "文本编辑器", "游戏"],
  },
];

export function ProtectionRingsVisualization() {
  const [selectedRing, setSelectedRing] = useState<number | null>(null);
  const [showComparison, setShowComparison] = useState(false);
  const [highlightedPrivilege, setHighlightedPrivilege] = useState<string | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        保护环 (Protection Rings) 可视化
      </h3>

      {/* Concentric Rings Diagram */}
      <div className="mb-8 flex justify-center">
        <svg width="500" height="500" viewBox="0 0 500 500" className="max-w-full">
          <defs>
            {rings.map((ring) => (
              <radialGradient
                key={`grad-${ring.level}`}
                id={`ring-gradient-${ring.level}`}
                cx="50%"
                cy="50%"
              >
                <stop offset="0%" stopColor={ring.color} stopOpacity="0.1" />
                <stop offset="100%" stopColor={ring.color} stopOpacity="0.4" />
              </radialGradient>
            ))}
          </defs>

          {/* Rings from outer to inner */}
          {[...rings].reverse().map((ring, index) => {
            const radius = 240 - ring.level * 60;
            const isSelected = selectedRing === ring.level;

            return (
              <g key={ring.level}>
                <motion.circle
                  cx="250"
                  cy="250"
                  r={radius}
                  fill={`url(#ring-gradient-${ring.level})`}
                  stroke={ring.color}
                  strokeWidth={isSelected ? 4 : 2}
                  className="cursor-pointer"
                  onClick={() =>
                    setSelectedRing(isSelected ? null : ring.level)
                  }
                  whileHover={{ scale: 1.02 }}
                  animate={{
                    strokeWidth: isSelected ? 4 : 2,
                    filter: isSelected
                      ? `drop-shadow(0 0 10px ${ring.color})`
                      : "none",
                  }}
                />

                {/* Ring Label */}
                <text
                  x="250"
                  y={250 - radius + 30}
                  textAnchor="middle"
                  fill={ring.color}
                  fontSize="16"
                  fontWeight="bold"
                  className="pointer-events-none"
                >
                  Ring {ring.level}
                </text>

                {/* Icon */}
                {ring.level === 0 && (
                  <g transform="translate(225, 225)">
                    <circle cx="25" cy="25" r="20" fill={ring.color} />
                    <Shield
                      className="w-8 h-8 text-white"
                      style={{ transform: "translate(9px, 9px)" }}
                    />
                  </g>
                )}
                {ring.level === 3 && (
                  <g transform="translate(380, 225)">
                    <circle cx="25" cy="25" r="20" fill={ring.color} />
                    <User
                      className="w-8 h-8 text-white"
                      style={{ transform: "translate(9px, 9px)" }}
                    />
                  </g>
                )}
              </g>
            );
          })}

          {/* Center Label */}
          <text
            x="250"
            y="255"
            textAnchor="middle"
            fontSize="14"
            fill="#6b7280"
            className="pointer-events-none"
          >
            最高特权
          </text>
        </svg>
      </div>

      {/* Ring Information */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {rings.map((ring) => {
          const isSelected = selectedRing === ring.level;
          return (
            <motion.div
              key={ring.level}
              onClick={() => setSelectedRing(isSelected ? null : ring.level)}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                isSelected
                  ? `shadow-lg`
                  : "border-gray-300 dark:border-gray-700"
              }`}
              style={{
                borderColor: isSelected ? ring.color : undefined,
                backgroundColor: isSelected ? `${ring.color}15` : undefined,
              }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-6 h-6 rounded-full"
                  style={{ backgroundColor: ring.color }}
                />
                <h4 className="font-semibold text-text-primary">
                  {ring.name}
                </h4>
              </div>
              <div className="text-sm text-text-secondary">
                <strong>CPL = {ring.level}</strong>
              </div>
              <div className="mt-2 text-xs text-text-secondary">
                {ring.examples.slice(0, 2).join(", ")}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Selected Ring Details */}
      {selectedRing !== null && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-6 rounded-lg border-2"
          style={{
            borderColor: rings[selectedRing].color,
            backgroundColor: `${rings[selectedRing].color}10`,
          }}
        >
          <h4 className="text-lg font-semibold mb-4 text-text-primary">
            {rings[selectedRing].name} - 详细特权
          </h4>

          <div className="mb-4">
            <h5 className="font-semibold mb-2 text-text-primary">
              允许的操作：
            </h5>
            <ul className="space-y-2">
              {rings[selectedRing].privileges.map((privilege, index) => (
                <li
                  key={index}
                  className="flex items-start gap-2 text-sm text-text-secondary"
                  onMouseEnter={() => setHighlightedPrivilege(privilege)}
                  onMouseLeave={() => setHighlightedPrivilege(null)}
                >
                  <span
                    className="flex-shrink-0 w-1.5 h-1.5 rounded-full mt-2"
                    style={{ backgroundColor: rings[selectedRing].color }}
                  />
                  <span
                    className={
                      highlightedPrivilege === privilege
                        ? "font-semibold text-text-primary"
                        : ""
                    }
                  >
                    {privilege}
                  </span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h5 className="font-semibold mb-2 text-text-primary">
              典型应用场景：
            </h5>
            <div className="flex flex-wrap gap-2">
              {rings[selectedRing].examples.map((example, index) => (
                <span
                  key={index}
                  className="px-3 py-1 rounded-full text-sm text-white"
                  style={{ backgroundColor: rings[selectedRing].color }}
                >
                  {example}
                </span>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Comparison Table */}
      <div className="mb-6">
        <button
          onClick={() => setShowComparison(!showComparison)}
          className="mb-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          {showComparison ? "隐藏对比" : "显示特权对比表"}
        </button>

        {showComparison && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="overflow-x-auto"
          >
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-100 dark:bg-gray-800">
                  <th className="px-4 py-3 text-left border border-border-subtle text-text-primary">
                    操作
                  </th>
                  {rings.map((ring) => (
                    <th
                      key={ring.level}
                      className="px-4 py-3 text-center border border-border-subtle"
                      style={{ color: ring.color }}
                    >
                      Ring {ring.level}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="px-4 py-2 border border-border-subtle font-semibold text-text-primary">
                    执行特权指令
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-green-600 dark:text-green-400 text-xl">
                      ✓
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      部分
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      部分
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-red-600 dark:text-red-400 text-xl">
                      ✗
                    </span>
                  </td>
                </tr>
                <tr className="bg-gray-50 dark:bg-gray-900">
                  <td className="px-4 py-2 border border-border-subtle font-semibold text-text-primary">
                    访问所有内存
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-green-600 dark:text-green-400 text-xl">
                      ✓
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      受限
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      受限
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-red-600 dark:text-red-400 text-xl">
                      ✗
                    </span>
                  </td>
                </tr>
                <tr>
                  <td className="px-4 py-2 border border-border-subtle font-semibold text-text-primary">
                    I/O 端口访问
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-green-600 dark:text-green-400 text-xl">
                      ✓
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      部分
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      部分
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-red-600 dark:text-red-400 text-xl">
                      ✗
                    </span>
                  </td>
                </tr>
                <tr className="bg-gray-50 dark:bg-gray-900">
                  <td className="px-4 py-2 border border-border-subtle font-semibold text-text-primary">
                    修改页表
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-green-600 dark:text-green-400 text-xl">
                      ✓
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-red-600 dark:text-red-400 text-xl">
                      ✗
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-red-600 dark:text-red-400 text-xl">
                      ✗
                    </span>
                  </td>
                  <td className="px-4 py-2 border border-border-subtle text-center">
                    <span className="text-red-600 dark:text-red-400 text-xl">
                      ✗
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </motion.div>
        )}
      </div>

      {/* Real-World Usage */}
      <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
        <h4 className="font-semibold mb-2 text-yellow-800 dark:text-yellow-300">
          实际应用
        </h4>
        <p className="text-sm text-text-secondary mb-2">
          <strong className="text-text-primary">现代操作系统</strong>：
          Linux、Windows、macOS 主要使用 <strong>Ring 0</strong> (内核态) 和{" "}
          <strong>Ring 3</strong> (用户态)，Ring 1 和 Ring 2 很少使用。
        </p>
        <p className="text-sm text-text-secondary">
          <strong className="text-text-primary">虚拟化场景</strong>：
          Hypervisor 运行在 Ring -1 (VMX root mode)，Guest OS 内核运行在 Ring 0
          (VMX non-root mode)。
        </p>
      </div>
    </div>
  );
}
