"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, ChevronDown, ChevronUp, CheckCircle, XCircle, Minus } from "lucide-react";

type RTOS = "vxworks" | "freertos" | "qnx" | "preempt_rt";

interface Feature {
  name: string;
  vxworks: string | boolean;
  freertos: string | boolean;
  qnx: string | boolean;
  preempt_rt: string | boolean;
  category: string;
}

const features: Feature[] = [
  { name: "许可证", vxworks: "商业", freertos: "MIT (免费)", qnx: "商业", preempt_rt: "GPL", category: "基本信息" },
  { name: "内核架构", vxworks: "单体内核", freertos: "单体内核", qnx: "微内核", preempt_rt: "单体内核 (补丁)", category: "基本信息" },
  { name: "内核大小", vxworks: "~500 KB", freertos: "6-12 KB", qnx: "~300 KB", preempt_rt: "~10 MB", category: "基本信息" },
  { name: "支持架构", vxworks: "x86, ARM, PowerPC, MIPS", freertos: "ARM Cortex-M/R/A, RISC-V, x86", qnx: "x86, ARM, AArch64", preempt_rt: "x86, ARM, RISC-V", category: "基本信息" },
  { name: "最大优先级数", vxworks: "256", freertos: "可配置 (通常 32-256)", qnx: "256", preempt_rt: "100 (RT) + 0-139", category: "调度" },
  { name: "调度策略", vxworks: "FIFO, RR, 时间片", freertos: "固定优先级抢占", qnx: "FIFO, RR, SPORADIC, 其他", preempt_rt: "FIFO, RR, DEADLINE", category: "调度" },
  { name: "上下文切换延迟", vxworks: "< 1 μs", freertos: "< 10 μs", qnx: "< 1 μs", preempt_rt: "< 10 μs", category: "性能" },
  { name: "中断延迟", vxworks: "< 1 μs", freertos: "< 5 μs", qnx: "< 1 μs", preempt_rt: "< 50 μs", category: "性能" },
  { name: "优先级反转处理", vxworks: "优先级继承", freertos: "互斥锁 (优先级继承)", qnx: "优先级继承/天花板", preempt_rt: "RT-Mutex + 优先级继承", category: "调度" },
  { name: "POSIX 兼容", vxworks: "部分 (PSE52)", freertos: "部分 (POSIX shim)", qnx: "完整 (PSE54)", preempt_rt: "完整 (Linux POSIX)", category: "兼容性" },
  { name: "内存保护 (MMU)", vxworks: "支持 (VxWorks 653)", freertos: "可选 (MPU)", qnx: "完整支持", preempt_rt: "完整支持", category: "安全" },
  { name: "文件系统", vxworks: "HRFS, DOS-FS", freertos: "FatFS, LittleFS", qnx: "QNX4, ETFS, 还有更多", preempt_rt: "ext4, XFS, Btrfs 等", category: "功能" },
  { name: "网络栈", vxworks: "WindNet (完整)", freertos: "lwIP 或 FreeRTOS+TCP", qnx: "完整 TCP/IP", preempt_rt: "Linux 完整网络栈", category: "功能" },
  { name: "多核支持", vxworks: "AMP, SMP", freertos: "SMP (有限)", qnx: "SMP, 自适应分区", preempt_rt: "完整 SMP", category: "调度" },
  { name: "安全认证", vxworks: "DO-178C, IEC 61508", freertos: "SAFERTOS (商业版)", qnx: "ISO 26262, IEC 61508", preempt_rt: "需额外认证", category: "安全" },
  { name: "调试工具", vxworks: "Wind River Workbench", freertos: "GDB, IDE 集成", qnx: "Momentics IDE", preempt_rt: "GDB, perf, ftrace", category: "工具" },
  { name: "社区/生态", vxworks: "中等 (商业支持)", freertos: "非常丰富 (开源)", qnx: "中等 (商业支持)", preempt_rt: "非常丰富 (Linux)", category: "生态" },
  { name: "典型应用", vxworks: "航空航天、军事、网络", freertos: "IoT、嵌入式、消费电子", qnx: "汽车、医疗、工业", preempt_rt: "机器人、工业自动化", category: "应用" },
];

const rtosMeta: Record<RTOS, { name: string; icon: string; color: string; desc: string }> = {
  vxworks: { name: "VxWorks", icon: "🛩️", color: "#3b82f6", desc: "航空航天与国防工业首选" },
  freertos: { name: "FreeRTOS", icon: "📦", color: "#10b981", desc: "最流行的开源 RTOS" },
  qnx: { name: "QNX Neutrino", icon: "🚗", color: "#f59e0b", desc: "微内核安全关键系统" },
  preempt_rt: { name: "PREEMPT_RT", icon: "🐧", color: "#8b5cf6", desc: "Linux 实时补丁方案" },
};

const categories = [...new Set(features.map((f) => f.category))];

function renderValue(v: string | boolean) {
  if (v === true) return <CheckCircle className="w-4 h-4 text-emerald-500" />;
  if (v === false) return <XCircle className="w-4 h-4 text-red-400" />;
  return <span className="text-sm text-gray-700 dark:text-gray-300">{v}</span>;
}

export default function RTOSComparison() {
  const [selected, setSelected] = useState<RTOS[]>(["vxworks", "freertos", "qnx", "preempt_rt"]);
  const [expandedCat, setExpandedCat] = useState<string | null>(categories[0]);

  const toggleRTOS = (r: RTOS) => {
    setSelected((prev) =>
      prev.includes(r) ? prev.filter((x) => x !== r) : [...prev, r]
    );
  };

  return (
    <div className="w-full space-y-6 p-4 bg-white dark:bg-gray-900 rounded-xl">
      <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
        <Cpu className="w-5 h-5 text-indigo-500" />
        RTOS 对比分析
      </h3>

      <div className="flex flex-wrap gap-2">
        {(Object.keys(rtosMeta) as RTOS[]).map((r) => {
          const meta = rtosMeta[r];
          const active = selected.includes(r);
          return (
            <button
              key={r}
              onClick={() => toggleRTOS(r)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all border-2 ${
                active
                  ? "border-current bg-opacity-10"
                  : "border-gray-200 dark:border-gray-700 opacity-50"
              }`}
              style={active ? { borderColor: meta.color, backgroundColor: meta.color + "15" } : {}}
            >
              <span>{meta.icon}</span>
              <span style={active ? { color: meta.color } : {}}>{meta.name}</span>
            </button>
          );
        })}
      </div>

      <div className="space-y-2">
        {categories.map((cat) => {
          const catFeatures = features.filter((f) => f.category === cat);
          const expanded = expandedCat === cat;

          return (
            <div key={cat} className="border dark:border-gray-700 rounded-lg overflow-hidden">
              <button
                onClick={() => setExpandedCat(expanded ? null : cat)}
                className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors"
              >
                <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                  {cat}
                </span>
                {expanded ? (
                  <ChevronUp className="w-4 h-4 text-gray-400" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                )}
              </button>

              <AnimatePresence>
                {expanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b dark:border-gray-700">
                            <th className="text-left text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-2 w-36">
                              特性
                            </th>
                            {selected.map((r) => (
                              <th
                                key={r}
                                className="text-left text-xs font-medium px-4 py-2"
                                style={{ color: rtosMeta[r].color }}
                              >
                                {rtosMeta[r].name}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {catFeatures.map((feat, i) => (
                            <tr
                              key={feat.name}
                              className={i % 2 === 0 ? "bg-white dark:bg-gray-900" : "bg-gray-50 dark:bg-gray-800/50"}
                            >
                              <td className="px-4 py-2 text-xs font-medium text-gray-600 dark:text-gray-400">
                                {feat.name}
                              </td>
                              {selected.map((r) => (
                                <td key={r} className="px-4 py-2">
                                  {renderValue(feat[r])}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {selected.map((r) => {
          const meta = rtosMeta[r];
          return (
            <motion.div
              key={r}
              layout
              className="p-3 rounded-lg border dark:border-gray-700"
              style={{ borderColor: meta.color + "60" }}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-lg">{meta.icon}</span>
                <span className="text-sm font-bold" style={{ color: meta.color }}>
                  {meta.name}
                </span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400">{meta.desc}</p>
            </motion.div>
          );
        })}
      </div>

      <div className="p-4 rounded-lg bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 space-y-3">
        <h4 className="text-sm font-bold text-indigo-700 dark:text-indigo-300">选型建议</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex items-start gap-2">
            <span className="text-lg">🛩️</span>
            <div>
              <div className="text-xs font-semibold text-blue-600 dark:text-blue-400">VxWorks</div>
              <p className="text-xs text-gray-600 dark:text-gray-400">安全关键系统（航空、国防），需 DO-178C 认证，预算充足</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-lg">📦</span>
            <div>
              <div className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">FreeRTOS</div>
              <p className="text-xs text-gray-600 dark:text-gray-400">资源受限的 MCU/IoT 设备，需免费开源方案，快速原型开发</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-lg">🚗</span>
            <div>
              <div className="text-xs font-semibold text-amber-600 dark:text-amber-400">QNX</div>
              <p className="text-xs text-gray-600 dark:text-gray-400">汽车电子（ISO 26262）、医疗设备，需微内核故障隔离</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-lg">🐧</span>
            <div>
              <div className="text-xs font-semibold text-violet-600 dark:text-violet-400">PREEMPT_RT</div>
              <p className="text-xs text-gray-600 dark:text-gray-400">已有 Linux 生态、需完整驱动支持、工业机器人/自动化</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
