"use client";

import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  Cpu,
  MemoryStick,
  HardDrive,
  Lock,
  Search,
  CheckCircle2,
  Lightbulb,
} from "lucide-react";

interface Symptom {
  id: string;
  label: string;
  category: "cpu" | "memory" | "io" | "lock";
}

interface Diagnosis {
  type: string;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
  label: string;
  description: string;
  solutions: string[];
  tools: string[];
}

const symptoms: Symptom[] = [
  { id: "cpu_high", label: "CPU 使用率持续 > 80%", category: "cpu" },
  { id: "runqueue", label: "运行队列长度 > CPU 核数", category: "cpu" },
  { id: "ipc_low", label: "IPC (每周期指令数) < 1", category: "cpu" },
  { id: "branch_miss", label: "分支预测失败率 > 5%", category: "cpu" },
  { id: "sys_high", label: "内核态 (%sys) 占比高", category: "cpu" },
  { id: "swap_active", label: "频繁 swap 活动 (si/so > 0)", category: "memory" },
  { id: "page_fault", label: "大量页故障 (page-faults)", category: "memory" },
  { id: "oom_kill", label: "OOM Killer 触发", category: "memory" },
  { id: "cache_miss", label: "缓存未命中率 > 10%", category: "memory" },
  { id: "mem_high", label: "内存使用率 > 85%", category: "memory" },
  { id: "iowait_high", label: "iowait > 20%", category: "io" },
  { id: "disk_util", label: "磁盘利用率 %util > 70%", category: "io" },
  { id: "io_latency", label: "I/O 延迟 > 10ms", category: "io" },
  { id: "io_small", label: "频繁小 I/O 请求", category: "io" },
  { id: "cs_high", label: "上下文切换数异常高", category: "lock" },
  { id: "low_cpu_high_latency", label: "CPU 利用率低但延迟高", category: "lock" },
  { id: "thread_wait", label: "线程长时间等待", category: "lock" },
  { id: "futex_contention", label: "futex/锁等待时间长", category: "lock" },
];

const diagnoses: Record<string, Diagnosis> = {
  cpu: {
    type: "cpu",
    icon: <Cpu className="w-6 h-6" />,
    color: "text-blue-600",
    bgColor: "bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700",
    label: "CPU 瓶颈",
    description: "CPU 资源饱和，计算能力不足以处理当前负载。",
    solutions: [
      "优化算法复杂度（O(n²) → O(n log n)）",
      "利用缓存局部性，减少 cache miss",
      "使用 SIMD 向量化指令加速计算",
      "启用编译器优化 (-O2/-O3/PGO)",
      "减少不必要的系统调用",
      "水平扩展：增加 CPU 核心或分布式处理",
    ],
    tools: ["perf stat", "perf record + report", "火焰图", "top -H", "mpstat"],
  },
  memory: {
    type: "memory",
    icon: <MemoryStick className="w-6 h-6" />,
    color: "text-purple-600",
    bgColor: "bg-purple-50 dark:bg-purple-900/30 border-purple-300 dark:border-purple-700",
    label: "内存瓶颈",
    description: "内存不足导致频繁换页或 OOM，严重影响性能。",
    solutions: [
      "检查内存泄漏（valgrind, AddressSanitizer）",
      "优化数据结构减小内存占用",
      "使用对象池/内存池减少分配开销",
      "降低 vm.swappiness 减少 swap",
      "增加物理内存或优化工作集大小",
      "使用 mmap 替代频繁 read/write",
    ],
    tools: ["free -h", "vmstat", "valgrind --tool=massif", "perf stat page-faults", "smem"],
  },
  io: {
    type: "io",
    icon: <HardDrive className="w-6 h-6" />,
    color: "text-amber-600",
    bgColor: "bg-amber-50 dark:bg-amber-900/30 border-amber-300 dark:border-amber-700",
    label: "I/O 瓶颈",
    description: "磁盘 I/O 成为瓶颈，读写延迟高或吞吐不足。",
    solutions: [
      "批量 I/O：合并小请求减少系统调用",
      "异步 I/O：使用 io_uring 避免阻塞",
      "零拷贝：sendfile/splice 减少数据拷贝",
      "增大预读 (readahead) 减少 I/O 次数",
      "使用 SSD 替代 HDD",
      "调整 I/O 调度器（SSD 用 none/mq-deadline）",
    ],
    tools: ["iostat -xz", "iotop", "pidstat -d", "strace -e trace=read,write", "blktrace"],
  },
  lock: {
    type: "lock",
    icon: <Lock className="w-6 h-6" />,
    color: "text-red-600",
    bgColor: "bg-red-50 dark:bg-red-900/30 border-red-300 dark:border-red-700",
    label: "锁竞争",
    description: "线程间锁竞争导致大量上下文切换和等待，CPU 利用率低但延迟高。",
    solutions: [
      "减小锁粒度：全局锁 → 分段锁 → 对象锁",
      "读写锁分离：rwlock 替代 mutex",
      "无锁编程：CAS 原子操作、无锁队列",
      "RCU：读多写少场景下读者无锁",
      "线程本地存储 (TLS) 消除共享",
      "调整线程数，避免过度并发",
    ],
    tools: ["vmstat (cs)", "pidstat -w", "perf lock record/report", "Off-CPU 火焰图", "strace -e futex"],
  },
};

function findDiagnosis(selectedIds: string[]): Diagnosis | null {
  const counts: Record<string, number> = { cpu: 0, memory: 0, io: 0, lock: 0 };
  for (const id of selectedIds) {
    const s = symptoms.find((sy) => sy.id === id);
    if (s) counts[s.category]++;
  }
  const max = Math.max(...Object.values(counts));
  if (max === 0) return null;
  const winner = Object.entries(counts).find(([, v]) => v === max)?.[0];
  return winner ? diagnoses[winner] : null;
}

export default function BottleneckIdentifier() {
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };

  const diagnosis = useMemo(() => findDiagnosis(Array.from(selected)), [selected]);

  const categoryCounts = useMemo(() => {
    const c: Record<string, number> = { cpu: 0, memory: 0, io: 0, lock: 0 };
    selected.forEach((id) => {
      const s = symptoms.find((sy) => sy.id === id);
      if (s) c[s.category]++;
    });
    return c;
  }, [selected]);

  const categoryLabels: Record<string, { label: string; color: string }> = {
    cpu: { label: "CPU", color: "bg-blue-500" },
    memory: { label: "内存", color: "bg-purple-500" },
    io: { label: "I/O", color: "bg-amber-500" },
    lock: { label: "锁", color: "bg-red-500" },
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2 text-center flex items-center justify-center gap-2">
        <Search className="w-7 h-7 text-rose-600" />
        瓶颈识别器
      </h3>
      <p className="text-center text-sm text-slate-500 dark:text-slate-400 mb-6">
        选择你观察到的症状，工具会自动判断瓶颈类型并推荐解决方案
      </p>

      <div className="flex gap-2 mb-4 justify-center">
        {Object.entries(categoryLabels).map(([key, { label, color }]) => (
          <span
            key={key}
            className={`text-xs px-3 py-1 rounded-full text-white ${color} flex items-center gap-1`}
          >
            {label}: {categoryCounts[key]}
          </span>
        ))}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-6">
        {symptoms.map((s) => {
          const active = selected.has(s.id);
          return (
            <motion.button
              key={s.id}
              whileTap={{ scale: 0.97 }}
              onClick={() => toggle(s.id)}
              className={`flex items-center gap-3 p-3 rounded-lg border-2 text-left transition-all ${
                active
                  ? "border-rose-400 bg-rose-50 dark:bg-rose-900/20 dark:border-rose-600"
                  : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-slate-300 dark:hover:border-slate-600"
              }`}
            >
              <div
                className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                  active
                    ? "border-rose-500 bg-rose-500"
                    : "border-slate-300 dark:border-slate-600"
                }`}
              >
                {active && <CheckCircle2 className="w-4 h-4 text-white" />}
              </div>
              <span className="text-sm text-slate-700 dark:text-slate-200">
                {s.label}
              </span>
            </motion.button>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        {diagnosis && (
          <motion.div
            key={diagnosis.type}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            className={`rounded-xl border-2 p-5 ${diagnosis.bgColor}`}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className={`${diagnosis.color}`}>{diagnosis.icon}</div>
              <div>
                <h4 className={`text-xl font-bold ${diagnosis.color}`}>
                  诊断结果: {diagnosis.label}
                </h4>
                <p className="text-sm text-slate-600 dark:text-slate-300">
                  {diagnosis.description}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-semibold text-sm text-slate-700 dark:text-slate-200 flex items-center gap-1 mb-2">
                  <Lightbulb className="w-4 h-4 text-amber-500" />
                  优化建议
                </h5>
                <ul className="space-y-1.5">
                  {diagnosis.solutions.map((s, i) => (
                    <motion.li
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="text-sm text-slate-600 dark:text-slate-300 flex items-start gap-2"
                    >
                      <span className="text-emerald-500 mt-0.5">✓</span>
                      {s}
                    </motion.li>
                  ))}
                </ul>
              </div>
              <div>
                <h5 className="font-semibold text-sm text-slate-700 dark:text-slate-200 flex items-center gap-1 mb-2">
                  <Search className="w-4 h-4 text-blue-500" />
                  推荐工具
                </h5>
                <div className="flex flex-wrap gap-2">
                  {diagnosis.tools.map((t, i) => (
                    <motion.span
                      key={i}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.05 }}
                      className="text-xs bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 px-2.5 py-1 rounded-full font-mono text-slate-700 dark:text-slate-200"
                    >
                      {t}
                    </motion.span>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {selected.size === 0 && (
        <div className="text-center py-6 text-slate-400 dark:text-slate-500">
          <AlertTriangle className="w-10 h-10 mx-auto mb-2 opacity-50" />
          <p className="text-sm">请勾选观察到的症状以获取诊断建议</p>
        </div>
      )}

      {selected.size > 0 && !diagnosis && (
        <div className="text-center py-4 text-slate-400">
          <p className="text-sm">请多选几个症状以确定瓶颈类型</p>
        </div>
      )}
    </div>
  );
}
