"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Cpu,
  MemoryStick,
  HardDrive,
  Lock,
  Wifi,
  ArrowRight,
  CheckCircle,
  Terminal,
} from "lucide-react";

interface Problem {
  id: string;
  label: string;
  icon: React.ReactNode;
  color: string;
  symptoms: string[];
  tools: Tool[];
}

interface Tool {
  name: string;
  cmd: string;
  purpose: string;
  example: string;
}

const problems: Problem[] = [
  {
    id: "cpu",
    label: "CPU 瓶颈",
    icon: <Cpu className="w-5 h-5" />,
    color: "from-blue-500 to-cyan-500",
    symptoms: ["高 CPU 使用率", "运行队列长", "响应延迟增加"],
    tools: [
      {
        name: "top / htop",
        cmd: "top -H",
        purpose: "查看哪个线程消耗 CPU 最多",
        example: "top -H -p <PID>",
      },
      {
        name: "mpstat",
        cmd: "mpstat -P ALL 1",
        purpose: "查看每个 CPU 核心的使用情况",
        example: "mpstat -P ALL 2 5",
      },
      {
        name: "perf stat",
        cmd: "perf stat -p <PID>",
        purpose: "统计 IPC、缓存未命中等硬件事件",
        example: "perf stat -e cycles,instructions,cache-misses ./prog",
      },
      {
        name: "perf record",
        cmd: "perf record -F 999 -g -p <PID>",
        purpose: "采样记录热点函数调用栈",
        example: "perf record -F 999 -g ./prog && perf report",
      },
      {
        name: "火焰图",
        cmd: "perf script | flamegraph.pl",
        purpose: "可视化 CPU 热点函数分布",
        example: "perf script | stackcollapse-perf.pl | flamegraph.pl > cpu.svg",
      },
    ],
  },
  {
    id: "memory",
    label: "内存瓶颈",
    icon: <MemoryStick className="w-5 h-5" />,
    color: "from-purple-500 to-pink-500",
    symptoms: ["频繁 swap 活动", "大量页故障", "OOM Killer 触发"],
    tools: [
      {
        name: "free",
        cmd: "free -h",
        purpose: "查看内存和 swap 使用概况",
        example: "free -h && cat /proc/meminfo",
      },
      {
        name: "vmstat",
        cmd: "vmstat 1",
        purpose: "观察 si/so（swap in/out）活动",
        example: "vmstat 1 10 | awk '{print $7,$8}'",
      },
      {
        name: "perf stat",
        cmd: "perf stat -e page-faults",
        purpose: "统计页故障次数",
        example: "perf stat -e minor-faults,major-faults ./prog",
      },
      {
        name: "valgrind massif",
        cmd: "valgrind --tool=massif ./prog",
        purpose: "分析堆内存分配热点",
        example: "ms_print massif.out.12345",
      },
    ],
  },
  {
    id: "io",
    label: "I/O 瓶颈",
    icon: <HardDrive className="w-5 h-5" />,
    color: "from-amber-500 to-orange-500",
    symptoms: ["高 iowait", "磁盘利用率高", "读写延迟大"],
    tools: [
      {
        name: "iostat",
        cmd: "iostat -xz 2",
        purpose: "查看磁盘 IOPS、吞吐量、延迟、利用率",
        example: "iostat -xz 1 5",
      },
      {
        name: "iotop",
        cmd: "iotop -oP",
        purpose: "查看进程级 I/O 使用",
        example: "sudo iotop -oP -d 2",
      },
      {
        name: "pidstat",
        cmd: "pidstat -d 1",
        purpose: "进程级 I/O 统计",
        example: "pidstat -d -p <PID> 1",
      },
      {
        name: "strace",
        cmd: "strace -e trace=read,write -T",
        purpose: "跟踪 I/O 系统调用及耗时",
        example: "strace -e trace=read,write -T -p <PID>",
      },
    ],
  },
  {
    id: "lock",
    label: "锁竞争",
    icon: <Lock className="w-5 h-5" />,
    color: "from-red-500 to-rose-500",
    symptoms: ["高上下文切换", "CPU 利用率低但延迟高", "线程等待"],
    tools: [
      {
        name: "vmstat",
        cmd: "vmstat 1",
        purpose: "查看 cs（上下文切换/秒）",
        example: "vmstat 1 | awk '{print $12}'",
      },
      {
        name: "pidstat -w",
        cmd: "pidstat -w 1",
        purpose: "查看进程级上下文切换",
        example: "pidstat -w -p <PID> 1",
      },
      {
        name: "perf lock",
        cmd: "perf lock record -p <PID>",
        purpose: "记录锁竞争事件",
        example: "perf lock record -p <PID> sleep 10 && perf lock report",
      },
      {
        name: "Off-CPU 火焰图",
        cmd: "perf record -e sched:sched_switch -g",
        purpose: "分析线程等待时间分布",
        example: "offcputime.pl -p <PID> 10 | flamegraph.pl > offcpu.svg",
      },
    ],
  },
  {
    id: "network",
    label: "网络瓶颈",
    icon: <Wifi className="w-5 h-5" />,
    color: "from-green-500 to-emerald-500",
    symptoms: ["网络延迟高", "丢包", "连接数满"],
    tools: [
      {
        name: "sar",
        cmd: "sar -n DEV 1",
        purpose: "查看网络接口吞吐量",
        example: "sar -n DEV,EDEV 1 5",
      },
      {
        name: "ss",
        cmd: "ss -s",
        purpose: "查看 TCP 连接状态统计",
        example: "ss -tnp | grep <PORT>",
      },
      {
        name: "tcpdump",
        cmd: "tcpdump -i eth0 -c 100",
        purpose: "抓包分析网络流量",
        example: "tcpdump -i any port 80 -nn",
      },
      {
        name: "perf trace",
        cmd: "perf trace -e 'net:*' -p <PID>",
        purpose: "跟踪网络相关内核事件",
        example: "perf trace -e 'tcp:*' -p <PID>",
      },
    ],
  },
];

export default function PerfToolsWorkflow() {
  const [selected, setSelected] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<number | null>(null);
  const problem = problems.find((p) => p.id === selected);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6 text-center flex items-center justify-center gap-2">
        <Search className="w-7 h-7 text-indigo-600" />
        性能分析工具导航
      </h3>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
        {problems.map((p) => (
          <motion.button
            key={p.id}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              setSelected(p.id === selected ? null : p.id);
              setActiveTool(null);
            }}
            className={`flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all ${
              selected === p.id
                ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30 shadow-lg"
                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
            }`}
          >
            <div
              className={`w-10 h-10 rounded-lg bg-gradient-to-br ${p.color} flex items-center justify-center text-white`}
            >
              {p.icon}
            </div>
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">
              {p.label}
            </span>
          </motion.button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {problem && (
          <motion.div
            key={problem.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-5"
          >
            <div className="flex items-center gap-3 mb-4">
              <div
                className={`w-8 h-8 rounded-lg bg-gradient-to-br ${problem.color} flex items-center justify-center text-white`}
              >
                {problem.icon}
              </div>
              <div>
                <h4 className="font-bold text-lg text-slate-800 dark:text-slate-100">
                  {problem.label} — 诊断流程
                </h4>
                <div className="flex gap-2 mt-1">
                  {problem.symptoms.map((s, i) => (
                    <span
                      key={i}
                      className="text-xs bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 px-2 py-0.5 rounded-full"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-2">
              {problem.tools.map((tool, i) => (
                <motion.div
                  key={tool.name}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.08 }}
                >
                  <button
                    onClick={() => setActiveTool(activeTool === i ? null : i)}
                    className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all ${
                      activeTool === i
                        ? "bg-indigo-50 dark:bg-indigo-900/30 border border-indigo-300 dark:border-indigo-600"
                        : "bg-slate-50 dark:bg-slate-700/50 hover:bg-slate-100 dark:hover:bg-slate-700 border border-transparent"
                    }`}
                  >
                    <span className="w-6 h-6 rounded-full bg-indigo-100 dark:bg-indigo-800 text-indigo-600 dark:text-indigo-300 flex items-center justify-center text-xs font-bold">
                      {i + 1}
                    </span>
                    <span className="flex-1 text-left font-semibold text-sm text-slate-700 dark:text-slate-200">
                      {tool.name}
                    </span>
                    <code className="text-xs bg-slate-200 dark:bg-slate-600 px-2 py-0.5 rounded font-mono text-slate-600 dark:text-slate-300">
                      {tool.cmd}
                    </code>
                    <ArrowRight
                      className={`w-4 h-4 text-slate-400 transition-transform ${
                        activeTool === i ? "rotate-90" : ""
                      }`}
                    />
                  </button>

                  <AnimatePresence>
                    {activeTool === i && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="ml-9 mt-2 p-3 bg-slate-50 dark:bg-slate-700/30 rounded-lg space-y-2">
                          <div className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                            <span className="text-sm text-slate-600 dark:text-slate-300">
                              {tool.purpose}
                            </span>
                          </div>
                          <div className="flex items-start gap-2">
                            <Terminal className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
                            <code className="text-xs bg-slate-200 dark:bg-slate-600 px-2 py-1 rounded font-mono text-slate-700 dark:text-slate-200 break-all">
                              {tool.example}
                            </code>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!selected && (
        <div className="text-center py-8 text-slate-400 dark:text-slate-500">
          <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>选择一个瓶颈类型，查看推荐的诊断工具和步骤</p>
        </div>
      )}
    </div>
  );
}
