"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Code, Cpu, Zap, CheckCircle, AlertTriangle, Play, RotateCcw, ChevronRight } from "lucide-react";

interface VerificationCheck {
  name: string;
  desc: string;
  status: "pending" | "running" | "pass" | "fail";
  detail: string;
}

interface PipelineStage {
  id: string;
  label: string;
  icon: React.ReactNode;
  color: string;
  bg: string;
  border: string;
  description: string;
  details: string[];
}

const stages: PipelineStage[] = [
  {
    id: "user",
    label: "用户程序",
    icon: <Code className="w-5 h-5" />,
    color: "text-blue-500",
    bg: "bg-blue-50 dark:bg-blue-950/30",
    border: "border-blue-300 dark:border-blue-700",
    description: "编写eBPF C程序，定义内核事件处理逻辑",
    details: [
      "#include <linux/bpf.h>",
      "SEC(\"tracepoint/syscalls/sys_enter_write\")",
      "int handle_write(struct trace_event_raw_sys_enter *ctx) {",
      "  bpf_printk(\"write called\");",
      "  return 0;",
      "}",
    ],
  },
  {
    id: "verifier",
    label: "验证器",
    icon: <Shield className="w-5 h-5" />,
    color: "text-amber-500",
    bg: "bg-amber-50 dark:bg-amber-950/30",
    border: "border-amber-300 dark:border-amber-700",
    description: "内核验证器静态分析程序安全性，确保不会导致内核崩溃",
    details: [
      "• 检查所有代码路径是否可达",
      "• 验证程序大小不超过指令限制",
      "• 确认所有内存访问在边界内",
      "• 检测不可达代码和死循环",
      "• 验证寄存器和栈状态一致性",
    ],
  },
  {
    id: "jit",
    label: "JIT编译",
    icon: <Cpu className="w-5 h-5" />,
    color: "text-green-500",
    bg: "bg-green-50 dark:bg-green-950/30",
    border: "border-green-300 dark:border-green-700",
    description: "将验证通过的字节码编译为本地机器指令，优化执行性能",
    details: [
      "• BPF字节码 → x86_64/ARM64机器码",
      "• 内联优化常用辅助函数",
      "• 寄存器分配与指令调度",
      "• 跳转目标重定位",
      "• 生成可执行的内核代码页",
    ],
  },
  {
    id: "hook",
    label: "内核挂载",
    icon: <Zap className="w-5 h-5" />,
    color: "text-purple-500",
    bg: "bg-purple-50 dark:bg-purple-950/30",
    border: "border-purple-300 dark:border-purple-700",
    description: "将JIT编译后的程序挂载到指定的内核hook点，开始事件处理",
    details: [
      "• 支持挂载点: kprobe, tracepoint, XDP, TC",
      "• 程序与hook点关联",
      "• 事件触发时执行eBPF程序",
      "• 通过BPF Map与用户空间通信",
      "• 支持热替换，无需重启",
    ],
  },
];

const initialChecks: VerificationCheck[] = [
  { name: "循环检测", desc: "确认程序中无无限循环", status: "pending", detail: "CFG分析：所有回边均有界" },
  { name: "边界检查", desc: "所有内存访问在有效范围内", status: "pending", detail: "指针运算追踪 + 偏移量验证" },
  { name: "类型检查", desc: "寄存器使用类型安全", status: "pending", detail: "每条指令的src/dst类型匹配" },
  { name: "终止性", desc: "程序必须在有限步骤内返回", status: "pending", detail: "指令数 ≤ MAX_BPF_INSNS (4096)" },
  { name: "辅助函数", desc: "仅调用允许的内核辅助函数", status: "pending", detail: "白名单校验 bpf_map_lookup, bpf_probe_read 等" },
];

const fadeSlide = {
  initial: { opacity: 0, x: 30 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -30 },
  transition: { duration: 0.35 },
};

export default function EBPFPipeline() {
  const [current, setCurrent] = useState(-1);
  const [checks, setChecks] = useState<VerificationCheck[]>(initialChecks);
  const [verifying, setVerifying] = useState(false);
  const [done, setDone] = useState(false);

  const active = current >= 0 && current < stages.length ? stages[current] : null;

  const reset = () => {
    setCurrent(-1);
    setChecks(initialChecks);
    setVerifying(false);
    setDone(false);
  };

  const runVerification = () => {
    setVerifying(true);
    setChecks(initialChecks.map((c) => ({ ...c, status: "pending" as const })));

    initialChecks.forEach((_, i) => {
      setTimeout(() => {
        setChecks((prev) =>
          prev.map((c, ci) => (ci === i ? { ...c, status: "running" as const } : c))
        );
      }, i * 700 + 100);

      setTimeout(() => {
        setChecks((prev) =>
          prev.map((c, ci) => (ci === i ? { ...c, status: "pass" as const } : c))
        );
        if (i === initialChecks.length - 1) {
          setTimeout(() => setVerifying(false), 400);
        }
      }, i * 700 + 600);
    });
  };

  const next = () => {
    if (current === 1 && !verifying) {
      runVerification();
    }
    if (current < stages.length - 1) {
      setCurrent((c) => c + 1);
    } else {
      setDone(true);
    }
  };

  const start = () => {
    reset();
    setCurrent(0);
  };

  return (
    <div className="w-full max-w-3xl mx-auto p-6 bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm my-6">
      <h3 className="text-xl font-bold mb-1 text-gray-900 dark:text-gray-100">eBPF 程序执行流水线</h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">从用户态程序编写到内核态挂载的完整流程</p>

      <div className="flex items-center gap-1 mb-8">
        {stages.map((s, i) => (
          <div key={s.id} className="flex items-center">
            <button
              onClick={() => { reset(); setCurrent(i); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                i === current
                  ? `${s.bg} ${s.color} ${s.border} border`
                  : i < current
                  ? "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600"
                  : "bg-gray-50 dark:bg-gray-800/50 text-gray-400 dark:text-gray-500 border border-gray-200 dark:border-gray-700"
              }`}
            >
              {s.icon}
              <span className="hidden sm:inline">{s.label}</span>
            </button>
            {i < stages.length - 1 && (
              <ChevronRight className="w-4 h-4 text-gray-300 dark:text-gray-600 mx-0.5" />
            )}
          </div>
        ))}
      </div>

      {current === -1 && !done && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center py-12 gap-4"
        >
          <Code className="w-12 h-12 text-gray-300 dark:text-gray-600" />
          <p className="text-gray-500 dark:text-gray-400 text-sm">点击开始，逐步了解 eBPF 程序的生命周期</p>
          <button
            onClick={start}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium transition-colors"
          >
            <Play className="w-4 h-4" /> 开始演示
          </button>
        </motion.div>
      )}

      <AnimatePresence mode="wait">
        {active && !done && (
          <motion.div key={active.id} {...fadeSlide} className="space-y-5">
            <div className={`rounded-xl p-5 border ${active.border} ${active.bg}`}>
              <div className="flex items-center gap-2 mb-3">
                <span className={active.color}>{active.icon}</span>
                <h4 className={`text-lg font-bold ${active.color}`}>阶段 {current + 1}: {active.label}</h4>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">{active.description}</p>

              {current === 0 && (
                <div className="bg-gray-900 dark:bg-gray-950 rounded-lg p-4 font-mono text-xs text-green-400 overflow-x-auto">
                  {active.details.map((line, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.08 }}
                      className="whitespace-pre"
                    >
                      {line}
                    </motion.div>
                  ))}
                </div>
              )}

              {current === 1 && (
                <div className="space-y-2">
                  {checks.map((ck, i) => (
                    <motion.div
                      key={ck.name}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.1 }}
                      className={`flex items-center gap-3 p-3 rounded-lg border text-sm transition-colors ${
                        ck.status === "pass"
                          ? "bg-green-50 dark:bg-green-950/30 border-green-300 dark:border-green-700"
                          : ck.status === "running"
                          ? "bg-amber-50 dark:bg-amber-950/30 border-amber-300 dark:border-amber-700"
                          : "bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700"
                      }`}
                    >
                      {ck.status === "pass" ? (
                        <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                      ) : ck.status === "running" ? (
                        <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }}>
                          <Shield className="w-4 h-4 text-amber-500 flex-shrink-0" />
                        </motion.div>
                      ) : (
                        <AlertTriangle className="w-4 h-4 text-gray-400 flex-shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <span className="font-medium text-gray-800 dark:text-gray-200">{ck.name}</span>
                        <span className="text-gray-500 dark:text-gray-400 ml-2">{ck.desc}</span>
                      </div>
                      {ck.status === "pass" && (
                        <span className="text-xs text-green-600 dark:text-green-400">{ck.detail}</span>
                      )}
                    </motion.div>
                  ))}
                </div>
              )}

              {current === 2 && (
                <div className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                  <motion.div
                    animate={{ rotate: [0, 360] }}
                    transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                    className="text-green-500"
                  >
                    <Cpu className="w-8 h-8" />
                  </motion.div>
                  <div className="space-y-1">
                    {active.details.map((d, i) => (
                      <motion.p
                        key={i}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: i * 0.15 }}
                        className="text-sm text-gray-700 dark:text-gray-300"
                      >
                        {d}
                      </motion.p>
                    ))}
                  </div>
                </div>
              )}

              {current === 3 && (
                <div className="grid grid-cols-2 gap-3">
                  {active.details.map((d, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.1 }}
                      className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 text-sm text-gray-700 dark:text-gray-300"
                    >
                      {d}
                    </motion.div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex items-center justify-between">
              <button
                onClick={reset}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" /> 重置
              </button>
              <button
                onClick={next}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium transition-colors"
              >
                {current < stages.length - 1 ? "下一步" : "完成"} <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {done && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex flex-col items-center py-10 gap-4"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 12 }}
          >
            <Zap className="w-14 h-14 text-purple-500" />
          </motion.div>
          <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100">eBPF 程序已挂载!</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center max-w-md">
            程序已通过验证器检查，JIT编译为本地指令，并挂载到内核hook点。
            当对应事件触发时，eBPF程序将在内核态高效执行。
          </p>
          <div className="flex gap-3 mt-2">
            <button
              onClick={reset}
              className="px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            >
              重新演示
            </button>
          </div>
        </motion.div>
      )}
    </div>
  );
}
