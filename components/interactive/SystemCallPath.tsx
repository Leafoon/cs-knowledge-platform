"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Terminal, Cpu, Shield } from "lucide-react";

type Mode = "User" | "Hardware" | "Kernel";

interface TraceStep {
  id: number;
  mode: Mode;
  title: string;
  code: string;
  description: string;
}

const traceSteps: TraceStep[] = [
  {
    id: 1,
    mode: "User",
    title: 'write(1, "hello", 5)',
    code: `int n = write(1, "hello", 5);`,
    description: "用户程序调用 write 系统调用，传入 fd=1(stdout)、缓冲区地址、字节数 5。",
  },
  {
    id: 2,
    mode: "User",
    title: "usys.S wrapper",
    code: `li a7, SYS_write   # a7 = 16\necall               # trap into kernel`,
    description: "C 库包装函数将系统调用号 16 放入 a7，然后执行 ecall 指令陷入内核。",
  },
  {
    id: 3,
    mode: "Hardware",
    title: "ecall trap",
    code: `sepc  ← PC\nscause ← 8        # ecall from U-mode\nstvec → trap entry\nmode  → S-mode`,
    description: "硬件保存 PC 到 sepc，设置 scause=8（U-mode ecall），切换到 S-mode，跳转到 stvec 指向的入口。",
  },
  {
    id: 4,
    mode: "Kernel",
    title: "uservec (trampoline.S)",
    code: `# save user registers to trapframe\ncsrr t0, sscratch\nsd ra, 0(t0)\nsd sp, 8(t0)\n... # save all 32 regs`,
    description: "陷入入口 uservec 将所有用户寄存器保存到当前进程的 trapframe 结构中。",
  },
  {
    id: 5,
    mode: "Kernel",
    title: "usertrap()",
    code: `void usertrap() {\n  r_scause() == 8  // ecall\n  syscall();\n}`,
    description: "读取 scause=8 判断为 ecall，调用 syscall() 处理系统调用。",
  },
  {
    id: 6,
    mode: "Kernel",
    title: "syscall()",
    code: `num = p->trapframe->a7;  // 16\nsyscalls[num]();         // sys_write`,
    description: "从 trapframe 中读取 a7=16，查系统调用表 syscalls[16] 得到 sys_write。",
  },
  {
    id: 7,
    mode: "Kernel",
    title: "argfd + filewrite",
    code: `argfd(0, 0, &f);   // fd=1 → file\nfilewrite(f, buf, n); // write to console`,
    description: "argfd 将 fd=1 解析为 file 结构，filewrite 将数据写入控制台。",
  },
  {
    id: 8,
    mode: "Kernel",
    title: "usertrapret()",
    code: `p->trapframe->a0 = 5;  // return value\nw_stvec(USERVEC);\nuserret(trapframe, satp);`,
    description: "设置返回值 a0=5 写入 trapframe，准备 satp、stvec，调用 userret 返回用户态。",
  },
  {
    id: 9,
    mode: "Kernel",
    title: "userret (trampoline.S)",
    code: `# restore user registers from trapframe\nld ra, 0(t0)\nld sp, 8(t0)\n...\nsret                    // return to U-mode`,
    description: "从 trapframe 恢复所有用户寄存器，执行 sret 切回 U-mode 并恢复 PC。",
  },
  {
    id: 10,
    mode: "User",
    title: "returns a0 = 5",
    code: `// write() returns 5\n// 5 bytes written to stdout`,
    description: "用户程序恢复执行，write() 返回 5 表示成功写入 5 个字节。",
  },
];

const modeConfig: Record<Mode, { color: string; bg: string; border: string; icon: React.ReactNode }> = {
  User: {
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    border: "border-blue-500/30",
    icon: <Terminal className="w-4 h-4" />,
  },
  Hardware: {
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    icon: <Cpu className="w-4 h-4" />,
  },
  Kernel: {
    color: "text-rose-400",
    bg: "bg-rose-500/10",
    border: "border-rose-500/30",
    icon: <Shield className="w-4 h-4" />,
  },
};

const circleColors: Record<Mode, string> = {
  User: "bg-blue-500",
  Hardware: "bg-amber-500",
  Kernel: "bg-rose-500",
};

const lineColors: Record<Mode, string> = {
  User: "bg-blue-500/30",
  Hardware: "bg-amber-500/30",
  Kernel: "bg-rose-500/30",
};

export default function SystemCallPath() {
  const [activeStep, setActiveStep] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentPlay, setCurrentPlay] = useState(-1);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setCurrentPlay((prev) => {
          if (prev >= traceSteps.length - 1) {
            setPlaying(false);
            return prev;
          }
          const next = prev + 1;
          stepRefs.current[next]?.scrollIntoView({ behavior: "smooth", block: "center" });
          setActiveStep(traceSteps[next].id);
          return next;
        });
      }, 1200);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing]);

  const handlePlay = () => {
    if (currentPlay >= traceSteps.length - 1) {
      setCurrentPlay(-1);
      setActiveStep(null);
    }
    setPlaying(true);
  };

  const handlePause = () => setPlaying(false);

  const handleReset = () => {
    setPlaying(false);
    setCurrentPlay(-1);
    setActiveStep(null);
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-gray-900 to-gray-950 rounded-xl shadow-lg text-gray-100">
      <h3 className="text-2xl font-bold text-white mb-2 text-center flex items-center justify-center gap-2">
        <Terminal className="w-7 h-7 text-emerald-400" />
        write(1, &quot;hello&quot;, 5) 系统调用路径
      </h3>
      <p className="text-gray-400 text-center text-sm mb-6">
        从用户态 write() 到内核处理再返回的完整 trap 路径
      </p>

      <div className="flex justify-center gap-3 mb-6">
        {!playing ? (
          <button
            onClick={handlePlay}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold transition-colors"
          >
            <Play className="w-4 h-4" />
            {currentPlay >= traceSteps.length - 1 ? "重播" : "播放"}
          </button>
        ) : (
          <button
            onClick={handlePause}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-semibold transition-colors"
          >
            <Pause className="w-4 h-4" />
            暂停
          </button>
        )}
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-white font-semibold transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      <div className="flex justify-center gap-4 mb-8">
        {(["User", "Hardware", "Kernel"] as Mode[]).map((m) => (
          <span key={m} className={`flex items-center gap-1.5 text-xs font-medium ${modeConfig[m].color}`}>
            {modeConfig[m].icon}
            {m}
          </span>
        ))}
      </div>

      <div className="relative">
        <div className="absolute left-[23px] top-0 bottom-0 w-0.5 bg-gray-700" />

        <div className="space-y-4">
          {traceSteps.map((step, idx) => {
            const cfg = modeConfig[step.mode];
            const isActive = activeStep === step.id;
            const isPlayed = currentPlay >= idx;

            return (
              <div
                key={step.id}
                ref={(el) => { stepRefs.current[idx] = el; }}
              >
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  onClick={() => {
                    setActiveStep(isActive ? null : step.id);
                  }}
                  className={`relative pl-14 cursor-pointer group`}
                >
                  <div className="absolute left-0 top-3 flex flex-col items-center">
                    <motion.div
                      animate={isPlayed ? { scale: [1, 1.3, 1] } : {}}
                      transition={{ duration: 0.3 }}
                      className={`w-12 h-12 rounded-full ${circleColors[step.mode]} flex items-center justify-center text-white font-bold text-sm z-10 shadow-lg ${
                        isActive ? "ring-2 ring-white/50" : ""
                      }`}
                    >
                      {step.id}
                    </motion.div>
                    {idx < traceSteps.length - 1 && (
                      <div className={`w-0.5 flex-1 min-h-[24px] ${lineColors[step.mode]}`} />
                    )}
                  </div>

                  <div
                    className={`rounded-lg border p-4 transition-all ${
                      isActive
                        ? `${cfg.bg} ${cfg.border} shadow-lg`
                        : "bg-gray-800/60 border-gray-700/50 hover:bg-gray-800"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold ${cfg.bg} ${cfg.color} ${cfg.border} border`}>
                        {cfg.icon}
                        {step.mode}
                      </span>
                      <span className="font-bold text-white text-sm">{step.title}</span>
                    </div>

                    <pre className="bg-gray-900 rounded-md p-3 text-xs font-mono text-emerald-300 overflow-x-auto mt-2">
                      {step.code}
                    </pre>

                    <AnimatePresence>
                      {isActive && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.25 }}
                          className="overflow-hidden"
                        >
                          <p className="text-gray-300 text-sm mt-3 leading-relaxed border-t border-gray-700 pt-3">
                            {step.description}
                          </p>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </motion.div>
              </div>
            );
          })}
        </div>
      </div>

      <AnimatePresence>
        {activeStep !== null && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            className="mt-6 bg-gray-800 border border-gray-700 rounded-lg p-4"
          >
            <h4 className="text-sm font-bold text-white mb-1">
              步骤 {activeStep} 详情
            </h4>
            <p className="text-gray-300 text-sm leading-relaxed">
              {traceSteps.find((s) => s.id === activeStep)?.description}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
