"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Network, ArrowRight, Cpu, Zap } from "lucide-react";

type Arch = "FSB" | "QPI" | "UPI";

interface ArchInfo {
  name: string;
  year: string;
  desc: string;
  topology: string;
  bandwidth: string;
  features: string[];
}

const ARCH_DATA: Record<Arch, ArchInfo> = {
  FSB: {
    name: "前端总线 (FSB)",
    year: "1990s-2008",
    desc: "共享总线架构，所有设备共享一条总线",
    topology: "共享总线",
    bandwidth: "最高 1.6 GT/s",
    features: ["共享带宽", "半双工", "单处理器优化", "瓶颈明显"],
  },
  QPI: {
    name: "快速通道互联 (QPI)",
    year: "2008-2017",
    desc: "点对点互联，Intel多处理器互联方案",
    topology: "点对点网状",
    bandwidth: "最高 9.6 GT/s",
    features: ["点对点连接", "全双工", "多处理器支持", "缓存一致性"],
  },
  UPI: {
    name: "超级通道互联 (UPI)",
    year: "2017-至今",
    desc: "Intel最新互联架构，替代QPI",
    topology: "高速网状",
    bandwidth: "最高 10.4 GT/s",
    features: ["更高带宽", "更低延迟", "可扩展架构", "支持多路服务器"],
  },
};

const COLORS: Record<Arch, string> = { FSB: "amber", QPI: "blue", UPI: "green" };

export function ModernBusArchViz() {
  const [arch, setArch] = useState<Arch>("FSB");
  const info = ARCH_DATA[arch];
  const color = COLORS[arch];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Network className="w-5 h-5 text-indigo-500" />
        现代总线架构演进
      </h3>
      <div className="flex gap-2 mb-4">
        {(Object.keys(ARCH_DATA) as Arch[]).map(a => (
          <button key={a} onClick={() => setArch(a)}
            className={`px-3 py-1.5 rounded text-sm ${arch === a ? `bg-${color}-500 text-white` : "bg-bg-subtle"}`}>
            {ARCH_DATA[a].name.split("(")[0].trim()}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-4">{info.desc}</p>
      <div className="mb-4">
        <div className="text-xs text-text-secondary mb-2">拓扑结构: {info.topology}</div>
        {arch === "FSB" && (
          <div className="flex items-center justify-center gap-4 py-4">
            <div className="px-3 py-2 bg-amber-500/20 border border-amber-500 rounded text-sm">
              <Cpu className="w-4 h-4 inline mr-1" />CPU
            </div>
            <div className="flex-1 h-1 bg-amber-500/30 relative">
              <motion.div animate={{ x: [0, 100, 0] }} transition={{ repeat: Infinity, duration: 2 }}
                className="absolute w-2 h-2 bg-amber-500 rounded-full top-[-3px]" />
            </div>
            <div className="px-3 py-2 bg-amber-500/20 border border-amber-500 rounded text-sm">内存</div>
            <div className="px-3 py-2 bg-amber-500/20 border border-amber-500 rounded text-sm">北桥</div>
          </div>
        )}
        {arch === "QPI" && (
          <div className="flex flex-col items-center gap-2 py-4">
            <div className="flex gap-8">
              {["CPU 0", "CPU 1"].map((c, i) => (
                <div key={i} className="px-3 py-2 bg-blue-500/20 border border-blue-500 rounded text-sm">
                  <Cpu className="w-4 h-4 inline mr-1" />{c}
                </div>
              ))}
            </div>
            <div className="flex gap-2 items-center">
              <ArrowRight className="w-4 h-4 text-blue-500" />
              <span className="text-xs text-blue-500">QPI Link</span>
              <ArrowRight className="w-4 h-4 text-blue-500 rotate-180" />
            </div>
            <div className="flex gap-8">
              {["内存通道", "内存通道"].map((m, i) => (
                <div key={i} className="px-3 py-2 bg-blue-500/20 border border-blue-500 rounded text-sm">{m}</div>
              ))}
            </div>
          </div>
        )}
        {arch === "UPI" && (
          <div className="flex flex-wrap justify-center gap-4 py-4">
            {["CPU 0", "CPU 1", "CPU 2", "CPU 3"].map((c, i) => (
              <motion.div key={i} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: i * 0.1 }}
                className="px-3 py-2 bg-green-500/20 border border-green-500 rounded text-sm">
                <Cpu className="w-4 h-4 inline mr-1" />{c}
              </motion.div>
            ))}
            <svg className="absolute inset-0 pointer-events-none" />
          </div>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-bg-subtle rounded p-3">
          <div className="text-xs text-text-secondary">年代</div>
          <div className="font-bold">{info.year}</div>
        </div>
        <div className="bg-bg-subtle rounded p-3">
          <div className="text-xs text-text-secondary">带宽</div>
          <div className="font-bold">{info.bandwidth}</div>
        </div>
      </div>
      <div className="mt-3">
        <div className="text-xs text-text-secondary mb-1">特性</div>
        <div className="flex flex-wrap gap-1">
          {info.features.map((f, i) => (
            <motion.span key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.1 }}
              className={`px-2 py-0.5 rounded text-xs bg-${color}-500/10 text-${color}-500 border border-${color}-500/30`}>
              {f}
            </motion.span>
          ))}
        </div>
      </div>
    </div>
  );
}
