"use client";
import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Region {
  label: string;
  vaStart: string;
  vaEnd: string;
  paStart: string;
  description: string;
  permissions: string;
  color: string;
  darkColor: string;
  borderColor: string;
  darkBorderColor: string;
}

const regions: Region[] = [
  {
    label: "Trampoline",
    vaStart: "0xFFFFFFFF80000000",
    vaEnd: "0xFFFFFFFF80001000",
    paStart: "trampoline 物理页",
    description: "用户态/内核态切换代码，所有页表共享映射",
    permissions: "R-X",
    color: "bg-purple-100",
    darkColor: "dark:bg-purple-900/40",
    borderColor: "border-purple-400",
    darkBorderColor: "dark:border-purple-600",
  },
  {
    label: "内核栈 (per-CPU)",
    vaStart: "0xFFFFFFFF80000000 - offset",
    vaEnd: "每个 CPU 2 页 + guard page",
    paStart: "独立物理页",
    description: "每个 CPU 核心独立的内核栈，带 guard page 防止栈溢出",
    permissions: "RW-",
    color: "bg-orange-100",
    darkColor: "dark:bg-orange-900/40",
    borderColor: "border-orange-400",
    darkBorderColor: "dark:border-orange-600",
  },
  {
    label: "物理内存 (直接映射)",
    vaStart: "0x80000000 + N",
    vaEnd: "0x80000000 + PHYSTOP",
    paStart: "= 虚拟地址 (1:1)",
    description: "内核代码之后的物理内存，直接映射（虚拟地址 = 物理地址）",
    permissions: "RW-",
    color: "bg-green-100",
    darkColor: "dark:bg-green-900/30",
    borderColor: "border-green-400",
    darkBorderColor: "dark:border-green-600",
  },
  {
    label: "内核代码 + 数据",
    vaStart: "0x80000000",
    vaEnd: "0x80000000 + etext",
    paStart: "= 虚拟地址 (1:1)",
    description: "内核 .text 和 .rodata 段，直接映射，可读可执行",
    permissions: "R-X",
    color: "bg-blue-100",
    darkColor: "dark:bg-blue-900/30",
    borderColor: "border-blue-400",
    darkBorderColor: "dark:border-blue-600",
  },
  {
    label: "PLIC",
    vaStart: "0x0C000000",
    vaEnd: "0x0C400000",
    paStart: "0x0C000000",
    description: "Platform Level Interrupt Controller，中断控制器寄存器",
    permissions: "RW-",
    color: "bg-red-100",
    darkColor: "dark:bg-red-900/30",
    borderColor: "border-red-400",
    darkBorderColor: "dark:border-red-600",
  },
  {
    label: "VIRTIO0",
    vaStart: "0x10000000",
    vaEnd: "0x10000100",
    paStart: "0x10000000",
    description: "虚拟磁盘设备寄存器，用于块设备 I/O",
    permissions: "RW-",
    color: "bg-cyan-100",
    darkColor: "dark:bg-cyan-900/30",
    borderColor: "border-cyan-400",
    darkBorderColor: "dark:border-cyan-600",
  },
  {
    label: "UART0",
    vaStart: "0x10000000",
    vaEnd: "0x10000008",
    paStart: "0x10000000",
    description: "串口通信控制器，用于 console 输入输出",
    permissions: "RW-",
    color: "bg-amber-100",
    darkColor: "dark:bg-amber-900/30",
    borderColor: "border-amber-400",
    darkBorderColor: "dark:border-amber-600",
  },
];

export default function KernelPageTableLayout() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">
        xv6 内核地址空间布局
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
        内核页表使用直接映射（虚拟地址 = 物理地址），点击各区域查看详情
      </p>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1">
          <div className="relative">
            <div className="absolute left-[45%] top-0 bottom-0 w-px bg-slate-200 dark:bg-slate-700" />
            <div className="flex flex-col gap-1">
              {regions.map((region, i) => (
                <motion.div
                  key={region.label}
                  onClick={() => setSelected(selected === i ? null : i)}
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                  className={`relative cursor-pointer rounded-lg border-2 p-3 transition-colors ${region.color} ${region.darkColor} ${region.borderColor} ${region.darkBorderColor} ${
                    selected === i
                      ? "ring-2 ring-blue-500 dark:ring-blue-400"
                      : ""
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm text-slate-800 dark:text-slate-100">
                        {region.label}
                      </span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 font-mono">
                        {region.permissions}
                      </span>
                    </div>
                    <span className="text-xs text-slate-500 dark:text-slate-400 font-mono hidden sm:block">
                      VA: {region.vaStart}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        <div className="lg:w-80">
          <AnimatePresence mode="wait">
            {selected !== null ? (
              <motion.div
                key={selected}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700"
              >
                <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-3">
                  {regions[selected].label}
                </h4>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">
                      虚拟地址范围：
                    </span>
                    <div className="font-mono text-xs text-slate-700 dark:text-slate-300 mt-0.5">
                      {regions[selected].vaStart}
                    </div>
                    <div className="font-mono text-xs text-slate-700 dark:text-slate-300">
                      {regions[selected].vaEnd}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">
                      物理地址映射：
                    </span>
                    <div className="font-mono text-xs text-slate-700 dark:text-slate-300 mt-0.5">
                      {regions[selected].paStart}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">
                      权限：
                    </span>
                    <span className="ml-2 font-mono text-xs px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
                      {regions[selected].permissions}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">
                      说明：
                    </span>
                    <p className="text-xs text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">
                      {regions[selected].description}
                    </p>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-slate-50 dark:bg-slate-800 rounded-lg p-4 border border-dashed border-slate-300 dark:border-slate-600 text-center"
              >
                <p className="text-sm text-slate-400 dark:text-slate-500">
                  点击左侧区域查看详细信息
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="mt-4 bg-slate-50 dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
            <h4 className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">
              关键映射关系
            </h4>
            <div className="space-y-1.5 text-xs font-mono text-slate-500 dark:text-slate-400">
              <div>
                UART/VIRTIO/PLIC → 直接映射到物理设备地址
              </div>
              <div>
                内核代码 → 直接映射，R-X 权限
              </div>
              <div>
                物理内存 → 直接映射，RW- 权限
              </div>
              <div>
                Trampoline → 映射到 trampoline 物理页
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


