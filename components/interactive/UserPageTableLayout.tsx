"use client";
import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Segment {
  label: string;
  range: string;
  description: string;
  permissions: string;
  grows: string;
  color: string;
  darkColor: string;
  borderColor: string;
  darkBorderColor: string;
  height: string;
}

const segments: Segment[] = [
  {
    label: "Trampoline",
    range: "0x3FFFFFFFE000 - 0x3FFFFFFFF000",
    description: "trap 处理代码，内核态/用户态切换时使用。所有用户页表映射到同一物理页面。",
    permissions: "R-X",
    grows: "固定",
    color: "bg-purple-100",
    darkColor: "dark:bg-purple-900/40",
    borderColor: "border-purple-400",
    darkBorderColor: "dark:border-purple-600",
    height: "h-10",
  },
  {
    label: "Trapframe",
    range: "0x3FFFFFFFDF00",
    description: "保存用户态寄存器，trap 进入内核时使用。每个进程独立的物理页面。",
    permissions: "RW-",
    grows: "固定",
    color: "bg-rose-100",
    darkColor: "dark:bg-rose-900/40",
    borderColor: "border-rose-400",
    darkBorderColor: "dark:border-rose-600",
    height: "h-10",
  },
  {
    label: "栈 (Stack)",
    range: "向下增长",
    description: "用户栈，存储局部变量、函数调用帧、函数参数。向下增长，下方有 guard page。",
    permissions: "RW-",
    grows: "↓ 向下",
    color: "bg-orange-100",
    darkColor: "dark:bg-orange-900/30",
    borderColor: "border-orange-400",
    darkBorderColor: "dark:border-orange-600",
    height: "h-24",
  },
  {
    label: "堆 (Heap)",
    range: "p->sz 向上增长",
    description: "动态内存分配区域，通过 sbrk 系统调用增长。malloc/free 管理此区域。",
    permissions: "RW-",
    grows: "↑ 向上",
    color: "bg-green-100",
    darkColor: "dark:bg-green-900/30",
    borderColor: "border-green-400",
    darkBorderColor: "dark:border-green-600",
    height: "h-32",
  },
  {
    label: ".data + .bss",
    range: "紧跟 .text 之后",
    description: ".data 存储已初始化全局变量，.bss 存储未初始化全局变量（零填充）。",
    permissions: "RW-",
    grows: "固定",
    color: "bg-cyan-100",
    darkColor: "dark:bg-cyan-900/30",
    borderColor: "border-cyan-400",
    darkBorderColor: "dark:border-cyan-600",
    height: "h-14",
  },
  {
    label: ".text (代码段)",
    range: "0x0 开始",
    description: "程序的可执行代码，从 ELF 文件加载。只读可执行，防止代码被篡改。",
    permissions: "R-X",
    grows: "固定",
    color: "bg-blue-100",
    darkColor: "dark:bg-blue-900/30",
    borderColor: "border-blue-400",
    darkBorderColor: "dark:border-blue-600",
    height: "h-16",
  },
];

export default function UserPageTableLayout() {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">
        xv6 用户地址空间布局
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
        每个进程独立的虚拟地址空间，点击各段查看详情
      </p>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2 text-xs text-slate-400 dark:text-slate-500 font-mono">
            <span>高地址 (MAXVA)</span>
            <span className="flex-1 text-center">↓</span>
            <span>低地址 (0x0)</span>
          </div>

          <div className="flex flex-col gap-1">
            {segments.map((seg, i) => (
              <motion.div
                key={seg.label}
                onClick={() => setSelected(selected === i ? null : i)}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                className={`relative cursor-pointer rounded-lg border-2 p-3 transition-colors ${seg.color} ${seg.darkColor} ${seg.borderColor} ${seg.darkBorderColor} ${seg.height} flex items-center ${
                  selected === i
                    ? "ring-2 ring-blue-500 dark:ring-blue-400"
                    : ""
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm text-slate-800 dark:text-slate-100">
                      {seg.label}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 font-mono">
                      {seg.permissions}
                    </span>
                    {seg.grows !== "固定" && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-200 dark:bg-amber-800 text-amber-700 dark:text-amber-300">
                        {seg.grows}
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-slate-500 dark:text-slate-400 font-mono hidden sm:block">
                    {seg.range}
                  </span>
                </div>
              </motion.div>
            ))}
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
                  {segments[selected].label}
                </h4>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">
                      地址范围：
                    </span>
                    <div className="font-mono text-xs text-slate-700 dark:text-slate-300 mt-0.5">
                      {segments[selected].range}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">
                      权限：
                    </span>
                    <span className="ml-2 font-mono text-xs px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
                      {segments[selected].permissions}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">
                      增长方向：
                    </span>
                    <span className="ml-2 text-xs text-slate-700 dark:text-slate-300">
                      {segments[selected].grows}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400">
                      说明：
                    </span>
                    <p className="text-xs text-slate-600 dark:text-slate-300 mt-1 leading-relaxed">
                      {segments[selected].description}
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
              关键设计
            </h4>
            <ul className="space-y-1.5 text-xs text-slate-500 dark:text-slate-400">
              <li>• 栈和堆之间有未映射区域，防止相互覆盖</li>
              <li>• Trampoline 在所有页表中映射到同一虚拟地址</li>
              <li>• Guard page 无映射，栈溢出触发页故障</li>
              <li>• 代码段 R-X 防止运行时篡改</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
