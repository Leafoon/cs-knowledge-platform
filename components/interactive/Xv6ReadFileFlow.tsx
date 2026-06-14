"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, ChevronDown } from "lucide-react";

interface FlowStep {
  id: number;
  layer: string;
  layerColor: string;
  action: string;
  detail: string;
}

const readFlow: FlowStep[] = [
  {
    id: 1,
    layer: "用户空间",
    layerColor: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    action: "read(fd, buf, n)",
    detail: "用户程序调用 read 系统调用，传入文件描述符 fd、缓冲区指针 buf 和读取字节数 n",
  },
  {
    id: 2,
    layer: "文件描述符层",
    layerColor: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300",
    action: "sys_read()",
    detail: "通过 argfd(0, &fd, &f) 从进程 fd 表获取 struct file 指针，然后调用 fileread(f, p, n)",
  },
  {
    id: 3,
    layer: "文件描述符层",
    layerColor: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300",
    action: "fileread(f, p, n)",
    detail: "检查 f->type：FD_PIPE 走 piperead()，FD_DEVICE 走 devsw[].read()，FD_INODE 走 readi()",
  },
  {
    id: 4,
    layer: "inode 层",
    layerColor: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
    action: "readi(ip, dst, off, n)",
    detail: "循环读取：每次计算需要读取的块，调用 bmap() 获取物理块号，再调用 bread() 读取",
  },
  {
    id: 5,
    layer: "inode 层",
    layerColor: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
    action: "bmap(ip, off/BSIZE)",
    detail: "将逻辑块号映射到物理块号：< NDIRECT 直接读 addrs[]，否则读间接块",
  },
  {
    id: 6,
    layer: "缓冲区缓存层",
    layerColor: "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300",
    action: "bread(dev, blockno)",
    detail: "调用 bget() 查找缓存：缓存命中直接返回 buf，未命中则从磁盘读取并标记 valid=1",
  },
  {
    id: 7,
    layer: "磁盘层",
    layerColor: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
    action: "virtio_disk_rw(b, 0)",
    detail: "仅在缓存未命中时执行。向磁盘发送读请求，读取 1024 字节到 buf->data",
  },
  {
    id: 8,
    layer: "inode 层",
    layerColor: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
    action: "memmove(dst, bp->data + off%BSIZE, m)",
    detail: "从缓冲区复制数据到用户缓冲区。m = min(n-tot, BSIZE - off%BSIZE)",
  },
  {
    id: 9,
    layer: "文件描述符层",
    layerColor: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300",
    action: "f->off += n",
    detail: "更新文件偏移量，brelse(bp) 释放缓冲区，返回实际读取字节数",
  },
];

export default function Xv6ReadFileFlow() {
  const [currentStep, setCurrentStep] = useState(-1);
  const [isRunning, setIsRunning] = useState(false);
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  const reset = useCallback(() => {
    setCurrentStep(-1);
    setIsRunning(false);
    setExpandedStep(null);
  }, []);

  const autoPlay = useCallback(() => {
    reset();
    setIsRunning(true);
    let i = 0;
    const interval = setInterval(() => {
      setCurrentStep(i);
      i++;
      if (i >= readFlow.length) {
        clearInterval(interval);
        setIsRunning(false);
      }
    }, 1200);
  }, [reset]);

  const stepForward = () => {
    if (currentStep < readFlow.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        read() 系统调用完整流程
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        追踪 read(fd, buf, n) 从用户空间到磁盘的完整调用链
      </p>

      <div className="flex gap-3 mb-6 justify-center flex-wrap">
        <button
          onClick={autoPlay}
          disabled={isRunning}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <Play className="w-4 h-4" />
          自动播放
        </button>
        <button
          onClick={stepForward}
          disabled={isRunning || currentStep >= readFlow.length - 1}
          className="flex items-center gap-2 px-4 py-2 bg-teal-500 text-white rounded-lg hover:bg-teal-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <ChevronDown className="w-4 h-4" />
          单步
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      {/* Flow steps */}
      <div className="space-y-2">
        {readFlow.map((step, i) => {
          const isVisible = i <= currentStep;
          const isCurrent = i === currentStep;

          return (
            <AnimatePresence key={step.id}>
              {isVisible && (
                <motion.div
                  initial={{ opacity: 0, y: -10, height: 0 }}
                  animate={{ opacity: 1, y: 0, height: "auto" }}
                  transition={{ duration: 0.3 }}
                >
                  <button
                    onClick={() =>
                      setExpandedStep(expandedStep === step.id ? null : step.id)
                    }
                    className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                      isCurrent
                        ? "border-cyan-400 dark:border-cyan-500 shadow-md bg-white dark:bg-gray-800"
                        : "border-slate-200 dark:border-gray-700 bg-white/60 dark:bg-gray-800/60 hover:border-slate-300 dark:hover:border-gray-600"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="w-6 h-6 rounded-full bg-slate-200 dark:bg-gray-700 flex items-center justify-center text-xs font-bold text-slate-600 dark:text-gray-300 shrink-0">
                        {step.id}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-mono shrink-0 ${step.layerColor}`}
                      >
                        {step.layer}
                      </span>
                      <span className="text-sm font-mono font-bold text-slate-800 dark:text-gray-100">
                        {step.action}
                      </span>
                      {isCurrent && (
                        <motion.span
                          className="ml-auto w-2 h-2 rounded-full bg-cyan-500"
                          animate={{ scale: [1, 1.3, 1] }}
                          transition={{ repeat: Infinity, duration: 1 }}
                        />
                      )}
                    </div>
                    {expandedStep === step.id && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        className="mt-3 ml-9 text-sm text-slate-600 dark:text-gray-300 leading-relaxed"
                      >
                        {step.detail}
                      </motion.div>
                    )}
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          );
        })}
      </div>

      {currentStep >= readFlow.length - 1 && !isRunning && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 text-center"
        >
          <span className="inline-block px-4 py-2 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-lg text-sm font-bold">
            完成！read() 返回实际读取的字节数
          </span>
        </motion.div>
      )}
    </div>
  );
}
