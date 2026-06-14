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
  isTransaction?: boolean;
}

const createFlow: FlowStep[] = [
  {
    id: 1,
    layer: "用户空间",
    layerColor: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    action: 'open("newfile.txt", O_CREATE | O_WRONLY)',
    detail: "用户程序调用 open 并指定 O_CREATE 标志创建新文件",
  },
  {
    id: 2,
    layer: "日志层",
    layerColor: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300",
    action: "begin_op()",
    detail: "开始日志事务。如果日志空间不足则等待。设置事务的写入块计数器。",
    isTransaction: true,
  },
  {
    id: 3,
    layer: "路径名层",
    layerColor: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
    action: 'nameiparent(path, "newfile.txt")',
    detail: "解析路径，找到父目录的 inode。例如 /home/user/newfile.txt → 返回 /home/user 的 inode",
    isTransaction: true,
  },
  {
    id: 4,
    layer: "inode 层",
    layerColor: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
    action: "ialloc(dev, T_FILE)",
    detail: "扫描 inode 表找到空闲 inode（type==0），初始化 dinode（type=T_FILE, size=0），log_write 写入日志",
    isTransaction: true,
  },
  {
    id: 5,
    layer: "目录层",
    layerColor: "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-300",
    action: 'dirlink(dp, "newfile.txt", inum)',
    detail: "在父目录中查找空闲目录项（inum==0），写入 {inum, name} 对。通过 writei() 写入目录数据块",
    isTransaction: true,
  },
  {
    id: 6,
    layer: "文件描述符层",
    layerColor: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300",
    action: "filealloc()",
    detail: "在全局 file 表中找一个空闲的 struct file，设置 type=FD_INODE, ip=新inode, off=0",
    isTransaction: true,
  },
  {
    id: 7,
    layer: "文件描述符层",
    layerColor: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300",
    action: "fdalloc(f)",
    detail: "在进程的 ofile[] 中找到最小的空闲 fd 槽位，将 struct file 指针存入",
    isTransaction: true,
  },
  {
    id: 8,
    layer: "日志层",
    layerColor: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300",
    action: "end_op()",
    detail: "提交日志事务：将日志中记录的所有块写入实际磁盘位置，然后清除日志头。保证崩溃一致性。",
    isTransaction: true,
  },
  {
    id: 9,
    layer: "用户空间",
    layerColor: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    action: "return fd",
    detail: "返回文件描述符（如 fd=3），用户程序可以用它进行后续的 read/write 操作",
  },
];

export default function Xv6CreateFileFlow() {
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
      if (i >= createFlow.length) {
        clearInterval(interval);
        setIsRunning(false);
      }
    }, 1200);
  }, [reset]);

  const stepForward = () => {
    if (currentStep < createFlow.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        open(O_CREATE) 完整流程
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        追踪文件创建过程：从 open() 到返回文件描述符
      </p>

      <div className="flex gap-3 mb-6 justify-center flex-wrap">
        <button
          onClick={autoPlay}
          disabled={isRunning}
          className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <Play className="w-4 h-4" />
          自动播放
        </button>
        <button
          onClick={stepForward}
          disabled={isRunning || currentStep >= createFlow.length - 1}
          className="flex items-center gap-2 px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
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

      {/* Transaction bracket indicator */}
      <div className="relative mb-2">
        {currentStep >= 1 && currentStep <= 7 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute -left-2 top-0 bottom-0 w-1 bg-amber-400 dark:bg-amber-500 rounded-full"
            style={{
              top: `${(1 / createFlow.length) * 100}%`,
              height: `${(7 / createFlow.length) * 100}%`,
            }}
          />
        )}
      </div>

      {/* Flow steps */}
      <div className="space-y-2">
        {createFlow.map((step, i) => {
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
                        ? "border-amber-400 dark:border-amber-500 shadow-md bg-white dark:bg-gray-800"
                        : "border-slate-200 dark:border-gray-700 bg-white/60 dark:bg-gray-800/60 hover:border-slate-300 dark:hover:border-gray-600"
                    } ${step.isTransaction ? "border-l-4 border-l-amber-300 dark:border-l-amber-600" : ""}`}
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
                      <span className="text-sm font-mono font-bold text-slate-800 dark:text-gray-100 truncate">
                        {step.action}
                      </span>
                      {step.isTransaction && (
                        <span className="text-[10px] px-1.5 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 rounded shrink-0">
                          TX
                        </span>
                      )}
                      {isCurrent && (
                        <motion.span
                          className="ml-auto w-2 h-2 rounded-full bg-amber-500 shrink-0"
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

      {currentStep >= createFlow.length - 1 && !isRunning && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 text-center"
        >
          <span className="inline-block px-4 py-2 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-lg text-sm font-bold">
            完成！文件已创建，返回 fd
          </span>
        </motion.div>
      )}

      <div className="mt-4 text-xs text-slate-500 dark:text-gray-400 text-center">
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-1 bg-amber-300 dark:bg-amber-600 rounded" />
          TX = 日志事务范围（begin_op → end_op）
        </span>
      </div>
    </div>
  );
}
