"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Zap, Info } from "lucide-react";

export default function StateTransitionTriggers() {
  const [selectedTransition, setSelectedTransition] = useState<number | null>(null);

  const transitionEvents = [
    {
      id: 0,
      transition: "新建 → 就绪",
      event: "进程初始化完成",
      trigger: "fork() 返回后子进程进入就绪",
      code: "allocproc(); // xv6\nproc->state = RUNNABLE;",
      color: "blue"
    },
    {
      id: 1,
      transition: "就绪 → 运行",
      event: "调度器选中进程",
      trigger: "时间片轮转、优先级调度",
      code: "schedule(); // 调度器\nswtch(&cpu->context, &proc->context);",
      color: "green"
    },
    {
      id: 2,
      transition: "运行 → 就绪",
      event: "时间片用完或被抢占",
      trigger: "10ms 时间片耗尽，定时器中断",
      code: "if (timer_interrupt) {\n  yield(); // 主动让出 CPU\n}",
      color: "yellow"
    },
    {
      id: 3,
      transition: "运行 → 阻塞",
      event: "进程主动等待事件",
      trigger: "read(fd, buf, n) 等待磁盘 I/O",
      code: "// 用户程序\nread(fd, buf, 1024);\n\n// 内核\nproc->state = SLEEPING;\nsleep(&disk_wait_queue);",
      color: "red"
    },
    {
      id: 4,
      transition: "阻塞 → 就绪",
      event: "等待的事件发生",
      trigger: "磁盘 I/O 完成，中断唤醒进程",
      code: "// 中断处理程序\nvoid disk_interrupt() {\n  wakeup(&disk_wait_queue);\n  proc->state = RUNNABLE;\n}",
      color: "blue"
    },
    {
      id: 5,
      transition: "运行 → 终止",
      event: "进程终止",
      trigger: "exit(0) 或 return 0 from main",
      code: "// 用户程序\nexit(0);\n\n// 内核\nproc->state = ZOMBIE;\nwakeup(proc->parent);",
      color: "gray"
    }
  ];

  const getColorClass = (color: string) => {
    const map: Record<string, { bg: string; border: string; text: string }> = {
      blue: { bg: "bg-blue-100", border: "border-blue-400", text: "text-blue-700" },
      green: { bg: "bg-green-100", border: "border-green-400", text: "text-green-700" },
      yellow: { bg: "bg-yellow-100", border: "border-yellow-400", text: "text-yellow-700" },
      red: { bg: "bg-red-100", border: "border-red-400", text: "text-red-700" },
      gray: { bg: "bg-gray-100", border: "border-gray-400", text: "text-gray-700" }
    };
    return map[color];
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-yellow-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Zap className="w-7 h-7 text-yellow-600" />
        状态转换条件与触发事件
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {transitionEvents.map((item, idx) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            whileHover={{ scale: 1.02 }}
            onClick={() => setSelectedTransition(item.id)}
            className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
              selectedTransition === item.id
                ? `${getColorClass(item.color).bg} ${getColorClass(item.color).border} shadow-lg`
                : "bg-white border-slate-200 hover:border-slate-300"
            }`}
          >
            <div className={`font-bold mb-2 ${getColorClass(item.color).text}`}>
              {item.transition}
            </div>
            <div className="text-sm text-slate-700 mb-1">
              <strong>事件：</strong>{item.event}
            </div>
            <div className="text-xs text-slate-600">
              <strong>触发：</strong>{item.trigger}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Detail Panel */}
      {selectedTransition !== null && (
        <motion.div
          key={selectedTransition}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-6 rounded-lg border-2 ${
            getColorClass(transitionEvents[selectedTransition].color).bg
          } ${getColorClass(transitionEvents[selectedTransition].color).border}`}
        >
          <div className="flex items-center gap-2 mb-4">
            <Info className="w-5 h-5" />
            <h4 className="text-lg font-bold text-slate-800">
              {transitionEvents[selectedTransition].transition}
            </h4>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h5 className="font-semibold text-slate-700 mb-2">触发条件</h5>
              <p className="text-sm text-slate-700 mb-3">
                {transitionEvents[selectedTransition].trigger}
              </p>
              <h5 className="font-semibold text-slate-700 mb-2">事件描述</h5>
              <p className="text-sm text-slate-700">
                {transitionEvents[selectedTransition].event}
              </p>
            </div>
            <div>
              <h5 className="font-semibold text-slate-700 mb-2">代码示例</h5>
              <pre className="bg-slate-900 text-green-400 p-3 rounded-lg text-xs overflow-x-auto">
                {transitionEvents[selectedTransition].code}
              </pre>
            </div>
          </div>
        </motion.div>
      )}

      {/* Example: read() System Call Flow */}
      <div className="mt-6 bg-white rounded-lg shadow-md p-6">
        <h4 className="font-bold text-slate-800 mb-4">示例：read() 系统调用的状态转换流程</h4>
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <div className="bg-yellow-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
              1
            </div>
            <div>
              <div className="font-semibold text-slate-800">进程处于 [运行] 状态</div>
              <div className="text-sm text-slate-600">执行 read() 系统调用</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
              2
            </div>
            <div>
              <div className="font-semibold text-slate-800">内核检查数据是否就绪</div>
              <div className="text-sm text-slate-600">
                如果数据在缓冲区：立即返回<br />
                如果数据未就绪：进程状态改为 [阻塞]
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
              3
            </div>
            <div>
              <div className="font-semibold text-slate-800">磁盘 I/O 完成，触发中断</div>
              <div className="text-sm text-slate-600">中断处理程序将进程状态改为 [就绪]</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="bg-yellow-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
              4
            </div>
            <div>
              <div className="font-semibold text-slate-800">调度器选中进程</div>
              <div className="text-sm text-slate-600">进程状态改为 [运行]，从 read() 返回</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
