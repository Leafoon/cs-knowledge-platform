"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Trash2, CheckCircle } from "lucide-react";

export default function ResourceCleanupTimeline() {
  const [phase, setPhase] = useState<"all" | "exit" | "wait">("all");

  const resources = [
    {
      name: "用户态资源",
      items: [
        { resource: "atexit() 回调函数", timing: "exit", retained: false, description: "按注册逆序执行清理回调" },
        { resource: "标准 I/O 流缓冲区", timing: "exit", retained: false, description: "fflush(stdout/stderr)" },
        { resource: "C 库打开的 FILE*", timing: "exit", retained: false, description: "fclose() 所有流" }
      ]
    },
    {
      name: "内核态资源（exit 时释放）",
      items: [
        { resource: "文件描述符", timing: "exit", retained: false, description: "关闭所有 fd，引用计数 -1" },
        { resource: "用户空间内存（代码/数据/堆/栈）", timing: "exit", retained: false, description: "uvmfree() 释放页表和物理页" },
        { resource: "当前目录 inode", timing: "exit", retained: false, description: "iput(cwd)" },
        { resource: "挂起的信号", timing: "exit", retained: false, description: "清空信号队列" }
      ]
    },
    {
      name: "保留资源（exit 后仍存在，wait 时释放）",
      items: [
        { resource: "进程控制块 PCB", timing: "wait", retained: true, description: "保留 pid、xstate、parent 等" },
        { resource: "PID", timing: "wait", retained: true, description: "防止 PID 复用，wait() 后释放" },
        { resource: "退出码", timing: "wait", retained: true, description: "保存在 p->xstate，wait() 返回给父进程" },
        { resource: "父进程指针", timing: "wait", retained: true, description: "用于 wakeup(parent)" },
        { resource: "进程表项", timing: "wait", retained: true, description: "proc[] 数组中的槽位" }
      ]
    },
    {
      name: "不释放资源（跨进程生命周期）",
      items: [
        { resource: "共享内存段", timing: "never", retained: true, description: "需显式 shmctl(IPC_RMID) 删除" },
        { resource: "System V 信号量/消息队列", timing: "never", retained: true, description: "需显式 semctl/msgctl 删除" },
        { resource: "mmap() 共享映射（MAP_SHARED）", timing: "never", retained: true, description: "其他进程仍可访问" },
        { resource: "已创建的文件/目录", timing: "never", retained: true, description: "持久化到磁盘，需显式 unlink()" }
      ]
    }
  ];

  const filteredResources = phase === "all"
    ? resources
    : resources.map(category => ({
        ...category,
        items: category.items.filter(item => item.timing === phase)
      })).filter(category => category.items.length > 0);

  const getTimingColor = (timing: string) => {
    const map: Record<string, string> = {
      exit: "bg-red-100 border-red-400 text-red-700",
      wait: "bg-yellow-100 border-yellow-400 text-yellow-700",
      never: "bg-blue-100 border-blue-400 text-blue-700"
    };
    return map[timing] || "bg-slate-100 border-slate-400 text-slate-700";
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-purple-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Trash2 className="w-7 h-7 text-purple-600" />
        进程终止资源清理时间线
      </h3>

      {/* Phase Filter */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={() => setPhase("all")}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            phase === "all"
              ? "bg-purple-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          全部资源
        </button>
        <button
          onClick={() => setPhase("exit")}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            phase === "exit"
              ? "bg-red-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          exit() 时释放
        </button>
        <button
          onClick={() => setPhase("wait")}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            phase === "wait"
              ? "bg-yellow-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          wait() 时释放
        </button>
      </div>

      {/* Timeline */}
      <div className="space-y-6">
        {filteredResources.map((category, cidx) => (
          <motion.div
            key={cidx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: cidx * 0.1 }}
            className="bg-white rounded-lg shadow-md overflow-hidden"
          >
            <div className="bg-slate-100 border-b-2 border-slate-200 p-4">
              <h4 className="font-bold text-slate-800 flex items-center gap-2">
                {category.items[0]?.retained ? (
                  <CheckCircle className="w-5 h-5 text-blue-600" />
                ) : (
                  <Trash2 className="w-5 h-5 text-red-600" />
                )}
                {category.name}
              </h4>
            </div>
            <div className="p-4">
              <div className="space-y-3">
                {category.items.map((item, iidx) => (
                  <motion.div
                    key={iidx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: cidx * 0.1 + iidx * 0.05 }}
                    className="flex items-start gap-3"
                  >
                    <div className={`px-3 py-1 rounded-lg border-2 text-xs font-semibold whitespace-nowrap ${
                      getTimingColor(item.timing)
                    }`}>
                      {item.timing === "exit" ? "exit() 时" :
                       item.timing === "wait" ? "wait() 时" :
                       "不释放"}
                    </div>
                    <div className="flex-1">
                      <div className="font-semibold text-slate-800">{item.resource}</div>
                      <div className="text-sm text-slate-600">{item.description}</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Timeline Diagram */}
      <div className="mt-8 bg-white rounded-lg shadow-md p-6">
        <h4 className="font-bold text-slate-800 mb-4 text-center">时间线示意图</h4>
        <div className="flex items-center justify-between gap-4">
          {/* Running */}
          <div className="flex-1 text-center">
            <div className="bg-green-100 border-2 border-green-400 rounded-lg p-4 mb-2">
              <div className="font-bold text-green-700">运行中</div>
              <div className="text-xs text-slate-600 mt-1">所有资源存在</div>
            </div>
          </div>

          <div className="text-2xl text-slate-400">→</div>

          {/* exit() */}
          <div className="flex-1 text-center">
            <div className="bg-red-100 border-2 border-red-400 rounded-lg p-4 mb-2">
              <div className="font-bold text-red-700">exit()</div>
              <div className="text-xs text-slate-600 mt-1">释放：文件、内存、I/O</div>
              <div className="text-xs text-slate-600">保留：PCB、PID、退出码</div>
            </div>
          </div>

          <div className="text-2xl text-slate-400">→</div>

          {/* Zombie */}
          <div className="flex-1 text-center">
            <div className="bg-yellow-100 border-2 border-yellow-400 rounded-lg p-4 mb-2">
              <div className="font-bold text-yellow-700">僵尸状态</div>
              <div className="text-xs text-slate-600 mt-1">等待父进程 wait()</div>
            </div>
          </div>

          <div className="text-2xl text-slate-400">→</div>

          {/* wait() */}
          <div className="flex-1 text-center">
            <div className="bg-blue-100 border-2 border-blue-400 rounded-lg p-4 mb-2">
              <div className="font-bold text-blue-700">wait()</div>
              <div className="text-xs text-slate-600 mt-1">释放：PCB、PID、进程表项</div>
            </div>
          </div>

          <div className="text-2xl text-slate-400">→</div>

          {/* Terminated */}
          <div className="flex-1 text-center">
            <div className="bg-slate-100 border-2 border-slate-400 rounded-lg p-4 mb-2">
              <div className="font-bold text-slate-700">完全终止</div>
              <div className="text-xs text-slate-600 mt-1">所有资源已释放</div>
            </div>
          </div>
        </div>
      </div>

      {/* Comparison: Normal vs Zombie */}
      <div className="mt-8 bg-white rounded-lg shadow-md overflow-hidden">
        <h4 className="font-bold text-slate-800 p-4 bg-slate-100 border-b border-slate-200">正常退出 vs 僵尸进程对比</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">场景</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">exit() 后状态</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">wait() 调用</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">资源占用</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-100">
              <td className="px-4 py-3 font-semibold text-slate-800">正常退出</td>
              <td className="px-4 py-3 text-slate-700">僵尸（短暂）</td>
              <td className="px-4 py-3 text-green-700 font-semibold">父进程及时调用 wait()</td>
              <td className="px-4 py-3 text-slate-700">PCB 很快释放</td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="px-4 py-3 font-semibold text-slate-800">僵尸进程</td>
              <td className="px-4 py-3 text-red-700 font-semibold">僵尸（长期）</td>
              <td className="px-4 py-3 text-red-700">父进程未调用 wait()</td>
              <td className="px-4 py-3 text-red-700">PCB 长期占用（泄漏）</td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="px-4 py-3 font-semibold text-slate-800">孤儿进程</td>
              <td className="px-4 py-3 text-slate-700">僵尸（短暂）</td>
              <td className="px-4 py-3 text-blue-700">init 自动调用 wait()</td>
              <td className="px-4 py-3 text-slate-700">init 自动回收</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Key Points */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 mb-3">关键要点</h4>
        <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
          <li>
            <strong>两阶段释放</strong>：exit() 释放大部分资源（文件、内存），wait() 释放 PCB 和 PID。
          </li>
          <li>
            <strong>僵尸进程的本质</strong>：已释放用户资源但 PCB 仍存在，等待父进程读取退出码。
          </li>
          <li>
            <strong>为什么要保留 PCB</strong>：父进程需要通过 wait() 获取子进程退出码（成功/失败/信号终止），
            如果 exit() 时直接释放 PCB，父进程无法获知子进程结果。
          </li>
          <li>
            <strong>僵尸进程的危害</strong>：长期未 wait() 会导致进程表耗尽（NPROC 限制），
            无法创建新进程（fork 返回 -1）。
          </li>
          <li>
            <strong>跨进程资源</strong>：共享内存、信号量、消息队列、文件系统对象不随进程终止自动释放，
            需显式清理（ipcrm、unlink 等）。
          </li>
        </ul>
      </div>

      {/* Code Example */}
      <div className="mt-6 bg-white rounded-lg shadow-md p-4">
        <h4 className="font-bold text-slate-800 mb-3">完整示例：正确的资源清理</h4>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
{`#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

// 清理函数（atexit 注册）
void cleanup() {
    printf("清理资源...\\n");
}

int main() {
    atexit(cleanup);  // 注册清理函数
    
    pid_t pid = fork();
    if (pid == 0) {
        // 子进程
        printf("子进程运行中 (PID %d)\\n", getpid());
        exit(42);  // 退出码 42
    } else {
        // 父进程
        int status;
        pid_t child = wait(&status);  // 阻塞等待子进程
        if (WIFEXITED(status)) {
            printf("子进程 %d 正常退出，退出码 %d\\n",
                   child, WEXITSTATUS(status));
        }
    }
    return 0;  // 隐式调用 exit(0)
}`}
        </pre>
      </div>
    </div>
  );
}
