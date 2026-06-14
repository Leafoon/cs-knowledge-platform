"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { GitBranch, Play } from "lucide-react";

export default function ForkExecWaitDemo() {
  const [step, setStep] = useState(0);

  const steps = [
    { title: "父进程运行", parent: "运行 Shell", child: null, action: "准备执行外部命令 ls" },
    { title: "fork() 创建子进程", parent: "调用 fork()", child: "新建子进程（PID 1235）", action: "复制进程，返回值：父→1235，子→0" },
    { title: "子进程 exec()", parent: "wait() 阻塞", child: "exec('/bin/ls')", action: "子进程内存替换为 ls 程序" },
    { title: "子进程执行", parent: "wait() 阻塞", child: "ls 执行中", action: "ls 列出目录内容" },
    { title: "子进程 exit()", parent: "wait() 阻塞", child: "exit(0) 僵尸", action: "子进程退出，变为僵尸状态" },
    { title: "父进程 wait() 返回", parent: "wait() 返回，回收子进程", child: "已回收", action: "父进程继续，Shell 显示提示符" }
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <GitBranch className="w-7 h-7 text-blue-600" />
        fork() + exec() + wait() 典型流程
      </h3>

      <div className="flex justify-center gap-4 mb-6">
        {steps.map((_, idx) => (
          <button
            key={idx}
            onClick={() => setStep(idx)}
            className={`px-4 py-2 rounded-lg font-semibold ${step === idx ? "bg-blue-600 text-white" : "bg-slate-200"}`}
          >
            {idx + 1}
          </button>
        ))}
      </div>

      <motion.div key={step} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-white rounded-lg shadow-md p-6 mb-4">
        <h4 className="font-bold text-xl mb-4">{steps[step].title}</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-green-100 border-2 border-green-400 p-4 rounded">
            <div className="font-bold text-green-800 mb-2">父进程 (Shell, PID 1234)</div>
            <div className="text-sm">{steps[step].parent}</div>
          </div>
          <div className={`p-4 rounded border-2 ${steps[step].child ? "bg-purple-100 border-purple-400" : "bg-slate-100 border-slate-300"}`}>
            <div className="font-bold text-purple-800 mb-2">子进程 {steps[step].child ? "(PID 1235)" : ""}</div>
            <div className="text-sm">{steps[step].child || "未创建"}</div>
          </div>
        </div>
        <div className="mt-4 bg-amber-50 p-3 rounded border border-amber-300">
          <div className="font-semibold text-amber-800">操作：{steps[step].action}</div>
        </div>
      </motion.div>

      <div className="bg-white rounded-lg shadow-md p-4">
        <h4 className="font-bold mb-2">代码示例</h4>
        <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto">
{`pid_t pid = fork();
if (pid == 0) {
    // 子进程
    execl("/bin/ls", "ls", "-l", NULL);
    perror("exec failed");
    exit(1);
} else {
    // 父进程
    int status;
    wait(&status);
    printf("子进程结束\\n");
}`}
        </pre>
      </div>
    </div>
  );
}
