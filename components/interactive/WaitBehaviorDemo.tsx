"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Clock, Play } from "lucide-react";

export default function WaitBehaviorDemo() {
  const [scenario, setScenario] = useState<"wait" | "waitpid" | "nowait">("wait");
  const [step, setStep] = useState(0);

  const scenarios = {
    wait: {
      title: "wait(NULL) - 等待任意子进程",
      description: "父进程阻塞直到任意一个子进程结束",
      steps: [
        { title: "父进程创建3个子进程", parent: "运行中", children: ["运行中", "运行中", "运行中"] },
        { title: "父进程调用 wait()", parent: "阻塞（等待）", children: ["运行中", "运行中", "运行中"] },
        { title: "子进程2结束", parent: "阻塞（等待）", children: ["运行中", "僵尸（退出码123）", "运行中"], zombie: 1 },
        { title: "wait() 返回，回收子进程2", parent: "运行中（获得 PID 1236，退出码 123）", children: ["运行中", "已回收", "运行中"], cleaned: 1 }
      ],
      code: `pid_t pid = wait(NULL);  // 等待任意子进程
printf("子进程 %d 已结束\\n", pid);

// 或获取退出状态
int status;
pid = wait(&status);
if (WIFEXITED(status)) {
    printf("退出码: %d\\n", WEXITSTATUS(status));
}`
    },
    waitpid: {
      title: "waitpid(pid, &status, 0) - 等待特定子进程",
      description: "父进程等待指定 PID 的子进程结束",
      steps: [
        { title: "父进程创建3个子进程", parent: "运行中", children: ["运行中 (PID 1235)", "运行中 (PID 1236)", "运行中 (PID 1237)"] },
        { title: "父进程调用 waitpid(1237, ...)", parent: "阻塞（等待 PID 1237）", children: ["运行中", "运行中", "运行中"], waitingFor: 2 },
        { title: "子进程2（PID 1236）结束", parent: "阻塞（仍等待 1237）", children: ["运行中", "僵尸（被忽略）", "运行中"], zombie: 1 },
        { title: "子进程3（PID 1237）结束", parent: "阻塞（等待 1237）", children: ["运行中", "僵尸", "僵尸（退出码 200）"], zombie: 2, targetZombie: 2 },
        { title: "waitpid() 返回，回收子进程3", parent: "运行中（获得 PID 1237，退出码 200）", children: ["运行中", "僵尸（仍需回收）", "已回收"], cleaned: 2 }
      ],
      code: `int status;
pid_t pid = waitpid(1237, &status, 0);  // 等待 PID 1237
printf("子进程 %d 已结束，退出码 %d\\n", 
       pid, WEXITSTATUS(status));

// 其他子进程变为僵尸，需额外 wait() 回收`
    },
    nowait: {
      title: "不调用 wait() - 僵尸进程产生",
      description: "父进程不调用 wait()，导致子进程变为僵尸进程",
      steps: [
        { title: "父进程创建子进程", parent: "运行中", children: ["运行中"] },
        { title: "子进程结束", parent: "运行中（未调用 wait）", children: ["僵尸（退出码 100）"], zombie: 0 },
        { title: "僵尸进程累积", parent: "运行中（仍未调用 wait）", children: ["僵尸（PCB 未释放）"], zombie: 0, warning: true },
        { title: "父进程结束", parent: "已退出", children: ["孤儿（被 init 收养并回收）"], orphan: 0 }
      ],
      code: `// 错误示例：不调用 wait()
pid_t pid = fork();
if (pid == 0) {
    exit(100);  // 子进程退出
} else {
    // 父进程继续运行，未调用 wait()
    // 子进程变为僵尸！
    while (1) { /* ... */ }
}

// 解决方案：调用 wait()
wait(NULL);  // 回收僵尸进程`
    }
  };

  const currentScenario = scenarios[scenario];
  const currentStepData: any = currentScenario.steps[step] || currentScenario.steps[0];

  const handleNext = () => {
    if (step < currentScenario.steps.length - 1) {
      setStep(step + 1);
    }
  };

  const handlePrev = () => {
    if (step > 0) {
      setStep(step - 1);
    }
  };

  const handleReset = () => {
    setStep(0);
  };

  const handleScenarioChange = (newScenario: "wait" | "waitpid" | "nowait") => {
    setScenario(newScenario);
    setStep(0);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Clock className="w-7 h-7 text-cyan-600" />
        wait() / waitpid() 行为演示
      </h3>

      {/* Scenario Selector */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={() => handleScenarioChange("wait")}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            scenario === "wait"
              ? "bg-blue-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          wait()
        </button>
        <button
          onClick={() => handleScenarioChange("waitpid")}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            scenario === "waitpid"
              ? "bg-blue-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          waitpid()
        </button>
        <button
          onClick={() => handleScenarioChange("nowait")}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            scenario === "nowait"
              ? "bg-red-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          不调用 wait()
        </button>
      </div>

      {/* Scenario Info */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <h4 className="font-bold text-slate-800">{currentScenario.title}</h4>
        <p className="text-sm text-slate-600">{currentScenario.description}</p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={handlePrev}
          disabled={step === 0}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          上一步
        </button>
        <button
          onClick={handleReset}
          className="px-6 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700"
        >
          重置
        </button>
        <button
          onClick={handleNext}
          disabled={step === currentScenario.steps.length - 1}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          下一步
        </button>
      </div>

      {/* Progress */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-slate-600 mb-2">
          <span>步骤 {step + 1} / {currentScenario.steps.length}</span>
          <span>{currentStepData.title}</span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-2">
          <motion.div
            className="bg-cyan-600 h-2 rounded-full"
            animate={{ width: `${((step) / (currentScenario.steps.length - 1)) * 100}%` }}
          />
        </div>
      </div>

      {/* Visualization */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="space-y-4">
          {/* Parent Process */}
          <div
            className={`p-4 rounded-lg border-2 ${
              currentStepData.parent.includes("阻塞")
                ? "bg-yellow-100 border-yellow-400"
                : currentStepData.parent.includes("已退出")
                ? "bg-slate-100 border-slate-400"
                : "bg-green-100 border-green-400"
            }`}
          >
            <div className="font-bold text-slate-800 mb-2">父进程 (PID 1234)</div>
            <div className="text-sm text-slate-700">{currentStepData.parent}</div>
          </div>

          {/* Child Processes */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {currentStepData.children.map((child: string, idx: number) => (
              <motion.div
                key={idx}
                animate={{
                  scale:
                    currentStepData.zombie === idx || (currentStepData as any).orphan === idx
                      ? [1, 1.05, 1]
                      : (currentStepData as any).cleaned === idx
                      ? [1, 0.95, 1]
                      : 1
                }}
                className={`p-4 rounded-lg border-2 ${
                  child.includes("僵尸")
                    ? "bg-red-100 border-red-400"
                    : child.includes("已回收")
                    ? "bg-slate-100 border-slate-400"
                    : child.includes("孤儿")
                    ? "bg-purple-100 border-purple-400"
                    : (currentStepData as any).waitingFor === idx
                    ? "bg-blue-100 border-blue-400 ring-4 ring-blue-300"
                    : "bg-cyan-100 border-cyan-400"
                }`}
              >
                <div className="font-bold text-slate-800 mb-1">子进程 {idx + 1}</div>
                <div className="text-sm text-slate-700">{child}</div>
                {currentStepData.zombie === idx && (
                  <div className="text-xs mt-2 bg-red-200 px-2 py-1 rounded inline-block">
                    ☠️ 僵尸状态
                  </div>
                )}
                {(currentStepData as any).cleaned === idx && (
                  <div className="text-xs mt-2 bg-green-200 px-2 py-1 rounded inline-block">
                    ✅ 已回收
                  </div>
                )}
                {(currentStepData as any).orphan === idx && (
                  <div className="text-xs mt-2 bg-purple-200 px-2 py-1 rounded inline-block">
                    👻 孤儿 → init 收养
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          {currentStepData.warning && (
            <div className="bg-red-100 border-2 border-red-400 p-3 rounded-lg">
              <div className="font-bold text-red-800">⚠️ 警告：僵尸进程未回收</div>
              <div className="text-sm text-red-700 mt-1">
                僵尸进程占用 PCB 和进程表项，大量僵尸进程可能导致无法创建新进程（达到 NPROC 限制）。
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <h4 className="font-bold text-slate-800 mb-3">示例代码</h4>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
          {currentScenario.code}
        </pre>
      </div>

      {/* Comparison Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <h4 className="font-bold text-slate-800 p-4 bg-slate-100 border-b border-slate-200">wait() vs waitpid() 对比</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">特性</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">wait()</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">waitpid()</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-100">
              <td className="px-4 py-3 font-semibold text-slate-800">等待对象</td>
              <td className="px-4 py-3 text-slate-700">任意子进程</td>
              <td className="px-4 py-3 text-slate-700">指定 PID 或进程组</td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="px-4 py-3 font-semibold text-slate-800">阻塞行为</td>
              <td className="px-4 py-3 text-slate-700">阻塞直到任意子进程结束</td>
              <td className="px-4 py-3 text-slate-700">可选 WNOHANG 非阻塞</td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="px-4 py-3 font-semibold text-slate-800">返回值</td>
              <td className="px-4 py-3 text-slate-700">结束的子进程 PID</td>
              <td className="px-4 py-3 text-slate-700">结束的子进程 PID（或 0 / -1）</td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="px-4 py-3 font-semibold text-slate-800">灵活性</td>
              <td className="px-4 py-3 text-slate-700">简单，适用于单个子进程</td>
              <td className="px-4 py-3 text-slate-700">高度灵活，支持多种选项</td>
            </tr>
            <tr>
              <td className="px-4 py-3 font-semibold text-slate-800">等价关系</td>
              <td className="px-4 py-3 text-slate-700 font-mono">wait(&status)</td>
              <td className="px-4 py-3 text-slate-700 font-mono">waitpid(-1, &status, 0)</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Key Points */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 mb-3">关键要点</h4>
        <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
          <li>
            <strong>wait() 等价于 waitpid(-1, &status, 0)</strong>：等待任意子进程，阻塞模式。
          </li>
          <li>
            <strong>waitpid() 更灵活</strong>：可等待特定 PID、进程组（pid &lt; -1），支持 WNOHANG（非阻塞）、WUNTRACED（暂停子进程也返回）。
          </li>
          <li>
            <strong>必须调用 wait()</strong>：否则子进程退出后变为僵尸，长期累积可能耗尽进程表。
          </li>
          <li>
            <strong>退出状态宏</strong>：WIFEXITED、WEXITSTATUS、WIFSIGNALED、WTERMSIG 等用于解析 status。
          </li>
          <li>
            <strong>父进程先退出</strong>：子进程变为孤儿，由 init（PID 1）收养并自动回收。
          </li>
        </ul>
      </div>
    </div>
  );
}
