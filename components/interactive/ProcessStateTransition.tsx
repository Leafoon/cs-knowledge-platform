"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Users, Clock, Activity, AlertCircle } from "lucide-react";

interface Process {
  id: number;
  name: string;
  state: "new" | "ready" | "running" | "waiting" | "terminated";
  priority: number;
  cpuTime: number;
  waitTime: number;
}

const stateColors = {
  new: "#8b5cf6",
  ready: "#3b82f6",
  running: "#10b981",
  waiting: "#f59e0b",
  terminated: "#6b7280",
};

const stateDescriptions = {
  new: "进程刚创建，等待操作系统接纳",
  ready: "进程已准备好运行，等待 CPU 分配",
  running: "进程正在 CPU 上执行",
  waiting: "进程等待 I/O 或其他事件",
  terminated: "进程执行完毕，等待回收资源",
};

export function ProcessStateTransition() {
  const [processes, setProcesses] = useState<Process[]>([
    { id: 1, name: "firefox", state: "running", priority: 5, cpuTime: 150, waitTime: 20 },
    { id: 2, name: "code", state: "ready", priority: 6, cpuTime: 80, waitTime: 5 },
    { id: 3, name: "chrome", state: "waiting", priority: 4, cpuTime: 200, waitTime: 100 },
  ]);

  const [selectedProcess, setSelectedProcess] = useState<number | null>(null);
  const [eventLog, setEventLog] = useState<string[]>([]);

  const addLogEvent = (message: string) => {
    setEventLog((prev) => [`[${new Date().toLocaleTimeString()}] ${message}`, ...prev].slice(0, 10));
  };

  const transitionProcess = (processId: number, newState: Process["state"]) => {
    setProcesses((prev) =>
      prev.map((p) => {
        if (p.id === processId) {
          addLogEvent(`进程 ${p.name} (PID ${p.id}): ${p.state} → ${newState}`);
          return { ...p, state: newState };
        }
        // If transitioning to running, set others to ready
        if (newState === "running" && p.state === "running") {
          return { ...p, state: "ready" };
        }
        return p;
      })
    );
  };

  const createProcess = () => {
    const newId = Math.max(...processes.map((p) => p.id), 0) + 1;
    const newProcess: Process = {
      id: newId,
      name: `proc-${newId}`,
      state: "new",
      priority: Math.floor(Math.random() * 10),
      cpuTime: 0,
      waitTime: 0,
    };
    setProcesses((prev) => [...prev, newProcess]);
    addLogEvent(`创建新进程: ${newProcess.name} (PID ${newId})`);
  };

  const deleteProcess = (processId: number) => {
    const process = processes.find((p) => p.id === processId);
    if (process) {
      setProcesses((prev) => prev.filter((p) => p.id !== processId));
      addLogEvent(`销毁进程: ${process.name} (PID ${processId})`);
      if (selectedProcess === processId) {
        setSelectedProcess(null);
      }
    }
  };

  const stateStats = {
    new: processes.filter((p) => p.state === "new").length,
    ready: processes.filter((p) => p.state === "ready").length,
    running: processes.filter((p) => p.state === "running").length,
    waiting: processes.filter((p) => p.state === "waiting").length,
    terminated: processes.filter((p) => p.state === "terminated").length,
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        进程状态转换模拟器
      </h3>

      {/* State Diagram */}
      <div className="mb-6 p-6 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <h4 className="font-semibold mb-4 text-text-primary">五状态模型</h4>
        <div className="relative">
          <svg width="100%" height="300" viewBox="0 0 800 300">
            {/* State Circles */}
            <circle cx="100" cy="150" r="40" fill={stateColors.new} opacity="0.3" />
            <text x="100" y="155" textAnchor="middle" fill="white" fontWeight="bold">
              New
            </text>

            <circle cx="300" cy="150" r="40" fill={stateColors.ready} opacity="0.3" />
            <text x="300" y="155" textAnchor="middle" fill="white" fontWeight="bold">
              Ready
            </text>

            <circle cx="500" cy="150" r="40" fill={stateColors.running} opacity="0.3" />
            <text x="500" y="155" textAnchor="middle" fill="white" fontWeight="bold">
              Running
            </text>

            <circle cx="500" cy="50" r="40" fill={stateColors.waiting} opacity="0.3" />
            <text x="500" y="55" textAnchor="middle" fill="white" fontWeight="bold">
              Waiting
            </text>

            <circle cx="700" cy="150" r="40" fill={stateColors.terminated} opacity="0.3" />
            <text x="700" y="155" textAnchor="middle" fill="white" fontWeight="bold">
              Terminated
            </text>

            {/* Arrows */}
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" fill="#374151" />
              </marker>
            </defs>

            {/* New → Ready (admit) */}
            <line x1="140" y1="150" x2="260" y2="150" stroke="#374151" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <text x="200" y="140" fontSize="12" fill="#6b7280">
              admit
            </text>

            {/* Ready ⇄ Running (dispatch/timeout) */}
            <line x1="340" y1="145" x2="460" y2="145" stroke="#374151" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <text x="380" y="135" fontSize="12" fill="#6b7280">
              dispatch
            </text>
            <line x1="460" y1="155" x2="340" y2="155" stroke="#374151" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <text x="380" y="170" fontSize="12" fill="#6b7280">
              timeout
            </text>

            {/* Running → Waiting (event wait) */}
            <line x1="500" y1="110" x2="500" y2="90" stroke="#374151" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <text x="510" y="100" fontSize="12" fill="#6b7280">
              I/O wait
            </text>

            {/* Waiting → Ready (event completion) */}
            <path d="M 460 50 Q 380 50 320 100" stroke="#374151" strokeWidth="2" fill="none" markerEnd="url(#arrowhead)" />
            <text x="380" y="60" fontSize="12" fill="#6b7280">
              I/O complete
            </text>

            {/* Running → Terminated (exit) */}
            <line x1="540" y1="150" x2="660" y2="150" stroke="#374151" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <text x="580" y="140" fontSize="12" fill="#6b7280">
              exit
            </text>
          </svg>
        </div>
      </div>

      {/* Statistics */}
      <div className="mb-6 grid grid-cols-5 gap-3">
        {Object.entries(stateStats).map(([state, count]) => (
          <div
            key={state}
            className="p-3 rounded-lg border-2"
            style={{
              borderColor: stateColors[state as keyof typeof stateColors],
              backgroundColor: `${stateColors[state as keyof typeof stateColors]}10`,
            }}
          >
            <div
              className="text-2xl font-bold"
              style={{ color: stateColors[state as keyof typeof stateColors] }}
            >
              {count}
            </div>
            <div className="text-xs text-text-secondary capitalize">{state}</div>
          </div>
        ))}
      </div>

      {/* Process Table */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-semibold text-text-primary">进程列表</h4>
          <button
            onClick={createProcess}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition text-sm"
          >
            + 创建进程
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  PID
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  进程名
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  状态
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  优先级
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  CPU 时间
                </th>
                <th className="px-3 py-2 text-left border border-border-subtle text-text-primary">
                  操作
                </th>
              </tr>
            </thead>
            <tbody>
              {processes.map((process) => (
                <tr
                  key={process.id}
                  onClick={() => setSelectedProcess(process.id)}
                  className={`cursor-pointer transition ${
                    selectedProcess === process.id
                      ? "bg-blue-50 dark:bg-blue-900/20"
                      : "hover:bg-gray-50 dark:hover:bg-gray-800"
                  }`}
                >
                  <td className="px-3 py-2 border border-border-subtle font-mono text-text-primary">
                    {process.id}
                  </td>
                  <td className="px-3 py-2 border border-border-subtle font-mono text-text-primary">
                    {process.name}
                  </td>
                  <td className="px-3 py-2 border border-border-subtle">
                    <span
                      className="px-2 py-1 rounded text-xs font-semibold text-white"
                      style={{ backgroundColor: stateColors[process.state] }}
                    >
                      {process.state}
                    </span>
                  </td>
                  <td className="px-3 py-2 border border-border-subtle text-text-primary">
                    {process.priority}
                  </td>
                  <td className="px-3 py-2 border border-border-subtle text-text-primary">
                    {process.cpuTime} ms
                  </td>
                  <td className="px-3 py-2 border border-border-subtle">
                    <div className="flex gap-1">
                      {process.state === "new" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            transitionProcess(process.id, "ready");
                          }}
                          className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
                        >
                          接纳
                        </button>
                      )}
                      {process.state === "ready" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            transitionProcess(process.id, "running");
                          }}
                          className="px-2 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700"
                        >
                          调度
                        </button>
                      )}
                      {process.state === "running" && (
                        <>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              transitionProcess(process.id, "waiting");
                            }}
                            className="px-2 py-1 bg-yellow-600 text-white rounded text-xs hover:bg-yellow-700"
                          >
                            I/O
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              transitionProcess(process.id, "ready");
                            }}
                            className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
                          >
                            超时
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              transitionProcess(process.id, "terminated");
                            }}
                            className="px-2 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-700"
                          >
                            退出
                          </button>
                        </>
                      )}
                      {process.state === "waiting" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            transitionProcess(process.id, "ready");
                          }}
                          className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
                        >
                          完成
                        </button>
                      )}
                      {process.state === "terminated" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteProcess(process.id);
                          }}
                          className="px-2 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700"
                        >
                          回收
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Process Details */}
      {selectedProcess && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800"
        >
          {(() => {
            const process = processes.find((p) => p.id === selectedProcess);
            if (!process) return null;
            return (
              <div>
                <h4 className="font-semibold mb-2 text-text-primary">
                  进程详情: {process.name} (PID {process.id})
                </h4>
                <p className="text-sm text-text-secondary mb-2">
                  <strong>当前状态：</strong>
                  <span
                    className="ml-2 px-2 py-1 rounded text-white font-semibold"
                    style={{ backgroundColor: stateColors[process.state] }}
                  >
                    {process.state}
                  </span>
                </p>
                <p className="text-sm text-text-secondary">
                  {stateDescriptions[process.state]}
                </p>
              </div>
            );
          })()}
        </motion.div>
      )}

      {/* Event Log */}
      <div>
        <h4 className="font-semibold mb-3 text-text-primary">事件日志</h4>
        <div className="bg-gray-900 text-green-400 font-mono text-xs p-4 rounded-lg h-48 overflow-y-auto">
          {eventLog.map((log, index) => (
            <div key={index} className="mb-1">
              {log}
            </div>
          ))}
          {eventLog.length === 0 && (
            <div className="text-gray-500">等待事件...</div>
          )}
        </div>
      </div>
    </div>
  );
}
