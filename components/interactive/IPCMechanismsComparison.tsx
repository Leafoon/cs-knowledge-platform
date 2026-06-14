"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { MessageSquare } from "lucide-react";

export default function IPCMechanismsComparison() {
  const [selectedIPC, setSelectedIPC] = useState<string>("pipe");

  const ipcMechanisms = {
    pipe: {
      name: "管道 (Pipe)",
      type: "单向通信",
      speed: "快",
      scope: "父子进程",
      pros: ["简单易用", "内核缓冲，无需同步"],
      cons: ["单向", "仅限父子进程"],
      code: `int pipefd[2];
pipe(pipefd);
if (fork() == 0) {
    close(pipefd[1]);
    read(pipefd[0], buf, sizeof(buf));
} else {
    close(pipefd[0]);
    write(pipefd[1], "data", 4);
}`
    },
    fifo: {
      name: "命名管道 (FIFO)",
      type: "单向通信",
      speed: "快",
      scope: "无关进程",
      pros: ["无需父子关系", "文件系统可见"],
      cons: ["单向", "需创建文件"],
      code: `mkfifo("/tmp/myfifo", 0666);
// 写进程
int fd = open("/tmp/myfifo", O_WRONLY);
write(fd, "data", 4);
// 读进程
int fd = open("/tmp/myfifo", O_RDONLY);
read(fd, buf, sizeof(buf));`
    },
    shm: {
      name: "共享内存 (Shared Memory)",
      type: "双向通信",
      speed: "极快",
      scope: "无关进程",
      pros: ["最快（无拷贝）", "双向通信"],
      cons: ["需手动同步（信号量）", "复杂"],
      code: `int shmid = shmget(IPC_PRIVATE, 4096, 0666);
void *addr = shmat(shmid, NULL, 0);
// 进程A写
strcpy((char*)addr, "data");
// 进程B读
printf("%s", (char*)addr);
shmdt(addr);`
    },
    mq: {
      name: "消息队列 (Message Queue)",
      type: "双向通信",
      speed: "中",
      scope: "无关进程",
      pros: ["消息边界", "支持优先级"],
      cons: ["内核缓冲区大小限制"],
      code: `int mqid = msgget(IPC_PRIVATE, 0666);
struct msgbuf msg = {1, "data"};
msgsnd(mqid, &msg, sizeof(msg.mtext), 0);
msgrcv(mqid, &msg, sizeof(msg.mtext), 1, 0);`
    },
    signal: {
      name: "信号 (Signal)",
      type: "异步通知",
      speed: "快",
      scope: "无关进程",
      pros: ["异步", "轻量"],
      cons: ["信息量极少（仅信号编号）"],
      code: `kill(pid, SIGUSR1);
// 接收进程
signal(SIGUSR1, handler);
void handler(int sig) {
    printf("收到信号 %d\\n", sig);
}`
    },
    socket: {
      name: "套接字 (Socket)",
      type: "双向通信",
      speed: "慢（网络开销）",
      scope: "本地/网络",
      pros: ["支持网络通信", "双向", "通用"],
      cons: ["开销大", "复杂"],
      code: `int sock = socket(AF_UNIX, SOCK_STREAM, 0);
bind(sock, &addr, sizeof(addr));
listen(sock, 5);
int client = accept(sock, NULL, NULL);
send(client, "data", 4, 0);`
    }
  };

  const current = ipcMechanisms[selectedIPC as keyof typeof ipcMechanisms];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <MessageSquare className="w-7 h-7 text-indigo-600" />
        IPC 机制对比
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-6">
        {Object.entries(ipcMechanisms).map(([key, mech]) => (
          <button key={key} onClick={() => setSelectedIPC(key)} className={`px-4 py-3 rounded-lg font-semibold text-sm transition-all ${selectedIPC === key ? "bg-indigo-600 text-white" : "bg-slate-200 text-slate-700"}`}>
            {mech.name}
          </button>
        ))}
      </div>

      <motion.div key={selectedIPC} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-xl mb-4">{current.name}</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className="bg-blue-50 p-3 rounded border border-blue-200">
            <div className="text-xs text-slate-600">通信方式</div>
            <div className="font-semibold text-slate-800">{current.type}</div>
          </div>
          <div className="bg-green-50 p-3 rounded border border-green-200">
            <div className="text-xs text-slate-600">速度</div>
            <div className="font-semibold text-slate-800">{current.speed}</div>
          </div>
          <div className="bg-purple-50 p-3 rounded border border-purple-200">
            <div className="text-xs text-slate-600">适用范围</div>
            <div className="font-semibold text-slate-800">{current.scope}</div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-green-50 border-2 border-green-300 rounded p-3">
            <div className="font-bold text-green-800 mb-2 text-sm">优点</div>
            <ul className="text-xs text-slate-700 space-y-1 list-disc list-inside">
              {current.pros.map((pro, idx) => <li key={idx}>{pro}</li>)}
            </ul>
          </div>
          <div className="bg-red-50 border-2 border-red-300 rounded p-3">
            <div className="font-bold text-red-800 mb-2 text-sm">缺点</div>
            <ul className="text-xs text-slate-700 space-y-1 list-disc list-inside">
              {current.cons.map((con, idx) => <li key={idx}>{con}</li>)}
            </ul>
          </div>
        </div>

        <div>
          <div className="font-semibold text-slate-800 mb-2">代码示例</div>
          <pre className="bg-slate-900 text-green-400 p-4 rounded text-xs overflow-x-auto">{current.code}</pre>
        </div>
      </motion.div>

      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-indigo-600 text-white">
              <th className="px-4 py-3 text-left">机制</th>
              <th className="px-4 py-3 text-left">速度</th>
              <th className="px-4 py-3 text-left">范围</th>
              <th className="px-4 py-3 text-left">典型场景</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b hover:bg-slate-50"><td className="px-4 py-3 font-semibold">管道</td><td>快</td><td>父子</td><td>Shell 管道（ls | grep）</td></tr>
            <tr className="border-b hover:bg-slate-50"><td className="px-4 py-3 font-semibold">FIFO</td><td>快</td><td>无关</td><td>客户端-服务器本地通信</td></tr>
            <tr className="border-b hover:bg-slate-50"><td className="px-4 py-3 font-semibold">共享内存</td><td>极快</td><td>无关</td><td>大数据量、高性能通信</td></tr>
            <tr className="border-b hover:bg-slate-50"><td className="px-4 py-3 font-semibold">消息队列</td><td>中</td><td>无关</td><td>消息传递、解耦</td></tr>
            <tr className="border-b hover:bg-slate-50"><td className="px-4 py-3 font-semibold">信号</td><td>快</td><td>无关</td><td>异步通知（Ctrl+C）</td></tr>
            <tr className="hover:bg-slate-50"><td className="px-4 py-3 font-semibold">Socket</td><td>慢</td><td>网络</td><td>网络通信、分布式系统</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
