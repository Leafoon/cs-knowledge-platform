'use client';

import React, { useState } from 'react';

interface LogEntry {
  id: number;
  time: string;
  source: 'client' | 'server' | 'tracker';
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
}

const mockLogs: LogEntry[] = [
  { id: 1, time: '12:00:01', source: 'client', level: 'info', message: '连接 RPC Tracker 192.168.1.100:9190' },
  { id: 2, time: '12:00:01', source: 'tracker', level: 'info', message: '收到设备请求: gpu.0, 超时=10s' },
  { id: 3, time: '12:00:02', source: 'tracker', level: 'info', message: '分配设备 gpu.0 → client (key: session_0x7f)' },
  { id: 4, time: '12:00:02', source: 'client', level: 'info', message: '上传编译库 lib_gpu.so (2.3MB)' },
  { id: 5, time: '12:00:03', source: 'server', level: 'info', message: '加载库文件成功，初始化 CUDA context' },
  { id: 6, time: '12:00:03', source: 'server', level: 'debug', message: '分配 GPU 内存: input (4MB), weight (16MB), output (4MB)' },
  { id: 7, time: '12:00:04', source: 'client', level: 'info', message: '上传输入数据 (shape=[1,3,224,224])' },
  { id: 8, time: '12:00:04', source: 'server', level: 'info', message: '开始执行 kernel: fused_conv2d_relu (grid=128, block=256)' },
  { id: 9, time: '12:00:04', source: 'server', level: 'warn', message: '检测到 shared memory bank conflict，建议优化 tiling' },
  { id: 10, time: '12:00:05', source: 'server', level: 'info', message: '执行完成，耗时 23.5ms' },
  { id: 11, time: '12:00:05', source: 'client', level: 'info', message: '接收结果数据，验证通过 ✓' },
  { id: 12, time: '12:00:06', source: 'client', level: 'info', message: '断开 RPC 连接，释放远程资源' },
];

const sourceColors = {
  client: 'text-indigo-400',
  server: 'text-blue-400',
  tracker: 'text-purple-400',
};

const levelColors = {
  info: 'text-green-400',
  warn: 'text-yellow-400',
  error: 'text-red-400',
  debug: 'text-gray-400',
};

export function RPCDebuggingDemo() {
  const [visibleLogs, setVisibleLogs] = useState(0);
  const [sourceFilter, setSourceFilter] = useState<string | null>(null);
  const [levelFilter, setLevelFilter] = useState<string | null>(null);

  const filtered = mockLogs.filter((l) => {
    if (sourceFilter && l.source !== sourceFilter) return false;
    if (levelFilter && l.level !== levelFilter) return false;
    return true;
  });

  const displayed = filtered.slice(0, visibleLogs);

  const addLog = () => {
    if (visibleLogs < filtered.length) {
      setVisibleLogs((v) => v + 1);
    }
  };

  const reset = () => setVisibleLogs(0);

  const playAll = () => {
    setVisibleLogs(0);
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setVisibleLogs(i);
      if (i >= filtered.length) clearInterval(interval);
    }, 300);
  };

  return (
    <div className="w-full rounded-xl border border-white/10 bg-gradient-to-br from-gray-900 via-gray-950 to-black p-6">
      <h3 className="mb-2 text-lg font-bold text-white">RPC 调试演示</h3>
      <p className="mb-4 text-sm text-gray-400">
        模拟 RPC 远程执行过程中的日志输出，逐步查看客户端、服务端和调度器的交互。
      </p>
      <div className="mb-3 flex flex-wrap gap-2">
        <div className="flex gap-1">
          {['client', 'tracker', 'server'].map((s) => (
            <button
              key={s}
              onClick={() => { setSourceFilter(sourceFilter === s ? null : s); setVisibleLogs(0); }}
              className={`rounded px-2 py-1 text-[10px] font-medium transition-all ${sourceFilter === s ? 'bg-indigo-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}
            >
              {s}
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          {['info', 'warn', 'error', 'debug'].map((l) => (
            <button
              key={l}
              onClick={() => { setLevelFilter(levelFilter === l ? null : l); setVisibleLogs(0); }}
              className={`rounded px-2 py-1 text-[10px] font-medium transition-all ${levelFilter === l ? 'bg-indigo-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}
            >
              {l}
            </button>
          ))}
        </div>
      </div>
      <div className="mb-3 flex gap-2">
        <button onClick={addLog} className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs text-white hover:bg-indigo-500">
          ▶ 下一条
        </button>
        <button onClick={playAll} className="rounded-lg bg-purple-600 px-3 py-1.5 text-xs text-white hover:bg-purple-500">
          ⏩ 自动播放
        </button>
        <button onClick={reset} className="rounded-lg bg-white/5 px-3 py-1.5 text-xs text-gray-300 hover:bg-white/10">
          ↺ 重置
        </button>
      </div>
      <div className="rounded-lg border border-white/10 bg-black/50 p-3 font-mono text-[11px] max-h-[280px] overflow-y-auto">
        {displayed.length === 0 && (
          <div className="text-center text-gray-600 py-4">点击"下一条"或"自动播放"查看日志</div>
        )}
        {displayed.map((log) => (
          <div key={log.id} className="flex gap-2 py-0.5 hover:bg-white/5 rounded px-1">
            <span className="text-gray-600 flex-shrink-0">{log.time}</span>
            <span className={`w-12 flex-shrink-0 ${sourceColors[log.source]}`}>[{log.source}]</span>
            <span className={`w-8 flex-shrink-0 uppercase ${levelColors[log.level]}`}>{log.level}</span>
            <span className="text-gray-300">{log.message}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 flex justify-between text-[10px] text-gray-600">
        <span>显示 {displayed.length}/{filtered.length} 条日志</span>
        <span>hover 各节点查看详情</span>
      </div>
    </div>
  );
}
