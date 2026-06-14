'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Server {
  id: string;
  name: string;
  load: number;
  maxLoad: number;
  status: 'online' | 'busy' | 'offline';
  device: string;
}

const initialServers: Server[] = [
  { id: 's1', name: 'Server-A', load: 0, maxLoad: 8, status: 'online', device: 'RTX 3090' },
  { id: 's2', name: 'Server-B', load: 0, maxLoad: 4, status: 'online', device: 'RTX 4090' },
  { id: 's3', name: 'Server-C', load: 0, maxLoad: 16, status: 'online', device: 'A100' },
  { id: 's4', name: 'Server-D', load: 0, maxLoad: 4, status: 'online', device: 'RTX 3080' },
];

interface Request {
  id: number;
  targetServer: string | null;
  status: 'pending' | 'dispatched' | 'done';
  label: string;
}

const requestLabels = ['ResNet推理', 'BERT编译', 'TVM调优', '模型量化', '图优化', 'Kernel编译'];

export default function TrackerLoadBalancing() {
  const [servers, setServers] = useState<Server[]>(initialServers);
  const [requests, setRequests] = useState<Request[]>([]);
  const [requestId, setRequestId] = useState(0);
  const [strategy, setStrategy] = useState<'roundrobin' | 'leastload' | 'random'>('leastload');
  const [rrIndex, setRrIndex] = useState(0);
  const [autoMode, setAutoMode] = useState(false);

  const dispatchRequest = useCallback(() => {
    const onlineServers = servers.filter((s) => s.status !== 'offline');
    if (onlineServers.length === 0) return;

    let target: Server;
    if (strategy === 'leastload') {
      target = onlineServers.reduce((min, s) =>
        s.load / s.maxLoad < min.load / min.maxLoad ? s : min
      );
    } else if (strategy === 'roundrobin') {
      const idx = rrIndex % onlineServers.length;
      target = onlineServers[idx];
      setRrIndex((prev) => prev + 1);
    } else {
      target = onlineServers[Math.floor(Math.random() * onlineServers.length)];
    }

    const newReq: Request = {
      id: requestId,
      targetServer: target.id,
      status: 'dispatched',
      label: requestLabels[requestId % requestLabels.length],
    };

    setRequestId((prev) => prev + 1);
    setRequests((prev) => [newReq, ...prev].slice(0, 20));
    setServers((prev) =>
      prev.map((s) =>
        s.id === target.id ? { ...s, load: Math.min(s.load + 1, s.maxLoad) } : s
      )
    );

    setTimeout(() => {
      setServers((prev) =>
        prev.map((s) =>
          s.id === target.id ? { ...s, load: Math.max(s.load - 1, 0) } : s
        )
      );
      setRequests((prev) =>
        prev.map((r) => (r.id === newReq.id ? { ...r, status: 'done' } : r))
      );
    }, 2000 + Math.random() * 2000);
  }, [servers, strategy, rrIndex, requestId]);

  useEffect(() => {
    if (!autoMode) return;
    const interval = setInterval(dispatchRequest, 800);
    return () => clearInterval(interval);
  }, [autoMode, dispatchRequest]);

  const toggleServer = (id: string) => {
    setServers((prev) =>
      prev.map((s) =>
        s.id === id
          ? { ...s, status: s.status === 'offline' ? 'online' : 'offline', load: 0 }
          : s
      )
    );
  };

  const getLoadColor = (load: number, max: number) => {
    const ratio = load / max;
    if (ratio < 0.3) return 'from-green-500 to-emerald-600';
    if (ratio < 0.7) return 'from-yellow-500 to-amber-600';
    return 'from-red-500 to-rose-600';
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        RPC Tracker 负载均衡
      </h2>
      <p className="text-gray-400 text-center text-sm mb-6">
        Tracker 如何在多个 RPC Server 之间分发编译和运行请求
      </p>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-center gap-4 mb-6">
        <div className="flex gap-2">
          {(['leastload', 'roundrobin', 'random'] as const).map((s) => (
            <button
              key={s}
              onClick={() => setStrategy(s)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                strategy === s
                  ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {s === 'leastload' ? '最少负载' : s === 'roundrobin' ? '轮询' : '随机'}
            </button>
          ))}
        </div>
        <button
          onClick={dispatchRequest}
          className="px-4 py-1.5 rounded-lg text-xs font-medium bg-blue-600 text-white hover:bg-blue-500 transition-all"
        >
          发送请求
        </button>
        <button
          onClick={() => setAutoMode(!autoMode)}
          className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${
            autoMode ? 'bg-red-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          {autoMode ? '停止自动' : '自动发送'}
        </button>
      </div>

      {/* Tracker */}
      <div className="flex justify-center mb-6">
        <motion.div
          layout
          className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 rounded-xl px-8 py-4 border border-indigo-500/30"
        >
          <div className="text-center">
            <div className="text-indigo-400 text-sm font-medium">RPC Tracker</div>
            <div className="text-xs text-gray-500 mt-1">策略: {strategy === 'leastload' ? '最少负载' : strategy === 'roundrobin' ? '轮询' : '随机'}</div>
            <div className="text-xs text-gray-500">已处理: {requestId}</div>
          </div>
        </motion.div>
      </div>

      {/* Servers */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {servers.map((server) => (
          <motion.div
            key={server.id}
            layout
            onClick={() => toggleServer(server.id)}
            className={`rounded-xl p-4 border cursor-pointer transition-all ${
              server.status === 'offline'
                ? 'bg-gray-800/30 border-gray-700/50 opacity-50'
                : 'bg-gray-800/60 border-gray-700 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-white text-sm font-medium">{server.name}</span>
              <span
                className={`w-2 h-2 rounded-full ${
                  server.status === 'offline' ? 'bg-red-500' : 'bg-green-500'
                }`}
              />
            </div>
            <div className="text-xs text-gray-500 mb-3">{server.device}</div>

            {/* Load bar */}
            <div className="bg-gray-900 rounded-full h-3 overflow-hidden mb-1">
              <motion.div
                className={`h-full rounded-full bg-gradient-to-r ${getLoadColor(
                  server.load,
                  server.maxLoad
                )}`}
                animate={{ width: `${(server.load / server.maxLoad) * 100}%` }}
                transition={{ type: 'spring', stiffness: 200 }}
              />
            </div>
            <div className="text-xs text-gray-500 text-right">
              {server.load}/{server.maxLoad}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Request log */}
      <div className="bg-gray-800/40 rounded-xl border border-gray-700 overflow-hidden">
        <div className="px-4 py-2 bg-gray-800/80 border-b border-gray-700">
          <span className="text-sm text-gray-300">请求日志</span>
        </div>
        <div className="max-h-48 overflow-y-auto">
          <AnimatePresence>
            {requests.slice(0, 10).map((req) => (
              <motion.div
                key={req.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-3 px-4 py-2 border-b border-gray-700/30"
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    req.status === 'done' ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'
                  }`}
                />
                <span className="text-xs text-gray-500 font-mono w-8">#{req.id}</span>
                <span className="text-xs text-gray-300 flex-1">{req.label}</span>
                <span className="text-xs text-indigo-400">
                  → {servers.find((s) => s.id === req.targetServer)?.name}
                </span>
                <span className="text-xs text-gray-500">
                  {req.status === 'done' ? '✓' : '...'}
                </span>
              </motion.div>
            ))}
          </AnimatePresence>
          {requests.length === 0 && (
            <div className="px-4 py-6 text-center text-gray-600 text-sm">
              点击「发送请求」或开启自动模式
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
