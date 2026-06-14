"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Network, HardDrive, Terminal, Globe, MessageSquare, Users, Lock, Eye, EyeOff } from "lucide-react";

interface NSInfo {
  id: string;
  name: string;
  icon: React.ReactNode;
  color: string;
  description: string;
  hostView: string[];
  containerAView: string[];
  containerBView: string[];
  isolation: string;
}

const namespaces: NSInfo[] = [
  {
    id: "pid",
    name: "PID Namespace",
    icon: <Terminal className="w-4 h-4" />,
    color: "from-blue-500 to-cyan-500",
    description: "隔离进程 ID 空间，容器内 PID 从 1 开始，看不到宿主机和其他容器的进程。",
    hostView: ["PID 1: systemd", "PID 500: containerd", "PID 1000: bash (容器A)", "PID 1001: nginx", "PID 2000: bash (容器B)", "PID 2001: mysql"],
    containerAView: ["PID 1: bash (init)", "PID 2: nginx"],
    containerBView: ["PID 1: bash (init)", "PID 2: mysql"],
    isolation: "进程完全隔离，容器内 PID 1 负责回收僵尸进程",
  },
  {
    id: "net",
    name: "Network Namespace",
    icon: <Network className="w-4 h-4" />,
    color: "from-green-500 to-emerald-500",
    description: "隔离网络接口、IP 地址、路由表、iptables 规则。每个容器有自己的网络栈。",
    hostView: ["eth0: 192.168.1.100", "docker0: 10.0.0.1", "veth-abc: (容器A)", "veth-def: (容器B)", "lo: 127.0.0.1"],
    containerAView: ["eth0: 10.0.0.2 (veth)", "lo: 127.0.0.1", "default gw: 10.0.0.1"],
    containerBView: ["eth0: 10.0.0.3 (veth)", "lo: 127.0.0.1", "default gw: 10.0.0.1"],
    isolation: "每个容器独立的 IP、路由、iptables，通过 veth pair + bridge 连接",
  },
  {
    id: "mnt",
    name: "Mount Namespace",
    icon: <HardDrive className="w-4 h-4" />,
    color: "from-orange-500 to-amber-500",
    description: "隔离文件系统挂载点。容器有自己的 rootfs，看不到宿主机文件系统。",
    hostView: ["/: ext4 (宿主rootfs)", "/home: ext4", "/mnt/data: xfs", "/proc: proc", "/sys: sysfs"],
    containerAView: ["/: overlay2 (容器A rootfs)", "/bin, /lib, /etc: (镜像层)", "/proc: proc (PID NS)", "/tmp: tmpfs"],
    containerBView: ["/: overlay2 (容器B rootfs)", "/bin, /lib, /etc: (镜像层)", "/proc: proc (PID NS)", "/tmp: tmpfs"],
    isolation: "独立的文件系统视图，overlay2 分层文件系统实现镜像共享",
  },
  {
    id: "uts",
    name: "UTS Namespace",
    icon: <Globe className="w-4 h-4" />,
    color: "from-violet-500 to-purple-500",
    description: "隔离主机名 (hostname) 和域名 (domainname)。每个容器可以有独立的主机名。",
    hostView: ["hostname: prod-server-01"],
    containerAView: ["hostname: web-app-01"],
    containerBView: ["hostname: db-master"],
    isolation: "容器修改 hostname 不影响宿主机和其他容器",
  },
  {
    id: "ipc",
    name: "IPC Namespace",
    icon: <MessageSquare className="w-4 h-4" />,
    color: "from-pink-500 to-rose-500",
    description: "隔离 System V IPC 和 POSIX 消息队列。容器间无法通过共享内存、信号量通信。",
    hostView: ["shm: 共享内存段 A, B", "sem: 信号量集 X", "msg: 消息队列 Q"],
    containerAView: ["shm: (空)", "sem: (空)", "msg: (空)"],
    containerBView: ["shm: (空)", "sem: (空)", "msg: (空)"],
    isolation: "容器 A 无法访问容器 B 的 IPC 资源",
  },
  {
    id: "user",
    name: "User Namespace",
    icon: <Users className="w-4 h-4" />,
    color: "from-teal-500 to-cyan-500",
    description: "隔离 UID/GID 映射。容器内 root (UID 0) 可映射到宿主非特权用户。",
    hostView: ["UID 0: root", "UID 100000-165536: 容器用户映射"],
    containerAView: ["UID 0: root → 宿主 UID 100000", "UID 1000: app → 宿主 UID 101000"],
    containerBView: ["UID 0: root → 宿主 UID 200000", "UID 1000: app → 宿主 UID 201000"],
    isolation: "容器内 root 在宿主机上只是普通用户，提升安全性 (rootless)",
  },
];

export default function NamespaceIsolation() {
  const [selectedNS, setSelectedNS] = useState("pid");
  const [showHost, setShowHost] = useState(true);
  const current = namespaces.find((n) => n.id === selectedNS)!;

  return (
    <div className="w-full space-y-5">
      <div className="flex flex-wrap gap-1.5">
        {namespaces.map((ns) => (
          <button
            key={ns.id}
            onClick={() => setSelectedNS(ns.id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              selectedNS === ns.id
                ? `bg-gradient-to-r ${ns.color} text-white shadow-lg`
                : "bg-gray-800/60 text-gray-400 hover:text-gray-200 border border-gray-700"
            }`}
          >
            {ns.icon}
            {ns.name.replace(" Namespace", "")}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selectedNS}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.2 }}
          className="space-y-4"
        >
          <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center gap-2 mb-2">
              <div className={`p-1.5 rounded-md bg-gradient-to-br ${current.color} text-white`}>
                {current.icon}
              </div>
              <h3 className="text-sm font-bold text-white">{current.name}</h3>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">{current.description}</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {showHost && (
              <motion.div
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-gray-800/60 rounded-xl p-4 border border-yellow-500/30"
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className="w-2 h-2 rounded-full bg-yellow-500" />
                  <h4 className="text-xs font-semibold text-yellow-400">宿主机视角</h4>
                </div>
                <ul className="space-y-1.5">
                  {current.hostView.map((item, i) => (
                    <motion.li
                      key={item}
                      initial={{ opacity: 0, x: -4 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="text-[11px] text-gray-300 font-mono bg-gray-900/50 px-2 py-1 rounded"
                    >
                      {item}
                    </motion.li>
                  ))}
                </ul>
              </motion.div>
            )}

            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gray-800/60 rounded-xl p-4 border border-blue-500/30"
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="w-2 h-2 rounded-full bg-blue-500" />
                <h4 className="text-xs font-semibold text-blue-400">容器 A 视角</h4>
              </div>
              <ul className="space-y-1.5">
                {current.containerAView.map((item, i) => (
                  <motion.li
                    key={item}
                    initial={{ opacity: 0, x: -4 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 + i * 0.05 }}
                    className="text-[11px] text-gray-300 font-mono bg-gray-900/50 px-2 py-1 rounded"
                  >
                    {item}
                  </motion.li>
                ))}
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="bg-gray-800/60 rounded-xl p-4 border border-green-500/30"
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                <h4 className="text-xs font-semibold text-green-400">容器 B 视角</h4>
              </div>
              <ul className="space-y-1.5">
                {current.containerBView.map((item, i) => (
                  <motion.li
                    key={item}
                    initial={{ opacity: 0, x: -4 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.15 + i * 0.05 }}
                    className="text-[11px] text-gray-300 font-mono bg-gray-900/50 px-2 py-1 rounded"
                  >
                    {item}
                  </motion.li>
                ))}
              </ul>
            </motion.div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowHost(!showHost)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs transition-colors"
            >
              {showHost ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
              {showHost ? "隐藏宿主机" : "显示宿主机"}
            </button>
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-gray-800/60 rounded-xl p-4 border border-gray-700 flex items-start gap-2"
          >
            <Lock className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-semibold text-purple-400 mb-1">隔离效果</p>
              <p className="text-xs text-gray-400">{current.isolation}</p>
            </div>
          </motion.div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
