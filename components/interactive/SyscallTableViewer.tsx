"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Table, Search } from "lucide-react";

export default function SyscallTableViewer() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  const syscalls = [
    { number: 0, name: "read", category: "文件I/O", params: "fd, buf, count", description: "从文件读取数据" },
    { number: 1, name: "write", category: "文件I/O", params: "fd, buf, count", description: "向文件写入数据" },
    { number: 2, name: "open", category: "文件I/O", params: "filename, flags, mode", description: "打开文件" },
    { number: 3, name: "close", category: "文件I/O", params: "fd", description: "关闭文件描述符" },
    { number: 9, name: "mmap", category: "内存管理", params: "addr, len, prot, flags, fd, off", description: "内存映射" },
    { number: 11, name: "munmap", category: "内存管理", params: "addr, len", description: "解除内存映射" },
    { number: 12, name: "brk", category: "内存管理", params: "addr", description: "改变数据段大小" },
    { number: 13, name: "rt_sigaction", category: "信号", params: "sig, act, oact", description: "设置信号处理程序" },
    { number: 39, name: "getpid", category: "进程管理", params: "void", description: "获取进程ID" },
    { number: 57, name: "fork", category: "进程管理", params: "void", description: "创建子进程" },
    { number: 59, name: "execve", category: "进程管理", params: "filename, argv, envp", description: "执行程序" },
    { number: 60, name: "exit", category: "进程管理", params: "status", description: "终止进程" },
    { number: 61, name: "wait4", category: "进程管理", params: "pid, status, options, rusage", description: "等待子进程" },
    { number: 62, name: "kill", category: "信号", params: "pid, sig", description: "发送信号" },
    { number: 72, name: "fcntl", category: "文件I/O", params: "fd, cmd, arg", description: "文件控制操作" },
    { number: 79, name: "getcwd", category: "文件系统", params: "buf, size", description: "获取当前目录" },
    { number: 80, name: "chdir", category: "文件系统", params: "path", description: "改变当前目录" },
    { number: 82, name: "rename", category: "文件系统", params: "oldpath, newpath", description: "重命名文件" },
    { number: 83, name: "mkdir", category: "文件系统", params: "path, mode", description: "创建目录" },
    { number: 84, name: "rmdir", category: "文件系统", params: "path", description: "删除目录" },
    { number: 96, name: "gettimeofday", category: "时间", params: "tv, tz", description: "获取当前时间" },
    { number: 102, name: "getuid", category: "用户/权限", params: "void", description: "获取用户ID" },
    { number: 104, name: "getgid", category: "用户/权限", params: "void", description: "获取组ID" },
    { number: 105, name: "setuid", category: "用户/权限", params: "uid", description: "设置用户ID" },
    { number: 106, name: "setgid", category: "用户/权限", params: "gid", description: "设置组ID" }
  ];

  const categories = ["all", "进程管理", "文件I/O", "文件系统", "内存管理", "信号", "时间", "用户/权限"];

  const filteredSyscalls = syscalls.filter(syscall => {
    const matchesSearch = syscall.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          syscall.description.includes(searchTerm);
    const matchesCategory = selectedCategory === "all" || syscall.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Table className="w-7 h-7 text-teal-600" />
        Linux x86-64 系统调用表查看器
      </h3>

      {/* Search & Filter */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="w-5 h-5 absolute left-3 top-3 text-slate-400" />
              <input
                type="text"
                placeholder="搜索系统调用名称或描述..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border-2 border-slate-300 rounded-lg focus:border-teal-500 focus:outline-none"
              />
            </div>
          </div>
          <div className="flex gap-2 flex-wrap">
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
                  selectedCategory === cat
                    ? "bg-teal-600 text-white"
                    : "bg-slate-200 text-slate-700 hover:bg-slate-300"
                }`}
              >
                {cat === "all" ? "全部" : cat}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-3 text-sm text-slate-600">
          找到 <strong>{filteredSyscalls.length}</strong> 个系统调用
        </div>
      </div>

      {/* Syscall Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-teal-600 text-white">
                <th className="px-4 py-3 text-left font-semibold">编号</th>
                <th className="px-4 py-3 text-left font-semibold">名称</th>
                <th className="px-4 py-3 text-left font-semibold">分类</th>
                <th className="px-4 py-3 text-left font-semibold">参数</th>
                <th className="px-4 py-3 text-left font-semibold">描述</th>
              </tr>
            </thead>
            <tbody>
              {filteredSyscalls.map((syscall, idx) => (
                <motion.tr
                  key={syscall.number}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: idx * 0.02 }}
                  className="border-b border-slate-100 hover:bg-teal-50 transition-colors"
                >
                  <td className="px-4 py-3 font-mono text-teal-700 font-bold">{syscall.number}</td>
                  <td className="px-4 py-3 font-mono font-semibold text-slate-800">{syscall.name}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      syscall.category === "进程管理" ? "bg-blue-200 text-blue-800" :
                      syscall.category === "文件I/O" ? "bg-green-200 text-green-800" :
                      syscall.category === "文件系统" ? "bg-purple-200 text-purple-800" :
                      syscall.category === "内存管理" ? "bg-orange-200 text-orange-800" :
                      syscall.category === "信号" ? "bg-red-200 text-red-800" :
                      syscall.category === "时间" ? "bg-yellow-200 text-yellow-800" :
                      "bg-slate-200 text-slate-800"
                    }`}>
                      {syscall.category}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-slate-600">{syscall.params}</td>
                  <td className="px-4 py-3 text-slate-700">{syscall.description}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Info */}
      <div className="mt-6 bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
        <h4 className="font-bold text-blue-800 mb-2">系统调用表说明</h4>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>编号固定</strong>：每个系统调用有唯一编号，存放在 rax 寄存器</li>
          <li><strong>内核查表</strong>：内核通过 sys_call_table[rax] 跳转到对应处理函数</li>
          <li><strong>架构差异</strong>：不同架构（x86, ARM）的系统调用编号不同</li>
          <li><strong>完整列表</strong>：Linux x86-64 有 300+ 系统调用，这里仅展示常用部分</li>
        </ul>
      </div>
    </div>
  );
}
