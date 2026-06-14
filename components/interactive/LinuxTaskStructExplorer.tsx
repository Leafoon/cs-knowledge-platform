"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Search, Database } from "lucide-react";

export default function LinuxTaskStructExplorer() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const taskStructCategories = [
    {
      name: "调度相关",
      count: 30,
      color: "blue",
      fields: [
        { name: "state", type: "volatile long", desc: "进程状态" },
        { name: "policy", type: "unsigned int", desc: "调度策略（SCHED_NORMAL、SCHED_FIFO 等）" },
        { name: "prio", type: "int", desc: "动态优先级" },
        { name: "static_prio", type: "int", desc: "静态优先级（nice 值）" },
        { name: "se", type: "struct sched_entity", desc: "CFS 调度实体（vruntime）" }
      ]
    },
    {
      name: "内存管理",
      count: 20,
      color: "green",
      fields: [
        { name: "mm", type: "struct mm_struct *", desc: "进程内存描述符" },
        { name: "active_mm", type: "struct mm_struct *", desc: "活动内存描述符" },
        { name: "stack", type: "void *", desc: "内核栈指针" }
      ]
    },
    {
      name: "进程关系",
      count: 15,
      color: "purple",
      fields: [
        { name: "pid", type: "pid_t", desc: "进程 ID" },
        { name: "tgid", type: "pid_t", desc: "线程组 ID" },
        { name: "parent", type: "struct task_struct *", desc: "父进程指针" },
        { name: "children", type: "struct list_head", desc: "子进程链表" },
        { name: "sibling", type: "struct list_head", desc: "兄弟进程链表" }
      ]
    },
    {
      name: "文件系统",
      count: 10,
      color: "orange",
      fields: [
        { name: "fs", type: "struct fs_struct *", desc: "文件系统信息（根目录、当前目录）" },
        { name: "files", type: "struct files_struct *", desc: "打开文件表" }
      ]
    },
    {
      name: "信号处理",
      count: 15,
      color: "red",
      fields: [
        { name: "signal", type: "struct signal_struct *", desc: "共享信号处理器" },
        { name: "sighand", type: "struct sighand_struct *", desc: "信号处理函数表" },
        { name: "blocked", type: "sigset_t", desc: "被阻塞的信号集" }
      ]
    },
    {
      name: "权限与安全",
      count: 12,
      color: "teal",
      fields: [
        { name: "cred", type: "const struct cred *", desc: "凭证（UID、GID、Capabilities）" }
      ]
    },
    {
      name: "命名空间",
      count: 8,
      color: "pink",
      fields: [
        { name: "nsproxy", type: "struct nsproxy *", desc: "PID、网络、IPC、UTS、挂载命名空间" }
      ]
    },
    {
      name: "cgroup",
      count: 6,
      color: "yellow",
      fields: [
        { name: "cgroups", type: "struct css_set *", desc: "cgroup 控制组" }
      ]
    },
    {
      name: "性能统计",
      count: 10,
      color: "indigo",
      fields: [
        { name: "utime", type: "u64", desc: "用户态 CPU 时间" },
        { name: "stime", type: "u64", desc: "内核态 CPU 时间" },
        { name: "nvcsw", type: "unsigned long", desc: "自愿上下文切换次数" },
        { name: "nivcsw", type: "unsigned long", desc: "非自愿上下文切换次数" }
      ]
    },
    {
      name: "其他",
      count: 50,
      color: "gray",
      fields: [
        { name: "comm", type: "char[16]", desc: "进程名称（15 字符）" }
      ]
    }
  ];

  const getColorClass = (color: string) => {
    const map: Record<string, { bg: string; border: string; text: string }> = {
      blue: { bg: "bg-blue-100", border: "border-blue-400", text: "text-blue-700" },
      green: { bg: "bg-green-100", border: "border-green-400", text: "text-green-700" },
      purple: { bg: "bg-purple-100", border: "border-purple-400", text: "text-purple-700" },
      orange: { bg: "bg-orange-100", border: "border-orange-400", text: "text-orange-700" },
      red: { bg: "bg-red-100", border: "border-red-400", text: "text-red-700" },
      teal: { bg: "bg-teal-100", border: "border-teal-400", text: "text-teal-700" },
      pink: { bg: "bg-pink-100", border: "border-pink-400", text: "text-pink-700" },
      yellow: { bg: "bg-yellow-100", border: "border-yellow-400", text: "text-yellow-700" },
      indigo: { bg: "bg-indigo-100", border: "border-indigo-400", text: "text-indigo-700" },
      gray: { bg: "bg-gray-100", border: "border-gray-400", text: "text-gray-700" }
    };
    return map[color];
  };

  const filteredCategories = taskStructCategories.filter(cat =>
    cat.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    cat.fields.some(f => f.name.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const totalFields = taskStructCategories.reduce((sum, cat) => sum + cat.count, 0);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Database className="w-7 h-7" />
        Linux task_struct 浏览器
      </h3>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow-md p-4 text-center">
          <div className="text-3xl font-bold text-blue-600">{totalFields}+</div>
          <div className="text-sm text-slate-600">总字段数</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-4 text-center">
          <div className="text-3xl font-bold text-green-600">~1.7KB</div>
          <div className="text-sm text-slate-600">内存占用</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-4 text-center">
          <div className="text-3xl font-bold text-purple-600">{taskStructCategories.length}</div>
          <div className="text-sm text-slate-600">功能分类</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-4 text-center">
          <div className="text-3xl font-bold text-orange-600">Linux 5.x</div>
          <div className="text-sm text-slate-600">内核版本</div>
        </div>
      </div>

      {/* Search */}
      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="搜索字段或分类..."
            className="w-full pl-10 pr-4 py-3 rounded-lg border-2 border-slate-200 focus:border-blue-400 focus:outline-none"
          />
        </div>
      </div>

      {/* Categories */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredCategories.map((category, idx) => (
          <motion.div
            key={category.name}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.05 }}
            whileHover={{ scale: 1.02 }}
            onClick={() => setSelectedCategory(selectedCategory === category.name ? null : category.name)}
            className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
              selectedCategory === category.name
                ? `${getColorClass(category.color).bg} ${getColorClass(category.color).border} shadow-lg`
                : "bg-white border-slate-200 hover:border-slate-300"
            }`}
          >
            <div className="flex justify-between items-center mb-2">
              <h4 className={`font-bold ${getColorClass(category.color).text}`}>
                {category.name}
              </h4>
              <span className="text-sm font-semibold bg-slate-200 px-2 py-1 rounded">
                {category.count}
              </span>
            </div>
            {selectedCategory === category.name && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="mt-3 space-y-2"
              >
                {category.fields.map((field, fidx) => (
                  <div key={fidx} className="bg-white bg-opacity-70 p-2 rounded text-sm">
                    <div className="font-mono font-semibold text-slate-800">{field.name}</div>
                    <div className="text-xs text-slate-600">{field.type}</div>
                    <div className="text-xs text-slate-700 mt-1">{field.desc}</div>
                  </div>
                ))}
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Info */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <p className="text-sm text-slate-700">
          <strong>task_struct</strong> 是 Linux 内核中最复杂的数据结构之一，包含超过 200 个字段，
          管理进程的调度、内存、文件、信号、权限、命名空间、cgroup 等方方面面。
        </p>
      </div>
    </div>
  );
}
