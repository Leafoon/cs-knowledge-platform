"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Share2, Copy } from "lucide-react";

export default function MemorySharingVisualization() {
  const [selectedResource, setSelectedResource] = useState<string | null>(null);

  const resources = [
    {
      id: "pid",
      name: "PID",
      sharing: "独立",
      color: "blue",
      parent: "1234",
      child: "1235",
      description: "每个进程有唯一的 PID"
    },
    {
      id: "code",
      name: "代码段",
      sharing: "共享",
      color: "green",
      parent: "0x400000 (物理页 0x1000)",
      child: "0x400000 (物理页 0x1000)",
      description: "只读，父子进程共享同一物理页"
    },
    {
      id: "data",
      name: "数据段",
      sharing: "COW 共享",
      color: "yellow",
      parent: "0x600000 (物理页 0x2000)",
      child: "0x600000 (物理页 0x2000，写时复制)",
      description: "初始共享，首次写入时复制"
    },
    {
      id: "heap",
      name: "堆",
      sharing: "COW 共享",
      color: "orange",
      parent: "0x700000 (物理页 0x3000)",
      child: "0x700000 (物理页 0x3000，写时复制)",
      description: "初始共享，首次写入时复制"
    },
    {
      id: "stack",
      name: "栈",
      sharing: "COW 共享",
      color: "purple",
      parent: "0x7FFF0000 (物理页 0x4000)",
      child: "0x7FFF0000 (物理页 0x4000，写时复制)",
      description: "初始共享，首次写入时复制"
    },
    {
      id: "files",
      name: "文件描述符",
      sharing: "共享",
      color: "red",
      parent: "[stdin, stdout, file.txt] (引用计数 2)",
      child: "[stdin, stdout, file.txt] (引用计数 2)",
      description: "父子进程共享文件表项"
    },
    {
      id: "file_offset",
      name: "文件偏移",
      sharing: "共享",
      color: "teal",
      parent: "offset = 100",
      child: "offset = 100",
      description: "父子进程共享文件偏移（一个 lseek() 影响另一个）"
    }
  ];

  const getColorClass = (color: string) => {
    const map: Record<string, { bg: string; border: string; text: string }> = {
      blue: { bg: "bg-blue-100", border: "border-blue-400", text: "text-blue-700" },
      green: { bg: "bg-green-100", border: "border-green-400", text: "text-green-700" },
      yellow: { bg: "bg-yellow-100", border: "border-yellow-400", text: "text-yellow-700" },
      orange: { bg: "bg-orange-100", border: "border-orange-400", text: "text-orange-700" },
      purple: { bg: "bg-purple-100", border: "border-purple-400", text: "text-purple-700" },
      red: { bg: "bg-red-100", border: "border-red-400", text: "text-red-700" },
      teal: { bg: "bg-teal-100", border: "border-teal-400", text: "text-teal-700" }
    };
    return map[color];
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Share2 className="w-7 h-7 text-blue-600" />
        fork() 后父子进程的内存关系
      </h3>

      {/* Legend */}
      <div className="flex justify-center gap-6 mb-6 flex-wrap">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-500 rounded"></div>
          <span className="text-sm text-slate-700">独立</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded"></div>
          <span className="text-sm text-slate-700">完全共享</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-yellow-500 rounded"></div>
          <span className="text-sm text-slate-700">COW 共享（写时复制）</span>
        </div>
      </div>

      {/* Resource Comparison Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden mb-6">
        <table className="w-full">
          <thead>
            <tr className="bg-slate-100 border-b-2 border-slate-200">
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">资源类型</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">共享方式</th>
              <th className="px-4 py-3 text-left text-green-700 font-semibold">父进程</th>
              <th className="px-4 py-3 text-left text-purple-700 font-semibold">子进程</th>
            </tr>
          </thead>
          <tbody>
            {resources.map((resource, idx) => (
              <motion.tr
                key={resource.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                onClick={() => setSelectedResource(resource.id)}
                className={`border-b border-slate-100 cursor-pointer transition-all ${
                  selectedResource === resource.id ? `${getColorClass(resource.color).bg}` : "hover:bg-slate-50"
                }`}
              >
                <td className="px-4 py-3 font-semibold text-slate-800">{resource.name}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${
                    resource.sharing === "独立" ? "bg-blue-200 text-blue-700" :
                    resource.sharing === "共享" ? "bg-green-200 text-green-700" :
                    "bg-yellow-200 text-yellow-700"
                  }`}>
                    {resource.sharing}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm font-mono text-slate-700">{resource.parent}</td>
                <td className="px-4 py-3 text-sm font-mono text-slate-700">{resource.child}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Detail Panel */}
      {selectedResource && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-6 rounded-lg border-2 ${
            getColorClass(resources.find(r => r.id === selectedResource)!.color).bg
          } ${getColorClass(resources.find(r => r.id === selectedResource)!.color).border}`}
        >
          <h4 className="font-bold text-slate-800 mb-2 flex items-center gap-2">
            {resources.find(r => r.id === selectedResource)?.sharing === "共享" ||
             resources.find(r => r.id === selectedResource)?.sharing === "COW 共享" ? (
              <Share2 className="w-5 h-5" />
            ) : (
              <Copy className="w-5 h-5" />
            )}
            {resources.find(r => r.id === selectedResource)?.name}
          </h4>
          <p className="text-sm text-slate-700">
            {resources.find(r => r.id === selectedResource)?.description}
          </p>
        </motion.div>
      )}

      {/* Visual Diagram */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Parent Process */}
        <div className="bg-white rounded-lg shadow-md p-4">
          <h4 className="font-bold text-green-700 mb-3 text-center">父进程 (PID 1234)</h4>
          <div className="space-y-2">
            <div className="bg-blue-100 p-2 rounded text-sm">PID: 1234 (独立)</div>
            <div className="bg-green-100 p-2 rounded text-sm">代码段: 共享物理页 0x1000</div>
            <div className="bg-yellow-100 p-2 rounded text-sm">数据段: 共享 (COW)</div>
            <div className="bg-orange-100 p-2 rounded text-sm">堆: 共享 (COW)</div>
            <div className="bg-purple-100 p-2 rounded text-sm">栈: 共享 (COW)</div>
            <div className="bg-red-100 p-2 rounded text-sm">文件: 共享文件表</div>
          </div>
        </div>

        {/* Child Process */}
        <div className="bg-white rounded-lg shadow-md p-4">
          <h4 className="font-bold text-purple-700 mb-3 text-center">子进程 (PID 1235)</h4>
          <div className="space-y-2">
            <div className="bg-blue-100 p-2 rounded text-sm">PID: 1235 (独立)</div>
            <div className="bg-green-100 p-2 rounded text-sm">代码段: 共享物理页 0x1000</div>
            <div className="bg-yellow-100 p-2 rounded text-sm">数据段: 共享 (COW)</div>
            <div className="bg-orange-100 p-2 rounded text-sm">堆: 共享 (COW)</div>
            <div className="bg-purple-100 p-2 rounded text-sm">栈: 共享 (COW)</div>
            <div className="bg-red-100 p-2 rounded text-sm">文件: 共享文件表</div>
          </div>
        </div>
      </div>

      {/* Info Box */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <p className="text-sm text-slate-700">
          <strong>写时复制（COW）</strong>：fork() 后，父子进程初始共享内存页（标记为只读）。
          首次写入时触发缺页异常，内核复制该页并恢复可写权限。这大幅提升了 fork() 性能（~1ms vs ~1000ms）。
        </p>
      </div>
    </div>
  );
}
