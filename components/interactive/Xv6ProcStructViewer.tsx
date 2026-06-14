"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Code2, Info } from "lucide-react";

export default function Xv6ProcStructViewer() {
  const [selectedField, setSelectedField] = useState<string>("state");

  const xv6ProcFields = [
    {
      name: "lock",
      type: "struct spinlock",
      category: "同步",
      description: "保护 PCB 的自旋锁",
      usage: "修改 PCB 字段前必须持有此锁"
    },
    {
      name: "state",
      type: "enum procstate",
      category: "状态",
      description: "进程状态",
      usage: "UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE"
    },
    {
      name: "chan",
      type: "void *",
      category: "同步",
      description: "睡眠通道",
      usage: "非零表示进程在此通道上睡眠"
    },
    {
      name: "killed",
      type: "int",
      category: "信号",
      description: "进程是否被杀死",
      usage: "非零表示进程应该退出"
    },
    {
      name: "xstate",
      type: "int",
      category: "状态",
      description: "退出状态",
      usage: "父进程通过 wait() 获取"
    },
    {
      name: "pid",
      type: "int",
      category: "身份",
      description: "进程 ID",
      usage: "唯一标识进程"
    },
    {
      name: "parent",
      type: "struct proc *",
      category: "关系",
      description: "父进程指针",
      usage: "用于 wait() 和资源回收"
    },
    {
      name: "kstack",
      type: "uint64",
      category: "内存",
      description: "内核栈虚拟地址",
      usage: "进程在内核态运行时使用"
    },
    {
      name: "sz",
      type: "uint64",
      category: "内存",
      description: "进程内存大小（字节）",
      usage: "虚拟内存总大小"
    },
    {
      name: "pagetable",
      type: "pagetable_t",
      category: "内存",
      description: "用户页表",
      usage: "虚拟地址到物理地址的映射"
    },
    {
      name: "trapframe",
      type: "struct trapframe *",
      category: "CPU",
      description: "用户态寄存器",
      usage: "系统调用/中断时保存用户态寄存器"
    },
    {
      name: "context",
      type: "struct context",
      category: "CPU",
      description: "内核态寄存器",
      usage: "进程切换时保存/恢复内核态寄存器"
    },
    {
      name: "ofile",
      type: "struct file *[NOFILE]",
      category: "文件",
      description: "打开文件数组",
      usage: "最多 16 个文件描述符"
    },
    {
      name: "cwd",
      type: "struct inode *",
      category: "文件",
      description: "当前工作目录",
      usage: "pwd 命令显示的目录"
    },
    {
      name: "name",
      type: "char[16]",
      category: "调试",
      description: "进程名称",
      usage: "用于调试（如 ps 命令）"
    }
  ];

  const categories = Array.from(new Set(xv6ProcFields.map(f => f.category)));
  const categoryColors: Record<string, string> = {
    "同步": "blue",
    "状态": "green",
    "信号": "red",
    "身份": "purple",
    "关系": "orange",
    "内存": "teal",
    "CPU": "pink",
    "文件": "yellow",
    "调试": "gray"
  };

  const getColorClass = (category: string) => {
    const color = categoryColors[category];
    const map: Record<string, { bg: string; border: string; text: string }> = {
      blue: { bg: "bg-blue-100", border: "border-blue-400", text: "text-blue-700" },
      green: { bg: "bg-green-100", border: "border-green-400", text: "text-green-700" },
      red: { bg: "bg-red-100", border: "border-red-400", text: "text-red-700" },
      purple: { bg: "bg-purple-100", border: "border-purple-400", text: "text-purple-700" },
      orange: { bg: "bg-orange-100", border: "border-orange-400", text: "text-orange-700" },
      teal: { bg: "bg-teal-100", border: "border-teal-400", text: "text-teal-700" },
      pink: { bg: "bg-pink-100", border: "border-pink-400", text: "text-pink-700" },
      yellow: { bg: "bg-yellow-100", border: "border-yellow-400", text: "text-yellow-700" },
      gray: { bg: "bg-gray-100", border: "border-gray-400", text: "text-gray-700" }
    };
    return map[color] || map.gray;
  };

  const selectedFieldData = xv6ProcFields.find(f => f.name === selectedField);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Code2 className="w-7 h-7" />
        xv6 struct proc 详解
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Field List */}
        <div className="lg:col-span-2 space-y-2">
          <h4 className="font-bold text-slate-800 mb-3">字段列表（按分类）</h4>
          {categories.map(category => (
            <div key={category} className="mb-4">
              <div className={`text-sm font-semibold mb-2 ${getColorClass(category).text}`}>
                {category}
              </div>
              <div className="space-y-1">
                {xv6ProcFields
                  .filter(f => f.category === category)
                  .map(field => (
                    <motion.div
                      key={field.name}
                      whileHover={{ scale: 1.02 }}
                      onClick={() => setSelectedField(field.name)}
                      className={`p-3 rounded-lg cursor-pointer transition-all border-2 ${
                        selectedField === field.name
                          ? `${getColorClass(category).bg} ${getColorClass(category).border} shadow-md`
                          : "bg-white border-slate-200 hover:border-slate-300"
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span className="font-mono font-semibold text-slate-800">{field.name}</span>
                        <span className="text-xs font-mono text-slate-600">{field.type}</span>
                      </div>
                    </motion.div>
                  ))}
              </div>
            </div>
          ))}
        </div>

        {/* Detail Panel */}
        <div className="lg:col-span-1">
          {selectedFieldData && (
            <motion.div
              key={selectedField}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`p-6 rounded-lg border-2 ${getColorClass(selectedFieldData.category).bg} ${getColorClass(selectedFieldData.category).border}`}
            >
              <div className="flex items-center gap-2 mb-4">
                <Info className="w-5 h-5" />
                <h4 className="text-lg font-bold text-slate-800">字段详情</h4>
              </div>
              <div className="space-y-3">
                <div>
                  <span className="text-sm font-semibold text-slate-600">名称：</span>
                  <p className="font-mono text-slate-800">{selectedFieldData.name}</p>
                </div>
                <div>
                  <span className="text-sm font-semibold text-slate-600">类型：</span>
                  <p className="font-mono text-sm text-slate-800">{selectedFieldData.type}</p>
                </div>
                <div>
                  <span className="text-sm font-semibold text-slate-600">分类：</span>
                  <p className="text-slate-800">{selectedFieldData.category}</p>
                </div>
                <div>
                  <span className="text-sm font-semibold text-slate-600">描述：</span>
                  <p className="text-slate-700">{selectedFieldData.description}</p>
                </div>
                <div>
                  <span className="text-sm font-semibold text-slate-600">用途：</span>
                  <p className="text-slate-700">{selectedFieldData.usage}</p>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Code View */}
      <div className="mt-6 bg-slate-900 rounded-lg p-4 overflow-x-auto">
        <pre className="text-sm text-green-400 font-mono">
{`// kernel/proc.h
struct proc {
  ${xv6ProcFields.map(f => `  ${f.type} ${f.name};${selectedField === f.name ? " // <-- 选中" : ""}`).join("\n")}
};`}
        </pre>
      </div>
    </div>
  );
}
