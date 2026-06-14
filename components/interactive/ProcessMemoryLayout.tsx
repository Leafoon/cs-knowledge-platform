"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Info } from "lucide-react";

export default function ProcessMemoryLayout() {
  const [hoveredSection, setHoveredSection] = useState<string | null>(null);

  const memoryLayout = [
    {
      id: "kernel",
      name: "内核空间",
      start: "0x00007FFFFFFFFFFF",
      end: "0x00007FFF00000000",
      color: "bg-red-500",
      textColor: "text-white",
      permission: "内核态访问",
      description: "内核代码和数据，用户态不可访问",
      height: "h-16"
    },
    {
      id: "stack",
      name: "栈（Stack）↓",
      start: "0x00007FFF00000000",
      end: "向下增长",
      color: "bg-blue-500",
      textColor: "text-white",
      permission: "rw-",
      description: "局部变量、函数调用栈、返回地址",
      height: "h-24"
    },
    {
      id: "gap1",
      name: "[未映射区域]",
      start: "动态分配",
      end: "动态分配",
      color: "bg-gray-200",
      textColor: "text-gray-600",
      permission: "---",
      description: "未使用的虚拟地址空间",
      height: "h-12"
    },
    {
      id: "libs",
      name: "共享库",
      start: "动态分配",
      end: "动态分配",
      color: "bg-purple-500",
      textColor: "text-white",
      permission: "r-x / rw-",
      description: "libc.so、ld.so 等动态链接库",
      height: "h-16"
    },
    {
      id: "gap2",
      name: "[未映射区域]",
      start: "动态分配",
      end: "动态分配",
      color: "bg-gray-200",
      textColor: "text-gray-600",
      permission: "---",
      description: "未使用的虚拟地址空间",
      height: "h-12"
    },
    {
      id: "heap",
      name: "堆（Heap）↑",
      start: "向上增长",
      end: "brk 指针",
      color: "bg-green-500",
      textColor: "text-white",
      permission: "rw-",
      description: "动态分配内存（malloc、new）",
      height: "h-20"
    },
    {
      id: "bss",
      name: "BSS 段",
      start: "编译时确定",
      end: "编译时确定",
      color: "bg-yellow-600",
      textColor: "text-white",
      permission: "rw-",
      description: "未初始化的全局变量和静态变量（初始化为 0）",
      height: "h-12"
    },
    {
      id: "data",
      name: "数据段（Data）",
      start: "编译时确定",
      end: "编译时确定",
      color: "bg-orange-500",
      textColor: "text-white",
      permission: "rw-",
      description: "已初始化的全局变量和静态变量",
      height: "h-12"
    },
    {
      id: "text",
      name: "代码段（Text）",
      start: "0x0000000000400000",
      end: "编译时确定",
      color: "bg-indigo-600",
      textColor: "text-white",
      permission: "r-x",
      description: "程序指令（只读、可执行）",
      height: "h-16"
    }
  ];

  const exampleCode = {
    text: "int add(int a, int b) { return a + b; }  // 编译为机器指令",
    data: "int global_var = 42;  // 已初始化全局变量",
    bss: "int uninitialized[1000];  // 未初始化全局变量",
    heap: "int *p = malloc(sizeof(int) * 100);  // 堆分配",
    stack: "void func() { int local = 10; }  // 局部变量在栈上"
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center">
        进程内存布局（Linux x86-64）
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Memory Layout Visualization */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-semibold text-slate-600">高地址</span>
              <span className="text-xs text-slate-500">128TB (64位系统)</span>
            </div>
            <div className="space-y-1">
              {memoryLayout.map((section, idx) => (
                <motion.div
                  key={section.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  onMouseEnter={() => setHoveredSection(section.id)}
                  onMouseLeave={() => setHoveredSection(null)}
                  className={`${section.color} ${section.textColor} ${section.height} rounded-lg p-3 cursor-pointer transition-all hover:shadow-lg ${
                    hoveredSection === section.id ? "scale-105 shadow-xl" : ""
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-bold">{section.name}</span>
                    <span className="text-xs opacity-80">{section.permission}</span>
                  </div>
                  <div className="text-xs opacity-90 mt-1">
                    {section.start} - {section.end}
                  </div>
                </motion.div>
              ))}
            </div>
            <div className="flex items-center justify-between mt-4">
              <span className="text-sm font-semibold text-slate-600">低地址</span>
              <span className="text-xs text-slate-500">0x0000000000400000</span>
            </div>
          </div>
        </div>

        {/* Detail Panel */}
        <div className="lg:col-span-1">
          {hoveredSection ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-lg shadow-md p-4"
            >
              <div className="flex items-center gap-2 mb-3">
                <Info className="w-5 h-5 text-blue-600" />
                <h4 className="font-bold text-slate-800">
                  {memoryLayout.find(s => s.id === hoveredSection)?.name}
                </h4>
              </div>
              <p className="text-sm text-slate-700 mb-3">
                {memoryLayout.find(s => s.id === hoveredSection)?.description}
              </p>
              <div className="space-y-2">
                <div>
                  <span className="text-xs font-semibold text-slate-600">权限：</span>
                  <span className="text-xs font-mono ml-2">
                    {memoryLayout.find(s => s.id === hoveredSection)?.permission}
                  </span>
                </div>
                <div>
                  <span className="text-xs font-semibold text-slate-600">地址范围：</span>
                  <div className="text-xs font-mono mt-1 bg-slate-100 p-2 rounded">
                    {memoryLayout.find(s => s.id === hoveredSection)?.start}
                    <br />-<br />
                    {memoryLayout.find(s => s.id === hoveredSection)?.end}
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="bg-white rounded-lg shadow-md p-4 flex items-center justify-center h-full">
              <p className="text-sm text-slate-500 text-center">
                悬停在内存区域上查看详情
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Code Examples */}
      <div className="mt-6 bg-white rounded-lg shadow-md p-4">
        <h4 className="font-bold text-slate-800 mb-3">代码示例（对应内存段）</h4>
        <div className="space-y-2">
          {Object.entries(exampleCode).map(([segment, code]) => (
            <div key={segment} className="bg-slate-50 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <div
                  className={`w-3 h-3 rounded ${
                    memoryLayout.find(s => s.id === segment)?.color || "bg-gray-400"
                  }`}
                />
                <span className="text-xs font-semibold text-slate-600">
                  {memoryLayout.find(s => s.id === segment)?.name || segment}
                </span>
              </div>
              <pre className="text-xs font-mono text-slate-700 overflow-x-auto">{code}</pre>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
