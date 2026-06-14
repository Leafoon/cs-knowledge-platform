"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Code, Zap } from "lucide-react";

export default function SyscallVsLibraryComparison() {
  const [selectedExample, setSelectedExample] = useState<string>("strlen");

  const examples = {
    strlen: {
      name: "strlen() - 纯库函数",
      type: "library",
      description: "计算字符串长度，纯用户态操作",
      code: `size_t strlen(const char *s) {
  const char *p = s;
  while (*p) p++;  // 遍历字符串
  return p - s;
}`,
      syscalls: 0,
      execution: "用户态执行，无系统调用",
      performance: "~10-50 CPU 周期（取决于字符串长度）"
    },
    fopen: {
      name: "fopen() - 库函数封装系统调用",
      type: "wrapper",
      description: "打开文件，内部调用 open() 系统调用",
      code: `FILE *fopen(const char *path, const char *mode) {
  int flags = parse_mode(mode);  // 用户态
  int fd = open(path, flags);    // 系统调用！
  if (fd < 0) return NULL;
  return fdopen(fd);             // 用户态
}`,
      syscalls: 1,
      execution: "用户态 + 内核态（open 系统调用）",
      performance: "~1000-5000 CPU 周期（系统调用开销）"
    },
    open: {
      name: "open() - 直接系统调用",
      type: "syscall",
      description: "打开文件，直接陷入内核",
      code: `int open(const char *path, int flags) {
  // 用户态：准备系统调用
  register long syscall_num = SYS_open;
  register const char *arg1 = path;
  register int arg2 = flags;
  
  // 触发系统调用（陷入内核）
  asm volatile("syscall");
  
  // 内核态：sys_open() 执行
  // 返回用户态
  return fd;  // 文件描述符
}`,
      syscalls: 1,
      execution: "用户态 → 内核态 → 用户态",
      performance: "~1000-5000 CPU 周期"
    }
  };

  const comparisonData = [
    { feature: "执行环境", syscall: "内核态（Ring 0）", library: "用户态（Ring 3）" },
    { feature: "CPU 特权级切换", syscall: "是（User → Kernel → User）", library: "否（始终 User）" },
    { feature: "触发方式", syscall: "syscall / int 0x80 指令", library: "call 指令" },
    { feature: "上下文切换", syscall: "保存/恢复寄存器、栈、PC", library: "仅函数栈帧" },
    { feature: "性能开销", syscall: "高（~1000 周期）", library: "低（~10 周期）" },
    { feature: "安全检查", syscall: "内核验证参数、权限", library: "无（用户态）" },
    { feature: "可移植性", syscall: "POSIX 标准", library: "依赖 C 库实现" },
    { feature: "错误处理", syscall: "返回 -1，设置 errno", library: "视具体函数" }
  ];

  const currentExample = examples[selectedExample as keyof typeof examples];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Code className="w-7 h-7 text-green-600" />
        系统调用 vs 库函数对比
      </h3>

      {/* Example Selector */}
      <div className="flex justify-center gap-4 mb-6">
        {Object.entries(examples).map(([key, example]) => (
          <button
            key={key}
            onClick={() => setSelectedExample(key)}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              selectedExample === key
                ? example.type === "library" ? "bg-blue-600 text-white" :
                  example.type === "wrapper" ? "bg-green-600 text-white" :
                  "bg-red-600 text-white"
                : "bg-slate-200 text-slate-700 hover:bg-slate-300"
            }`}
          >
            {example.name.split(" - ")[0]}
          </button>
        ))}
      </div>

      {/* Code Example */}
      <motion.div
        key={selectedExample}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-md p-6 mb-6"
      >
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-bold text-slate-800">{currentExample.name}</h4>
          <div className={`px-3 py-1 rounded text-xs font-semibold ${
            currentExample.type === "library" ? "bg-blue-200 text-blue-800" :
            currentExample.type === "wrapper" ? "bg-green-200 text-green-800" :
            "bg-red-200 text-red-800"
          }`}>
            {currentExample.syscalls === 0 ? "无系统调用" : `${currentExample.syscalls} 个系统调用`}
          </div>
        </div>
        <p className="text-sm text-slate-600 mb-4">{currentExample.description}</p>
        
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto mb-4">
          {currentExample.code}
        </pre>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-50 p-3 rounded border border-slate-200">
            <div className="text-xs font-semibold text-slate-600 mb-1">执行路径</div>
            <div className="text-sm text-slate-800">{currentExample.execution}</div>
          </div>
          <div className="bg-slate-50 p-3 rounded border border-slate-200">
            <div className="text-xs font-semibold text-slate-600 mb-1">性能开销</div>
            <div className="text-sm text-slate-800 flex items-center gap-2">
              <Zap className={`w-4 h-4 ${currentExample.syscalls === 0 ? "text-green-600" : "text-orange-600"}`} />
              {currentExample.performance}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Comparison Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden mb-6">
        <h4 className="font-bold text-slate-800 p-4 bg-slate-100 border-b border-slate-200">详细对比表</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">对比维度</th>
              <th className="px-4 py-3 text-left text-red-700 font-semibold">系统调用</th>
              <th className="px-4 py-3 text-left text-blue-700 font-semibold">库函数</th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.map((row, idx) => (
              <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="px-4 py-3 font-semibold text-slate-800">{row.feature}</td>
                <td className="px-4 py-3 text-slate-700">{row.syscall}</td>
                <td className="px-4 py-3 text-slate-700">{row.library}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Visual Flow */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h4 className="font-bold text-slate-800 mb-4">执行流程对比</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Library Function */}
          <div className="space-y-3">
            <div className="text-center font-bold text-blue-700">纯库函数（strlen）</div>
            <div className="bg-blue-100 border-2 border-blue-400 rounded-lg p-4">
              <div className="text-sm font-semibold text-blue-800 mb-2">用户态</div>
              <div className="space-y-2">
                <div className="bg-white p-2 rounded text-xs">调用 strlen()</div>
                <div className="bg-white p-2 rounded text-xs">遍历字符串（纯计算）</div>
                <div className="bg-white p-2 rounded text-xs">返回结果</div>
              </div>
            </div>
            <div className="text-center text-xs text-slate-600">无特权级切换</div>
          </div>

          {/* System Call */}
          <div className="space-y-3">
            <div className="text-center font-bold text-red-700">系统调用（open）</div>
            <div className="space-y-2">
              <div className="bg-blue-100 border-2 border-blue-400 rounded-lg p-3">
                <div className="text-sm font-semibold text-blue-800 mb-1">用户态</div>
                <div className="bg-white p-2 rounded text-xs">准备参数、系统调用号</div>
              </div>
              <div className="flex justify-center">
                <div className="w-1 h-6 bg-yellow-400 animate-pulse"></div>
              </div>
              <div className="bg-red-100 border-2 border-red-400 rounded-lg p-3">
                <div className="text-sm font-semibold text-red-800 mb-1">内核态</div>
                <div className="bg-white p-2 rounded text-xs mb-1">保存上下文</div>
                <div className="bg-white p-2 rounded text-xs mb-1">sys_open() 执行</div>
                <div className="bg-white p-2 rounded text-xs">恢复上下文</div>
              </div>
              <div className="flex justify-center">
                <div className="w-1 h-6 bg-yellow-400 animate-pulse"></div>
              </div>
              <div className="bg-blue-100 border-2 border-blue-400 rounded-lg p-3">
                <div className="text-sm font-semibold text-blue-800 mb-1">用户态</div>
                <div className="bg-white p-2 rounded text-xs">返回结果（fd）</div>
              </div>
            </div>
            <div className="text-center text-xs text-slate-600">2次特权级切换</div>
          </div>
        </div>
      </div>

      {/* Key Points */}
      <div className="mt-6 bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 mb-2">关键要点</h4>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>库函数可能调用系统调用</strong>：如 fopen() 内部调用 open()</li>
          <li><strong>性能差异显著</strong>：系统调用开销是普通函数调用的 100-1000 倍</li>
          <li><strong>减少系统调用</strong>：批量 I/O、缓冲、mmap 等技术降低系统调用频率</li>
          <li><strong>POSIX 兼容性</strong>：系统调用跨 Unix-like 系统兼容，库函数依赖具体实现</li>
        </ul>
      </div>
    </div>
  );
}
