"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Zap } from "lucide-react";

export default function VDSOPerformanceComparison() {
  const [selectedSyscall, setSelectedSyscall] = useState<"gettimeofday" | "clock_gettime" | "getcpu">("gettimeofday");

  const syscallData = {
    gettimeofday: {
      traditional: 1200,
      vdso: 50,
      desc: "获取当前时间（微秒精度）",
      speedup: "24x"
    },
    clock_gettime: {
      traditional: 1300,
      vdso: 60,
      desc: "获取高精度时间（纳秒精度）",
      speedup: "21x"
    },
    getcpu: {
      traditional: 1000,
      vdso: 30,
      desc: "获取当前 CPU 编号",
      speedup: "33x"
    }
  };

  const current = syscallData[selectedSyscall];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Zap className="w-7 h-7 text-violet-600" />
        vDSO 性能对比
      </h3>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-lg mb-4">什么是 vDSO？</h4>
        <div className="bg-gradient-to-r from-violet-100 to-purple-100 p-6 rounded-lg border-2 border-violet-300">
          <div className="text-sm text-slate-800 space-y-2">
            <p><strong>vDSO (Virtual Dynamic Shared Object)</strong> 是 Linux 内核映射到每个进程地址空间的一小段代码，用于在用户态直接执行某些系统调用，避免陷入内核。</p>
            <div className="bg-white p-4 rounded mt-3 border border-violet-200">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="font-semibold text-violet-800 mb-2">传统系统调用</div>
                  <div className="text-xs text-slate-600">用户态 → 陷入内核 → 内核处理 → 返回用户态</div>
                  <div className="text-red-600 font-bold mt-1">开销：~1000-1500 周期</div>
                </div>
                <div className="text-3xl text-slate-400 mx-4">vs</div>
                <div className="flex-1">
                  <div className="font-semibold text-green-800 mb-2">vDSO 优化</div>
                  <div className="text-xs text-slate-600">用户态 → 直接调用 vDSO → 读取共享内存</div>
                  <div className="text-green-600 font-bold mt-1">开销：~30-60 周期</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-center gap-4 mb-6">
        <button onClick={() => setSelectedSyscall("gettimeofday")} className={`px-6 py-3 rounded-lg font-semibold ${selectedSyscall === "gettimeofday" ? "bg-violet-600 text-white" : "bg-slate-200"}`}>gettimeofday</button>
        <button onClick={() => setSelectedSyscall("clock_gettime")} className={`px-6 py-3 rounded-lg font-semibold ${selectedSyscall === "clock_gettime" ? "bg-purple-600 text-white" : "bg-slate-200"}`}>clock_gettime</button>
        <button onClick={() => setSelectedSyscall("getcpu")} className={`px-6 py-3 rounded-lg font-semibold ${selectedSyscall === "getcpu" ? "bg-pink-600 text-white" : "bg-slate-200"}`}>getcpu</button>
      </div>

      <motion.div key={selectedSyscall} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-xl mb-4">{selectedSyscall}() 性能对比</h4>
        <p className="text-sm text-slate-600 mb-6">{current.desc}</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="bg-gradient-to-br from-red-100 to-orange-100 rounded-lg p-6 border-2 border-red-300">
            <div className="text-center">
              <div className="text-xs text-slate-600 mb-2">传统系统调用</div>
              <div className="text-5xl font-bold text-red-700 mb-2">{current.traditional}</div>
              <div className="text-sm text-slate-700">CPU 周期</div>
              <div className="mt-4 bg-white p-3 rounded text-xs text-left">
                <div className="font-semibold mb-1">包含开销：</div>
                <ul className="space-y-1 list-disc list-inside text-slate-600">
                  <li>用户态 → 内核态切换：~100 周期</li>
                  <li>保存/恢复上下文：~60 周期</li>
                  <li>内核执行：~100 周期</li>
                  <li>内核态 → 用户态切换：~100 周期</li>
                  <li>其他开销：~840 周期</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-100 to-teal-100 rounded-lg p-6 border-2 border-green-300">
            <div className="text-center">
              <div className="text-xs text-slate-600 mb-2">vDSO 优化版本</div>
              <div className="text-5xl font-bold text-green-700 mb-2">{current.vdso}</div>
              <div className="text-sm text-slate-700">CPU 周期</div>
              <div className="mt-4 bg-white p-3 rounded">
                <div className="text-3xl font-bold text-green-600 mb-2">{current.speedup} 倍提升</div>
                <div className="text-xs text-slate-600">无需陷入内核</div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-slate-50 p-6 rounded-lg border border-slate-200">
          <h5 className="font-semibold text-slate-800 mb-3">周期对比图</h5>
          <div className="space-y-3">
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-semibold text-red-700">传统系统调用</span>
                <span className="text-xs text-slate-600">{current.traditional} 周期</span>
              </div>
              <div className="h-8 bg-red-500 rounded relative flex items-center justify-end pr-2">
                <span className="text-white text-xs font-bold">100%</span>
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-semibold text-green-700">vDSO 优化</span>
                <span className="text-xs text-slate-600">{current.vdso} 周期</span>
              </div>
              <div className="h-8 bg-green-500 rounded relative flex items-center justify-end pr-2" style={{ width: `${(current.vdso / current.traditional) * 100}%` }}>
                <span className="text-white text-xs font-bold">{Math.round((current.vdso / current.traditional) * 100)}%</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h5 className="font-bold text-slate-800 mb-4">vDSO 实现原理</h5>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-sm font-semibold text-slate-700 mb-2">内核映射</div>
            <pre className="bg-slate-900 text-green-400 p-3 rounded text-xs overflow-x-auto">
{`// 内核在进程启动时映射 vDSO
7ffff7ffd000-7ffff7fff000 r-xp 
  [vdso]

// 使用 ldd 查看
$ ldd /bin/ls
  linux-vdso.so.1 => (0x00007ffff7ffd000)
  libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6`}
            </pre>
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-700 mb-2">用户调用</div>
            <pre className="bg-slate-900 text-green-400 p-3 rounded text-xs overflow-x-auto">
{`#include <sys/time.h>

struct timeval tv;
// glibc 自动使用 vDSO 版本
gettimeofday(&tv, NULL);

// 内部：直接读取内核共享的时间戳
// 无需 syscall 指令`}
            </pre>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">支持的系统调用</h5>
          <table className="w-full text-sm">
            <thead><tr className="border-b"><th className="text-left py-2">系统调用</th><th className="text-left py-2">用途</th></tr></thead>
            <tbody>
              <tr className="border-b"><td className="py-2 font-mono">gettimeofday</td><td>获取时间</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">clock_gettime</td><td>高精度时间</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">time</td><td>获取秒级时间</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">getcpu</td><td>获取 CPU 编号</td></tr>
              <tr><td className="py-2 font-mono">__vdso_sgx_enter_enclave</td><td>SGX 支持</td></tr>
            </tbody>
          </table>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">优势</h5>
          <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
            <li><strong>极快</strong>：避免上下文切换开销</li>
            <li><strong>透明</strong>：应用无需修改代码</li>
            <li><strong>可靠</strong>：内核自动更新共享内存</li>
            <li><strong>安全</strong>：只读映射，用户无法修改</li>
            <li>高频调用场景性能提升显著（如时间戳）</li>
          </ul>
        </div>
      </div>

      <div className="bg-violet-50 border-l-4 border-violet-400 p-4 rounded">
        <h5 className="font-bold text-violet-800 mb-2">适用场景</h5>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>高频时间查询</strong>：日志、性能分析、计时器</li>
          <li><strong>CPU 感知调度</strong>：NUMA 优化、线程亲和性</li>
          <li><strong>无副作用操作</strong>：只读内核数据（时间、CPU 编号等）</li>
          <li>xv6 未实现 vDSO（教学简化），现代 Linux 全面支持</li>
        </ul>
      </div>
    </div>
  );
}
