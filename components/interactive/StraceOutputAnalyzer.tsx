"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Search, Play, Pause } from "lucide-react";

export default function StraceOutputAnalyzer() {
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);

  const straceOutput = [
    { call: 'execve("/bin/ls", ["ls", "-l"], 0x7ffc...)', ret: '0', time: '0.000045', desc: '执行 ls 程序' },
    { call: 'brk(NULL)', ret: '0x55c9a0', time: '0.000023', desc: '获取堆顶地址' },
    { call: 'openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC)', ret: '3', time: '0.000067', desc: '打开动态链接器缓存' },
    { call: 'fstat(3, {st_mode=S_IFREG|0644, st_size=28503, ...})', ret: '0', time: '0.000034', desc: '获取文件状态' },
    { call: 'mmap(NULL, 28503, PROT_READ, MAP_PRIVATE, 3, 0)', ret: '0x7f8a...', time: '0.000056', desc: '映射文件到内存' },
    { call: 'close(3)', ret: '0', time: '0.000021', desc: '关闭 fd=3' },
    { call: 'openat(AT_FDCWD, ".", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY)', ret: '3', time: '0.000089', desc: '打开当前目录' },
    { call: 'getdents64(3, /* 15 entries */, 32768)', ret: '480', time: '0.000112', desc: '读取目录项' },
    { call: 'write(1, "total 48\\n-rw-r--r-- 1 user...", 256)', ret: '256', time: '0.000134', desc: '输出到 stdout' },
    { call: 'close(3)', ret: '0', time: '0.000019', desc: '关闭目录 fd' },
    { call: 'exit_group(0)', ret: '?', time: '0.000045', desc: '进程退出' }
  ];

  const currentCalls = straceOutput.slice(0, running ? step + 1 : straceOutput.length);

  React.useEffect(() => {
    if (running && step < straceOutput.length - 1) {
      const timer = setTimeout(() => setStep(step + 1), 800);
      return () => clearTimeout(timer);
    } else if (step >= straceOutput.length - 1) {
      setRunning(false);
    }
  }, [running, step]);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Search className="w-7 h-7 text-rose-600" />
        strace 输出分析器
      </h3>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-bold text-lg">strace ls -l</h4>
          <div className="flex gap-3">
            <button onClick={() => { setRunning(!running); if (step >= straceOutput.length - 1) setStep(0); }} className={`px-6 py-2 rounded-lg font-semibold flex items-center gap-2 ${running ? "bg-rose-600 text-white" : "bg-green-600 text-white"}`}>
              {running ? <><Pause className="w-4 h-4" />暂停</> : <><Play className="w-4 h-4" />播放</>}
            </button>
            <button onClick={() => { setStep(0); setRunning(false); }} className="px-6 py-2 bg-slate-300 rounded-lg font-semibold">重置</button>
          </div>
        </div>

        <div className="bg-slate-900 text-green-400 p-4 rounded font-mono text-xs overflow-x-auto max-h-96 overflow-y-auto">
          {currentCalls.map((line, idx) => (
            <motion.div key={idx} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className={`mb-1 ${idx === step && running ? "bg-yellow-900" : ""}`}>
              <span className="text-cyan-400">{line.call}</span>
              <span className="text-purple-400"> = {line.ret}</span>
              <span className="text-slate-500"> &lt;{line.time}&gt;</span>
            </motion.div>
          ))}
          {running && step < straceOutput.length && (
            <motion.div animate={{ opacity: [1, 0.3, 1] }} transition={{ repeat: Infinity, duration: 0.8 }}>_</motion.div>
          )}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h5 className="font-bold text-slate-800 mb-4">系统调用详解</h5>
        <div className="space-y-3">
          {currentCalls.map((call, idx) => (
            <motion.div key={idx} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className={`p-4 rounded-lg border-2 ${idx === step && running ? "border-rose-500 bg-rose-50" : "border-slate-200 bg-slate-50"}`}>
              <div className="flex items-start gap-3">
                <div className="bg-rose-600 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">{idx + 1}</div>
                <div className="flex-1">
                  <div className="font-mono text-sm text-slate-800 mb-1">{call.call.split('(')[0]}()</div>
                  <div className="text-xs text-slate-600">{call.desc}</div>
                  <div className="flex gap-4 mt-2 text-xs">
                    <div><span className="text-slate-500">返回:</span> <span className="font-semibold text-green-700">{call.ret}</span></div>
                    <div><span className="text-slate-500">耗时:</span> <span className="font-semibold text-purple-700">{call.time}s</span></div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">常用 strace 选项</h5>
          <table className="w-full text-sm">
            <thead><tr className="border-b"><th className="text-left py-2">选项</th><th className="text-left py-2">说明</th></tr></thead>
            <tbody>
              <tr className="border-b"><td className="py-2 font-mono">-e trace=open,read</td><td>仅跟踪指定系统调用</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">-c</td><td>统计系统调用次数和耗时</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">-p &lt;pid&gt;</td><td>附加到运行中的进程</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">-f</td><td>跟踪子进程</td></tr>
              <tr><td className="py-2 font-mono">-T</td><td>显示每个系统调用耗时</td></tr>
            </tbody>
          </table>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">统计输出示例（strace -c）</h5>
          <pre className="bg-slate-900 text-green-400 p-3 rounded text-xs overflow-x-auto">
{`% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- --------
 45.23    0.000234          15        15           write
 22.34    0.000115          23         5           openat
 18.92    0.000098          19         5         1 access
  8.21    0.000042          14         3           close
  5.30    0.000027          27         1           execve
------ ----------- ----------- --------- --------- --------
100.00    0.000516                    29         1 total`}
          </pre>
        </div>
      </div>

      <div className="bg-rose-50 border-l-4 border-rose-400 p-4 rounded">
        <h5 className="font-bold text-rose-800 mb-2">strace 应用场景</h5>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>调试</strong>：查看程序执行了哪些系统调用，定位失败原因（如 ENOENT）</li>
          <li><strong>性能分析</strong>：找出耗时系统调用（如频繁 read/write）</li>
          <li><strong>学习</strong>：理解程序行为（如 ls 如何列出目录）</li>
          <li><strong>安全审计</strong>：检测可疑系统调用（如 execve 注入）</li>
          <li>类似工具：ltrace（库函数调用）、perf（性能事件）</li>
        </ul>
      </div>
    </div>
  );
}
