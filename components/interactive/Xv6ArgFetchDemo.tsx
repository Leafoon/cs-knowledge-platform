"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Download } from "lucide-react";

export default function Xv6ArgFetchDemo() {
  const [selectedFunc, setSelectedFunc] = useState<"argint" | "argptr" | "argstr">("argint");

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Download className="w-7 h-7 text-emerald-600" />
        xv6 系统调用参数获取
      </h3>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-lg mb-4">参数存储位置</h4>
        <div className="bg-gradient-to-r from-emerald-100 to-teal-100 p-6 rounded-lg border-2 border-emerald-300">
          <div className="text-sm text-slate-800 space-y-3">
            <div className="flex items-center gap-3">
              <div className="bg-emerald-600 text-white rounded px-3 py-1 font-bold">参数 0</div>
              <div className="flex-1">
                <div className="font-mono bg-white p-2 rounded border">trapframe{'->'}esp + 4 (返回地址后)</div>
                <div className="text-xs text-slate-600 mt-1">第一个参数（32位 x86 栈传参）</div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="bg-teal-600 text-white rounded px-3 py-1 font-bold">参数 1</div>
              <div className="flex-1">
                <div className="font-mono bg-white p-2 rounded border">trapframe{'->'}esp + 8</div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="bg-cyan-600 text-white rounded px-3 py-1 font-bold">参数 n</div>
              <div className="flex-1">
                <div className="font-mono bg-white p-2 rounded border">trapframe{'->'}esp + 4 + 4*n</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-center gap-4 mb-6">
        <button onClick={() => setSelectedFunc("argint")} className={`px-6 py-3 rounded-lg font-semibold ${selectedFunc === "argint" ? "bg-emerald-600 text-white" : "bg-slate-200"}`}>argint()</button>
        <button onClick={() => setSelectedFunc("argptr")} className={`px-6 py-3 rounded-lg font-semibold ${selectedFunc === "argptr" ? "bg-teal-600 text-white" : "bg-slate-200"}`}>argptr()</button>
        <button onClick={() => setSelectedFunc("argstr")} className={`px-6 py-3 rounded-lg font-semibold ${selectedFunc === "argstr" ? "bg-cyan-600 text-white" : "bg-slate-200"}`}>argstr()</button>
      </div>

      <motion.div key={selectedFunc} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-white rounded-lg shadow-md p-6 mb-6">
        {selectedFunc === "argint" && (
          <>
            <h4 className="font-bold text-xl mb-4">argint() - 获取整数参数</h4>
            <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto mb-4">
{`int argint(int n, int *ip) {
  return fetchint(
    (myproc()->tf->esp) + 4 + 4*n, ip
  );
}

// 示例：sys_kill()
int sys_kill(void) {
  int pid;
  if (argint(0, &pid) < 0)
    return -1;
  return kill(pid);
}`}
            </pre>
            <div className="bg-emerald-50 border-l-4 border-emerald-400 p-4 rounded">
              <h5 className="font-bold text-emerald-800 mb-2">工作流程</h5>
              <ol className="text-sm text-slate-700 space-y-1 list-decimal list-inside">
                <li>计算参数地址：<code className="bg-white px-2 py-1 rounded">esp + 4 + 4*n</code></li>
                <li>调用 fetchint() 从用户空间读取 4 字节整数</li>
                <li>验证地址有效性（防止访问越界）</li>
                <li>返回参数值</li>
              </ol>
            </div>
          </>
        )}

        {selectedFunc === "argptr" && (
          <>
            <h4 className="font-bold text-xl mb-4">argptr() - 获取指针参数</h4>
            <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto mb-4">
{`int argptr(int n, char **pp, int size) {
  int i;
  if (argint(n, &i) < 0)
    return -1;
  if (size < 0 || 
      (uint)i >= myproc()->sz ||
      (uint)i+size > myproc()->sz)
    return -1;
  *pp = (char*)i;
  return 0;
}

// 示例：sys_read()
int sys_read(void) {
  int fd, n;
  char *p;
  if (argint(0, &fd) < 0 ||
      argint(2, &n) < 0 ||
      argptr(1, &p, n) < 0)
    return -1;
  return fileread(fd, p, n);
}`}
            </pre>
            <div className="bg-teal-50 border-l-4 border-teal-400 p-4 rounded">
              <h5 className="font-bold text-teal-800 mb-2">安全检查</h5>
              <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
                <li><strong>地址合法性</strong>：<code>ptr &lt; proc-&gt;sz</code></li>
                <li><strong>范围检查</strong>：<code>ptr + size ≤ proc-&gt;sz</code></li>
                <li><strong>防止内核越界</strong>：用户不能传递内核地址</li>
                <li>size 参数必须非负</li>
              </ul>
            </div>
          </>
        )}

        {selectedFunc === "argstr" && (
          <>
            <h4 className="font-bold text-xl mb-4">argstr() - 获取字符串参数</h4>
            <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto mb-4">
{`int argstr(int n, char **pp) {
  int addr;
  if (argint(n, &addr) < 0)
    return -1;
  return fetchstr(addr, pp);
}

int fetchstr(uint addr, char **pp) {
  char *s, *ep;
  ep = (char*)myproc()->sz;
  for(s = (char*)addr; s < ep; s++) {
    if(*s == 0) {
      *pp = (char*)addr;
      return s - (char*)addr;
    }
  }
  return -1;
}

// 示例：sys_open()
int sys_open(void) {
  char *path;
  int fd, omode;
  if (argstr(0, &path) < 0 ||
      argint(1, &omode) < 0)
    return -1;
  return open(path, omode);
}`}
            </pre>
            <div className="bg-cyan-50 border-l-4 border-cyan-400 p-4 rounded">
              <h5 className="font-bold text-cyan-800 mb-2">字符串验证</h5>
              <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
                <li><strong>遍历字符</strong>：从 addr 开始逐字节检查</li>
                <li><strong>边界检查</strong>：确保不超出进程地址空间（proc-&gt;sz）</li>
                <li><strong>空字符终止</strong>：必须在合法范围内找到 '\0'</li>
                <li>若未找到空字符返回 -1（防止恶意字符串）</li>
              </ul>
            </div>
          </>
        )}
      </motion.div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h5 className="font-bold text-slate-800 mb-3">参数获取流程对比</h5>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-emerald-600 text-white">
              <th className="px-4 py-3 text-left">函数</th>
              <th className="px-4 py-3 text-left">用途</th>
              <th className="px-4 py-3 text-left">额外验证</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b hover:bg-slate-50">
              <td className="px-4 py-3 font-mono">argint(n, *ip)</td>
              <td>获取整数（fd, pid, size 等）</td>
              <td>基本地址检查</td>
            </tr>
            <tr className="border-b hover:bg-slate-50">
              <td className="px-4 py-3 font-mono">argptr(n, **pp, sz)</td>
              <td>获取指针（buffer 等）</td>
              <td>地址 + 范围检查</td>
            </tr>
            <tr className="hover:bg-slate-50">
              <td className="px-4 py-3 font-mono">argstr(n, **pp)</td>
              <td>获取字符串（路径等）</td>
              <td>逐字节遍历找 '\0'</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h5 className="font-bold text-amber-800 mb-2">安全性</h5>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>防止越界</strong>：所有用户指针必须验证范围</li>
          <li><strong>防止内核攻击</strong>：用户不能传递 &gt;= proc-&gt;sz 的地址</li>
          <li><strong>原子性</strong>：参数获取过程中进程不能被切换</li>
          <li>xv6 使用简单的栈传参（x86-32），现代系统多用寄存器</li>
        </ul>
      </div>
    </div>
  );
}
