"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Terminal, ChevronDown, ChevronUp } from "lucide-react";

export default function ExecVariantsComparison() {
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const variants = [
    {
      id: "execl",
      name: "execl()",
      signature: "int execl(const char *path, const char *arg0, ..., NULL)",
      description: "参数以列表形式提供，以 NULL 结尾",
      example: `execl("/bin/ls", "ls", "-l", "-a", NULL);`,
      useCase: "参数数量已知且固定",
      notes: "l = list（列表）"
    },
    {
      id: "execv",
      name: "execv()",
      signature: "int execv(const char *path, char *const argv[])",
      description: "参数以数组形式提供",
      example: `char *args[] = {"ls", "-l", "-a", NULL};
execv("/bin/ls", args);`,
      useCase: "参数数量动态或从数组构建",
      notes: "v = vector（向量/数组）"
    },
    {
      id: "execle",
      name: "execle()",
      signature: "int execle(const char *path, const char *arg0, ..., NULL, char *const envp[])",
      description: "参数列表 + 自定义环境变量",
      example: `char *env[] = {"PATH=/usr/bin", "HOME=/home/user", NULL};
execle("/bin/ls", "ls", "-l", NULL, env);`,
      useCase: "需要自定义环境变量",
      notes: "e = environment（环境）"
    },
    {
      id: "execve",
      name: "execve()",
      signature: "int execve(const char *path, char *const argv[], char *const envp[])",
      description: "参数数组 + 自定义环境变量（真正的系统调用）",
      example: `char *args[] = {"ls", "-l", NULL};
char *env[] = {"PATH=/usr/bin", NULL};
execve("/bin/ls", args, env);`,
      useCase: "完全控制参数和环境变量",
      notes: "唯一真正的系统调用，其他是 wrapper"
    },
    {
      id: "execlp",
      name: "execlp()",
      signature: "int execlp(const char *file, const char *arg0, ..., NULL)",
      description: "参数列表 + PATH 搜索",
      example: `execlp("ls", "ls", "-l", "-a", NULL);`,
      useCase: "不需要完整路径，依赖 PATH 搜索",
      notes: "p = PATH 搜索"
    },
    {
      id: "execvp",
      name: "execvp()",
      signature: "int execvp(const char *file, char *const argv[])",
      description: "参数数组 + PATH 搜索",
      example: `char *args[] = {"ls", "-l", "-a", NULL};
execvp("ls", args);`,
      useCase: "动态参数 + PATH 搜索（最常用）",
      notes: "Shell 命令执行常用"
    }
  ];

  const comparisonTable = [
    { feature: "参数传递", execl: "列表", execv: "数组", execle: "列表", execve: "数组", execlp: "列表", execvp: "数组" },
    { feature: "路径", execl: "完整路径", execv: "完整路径", execle: "完整路径", execve: "完整路径", execlp: "PATH 搜索", execvp: "PATH 搜索" },
    { feature: "环境变量", execl: "继承", execv: "继承", execle: "自定义", execve: "自定义", execlp: "继承", execvp: "继承" },
    { feature: "系统调用", execl: "否（库函数）", execv: "否（库函数）", execle: "否（库函数）", execve: "是", execlp: "否（库函数）", execvp: "否（库函数）" }
  ];

  const toggleRow = (id: string) => {
    setExpandedRow(expandedRow === id ? null : id);
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Terminal className="w-7 h-7 text-teal-600" />
        exec 系列函数对比
      </h3>

      {/* Variants List */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden mb-6">
        {variants.map((variant, idx) => (
          <div key={variant.id} className="border-b border-slate-200 last:border-b-0">
            <motion.div
              onClick={() => toggleRow(variant.id)}
              className={`p-4 cursor-pointer transition-all ${
                expandedRow === variant.id ? "bg-teal-50" : "hover:bg-slate-50"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-teal-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
                      {idx + 1}
                    </div>
                    <div>
                      <div className="font-bold text-slate-800">{variant.name}</div>
                      <div className="text-sm text-slate-600">{variant.description}</div>
                    </div>
                  </div>
                </div>
                {expandedRow === variant.id ? (
                  <ChevronUp className="w-5 h-5 text-teal-600" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-slate-400" />
                )}
              </div>
            </motion.div>

            {expandedRow === variant.id && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="bg-teal-50 p-4 border-t border-teal-200"
              >
                <div className="space-y-3">
                  <div>
                    <div className="text-xs font-semibold text-slate-600 mb-1">函数签名</div>
                    <pre className="bg-white p-2 rounded border border-teal-200 text-xs font-mono text-slate-800 overflow-x-auto">
                      {variant.signature}
                    </pre>
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-slate-600 mb-1">示例代码</div>
                    <pre className="bg-slate-900 text-green-400 p-3 rounded text-xs overflow-x-auto">
                      {variant.example}
                    </pre>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white p-2 rounded border border-teal-200">
                      <div className="text-xs font-semibold text-slate-600">适用场景</div>
                      <div className="text-sm text-slate-700">{variant.useCase}</div>
                    </div>
                    <div className="bg-white p-2 rounded border border-teal-200">
                      <div className="text-xs font-semibold text-slate-600">备注</div>
                      <div className="text-sm text-slate-700">{variant.notes}</div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        ))}
      </div>

      {/* Comparison Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden mb-6">
        <h4 className="font-bold text-slate-800 p-4 bg-slate-100 border-b border-slate-200">功能对比表</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">特性</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">execl</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">execv</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">execle</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">execve</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">execlp</th>
              <th className="px-4 py-3 text-left text-slate-700 font-semibold">execvp</th>
            </tr>
          </thead>
          <tbody>
            {comparisonTable.map((row, idx) => (
              <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="px-4 py-3 font-semibold text-slate-800">{row.feature}</td>
                <td className="px-4 py-3 text-slate-700">{row.execl}</td>
                <td className="px-4 py-3 text-slate-700">{row.execv}</td>
                <td className="px-4 py-3 text-slate-700">{row.execle}</td>
                <td className="px-4 py-3 text-slate-700 font-semibold">{row.execve}</td>
                <td className="px-4 py-3 text-slate-700">{row.execlp}</td>
                <td className="px-4 py-3 text-slate-700">{row.execvp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Naming Convention */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-slate-800 mb-4">命名规则</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
            <div className="font-bold text-blue-700 mb-2">l（list）</div>
            <div className="text-sm text-slate-700">参数以列表形式提供，以 NULL 结尾</div>
          </div>
          <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
            <div className="font-bold text-green-700 mb-2">v（vector）</div>
            <div className="text-sm text-slate-700">参数以数组（向量）形式提供</div>
          </div>
          <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
            <div className="font-bold text-purple-700 mb-2">e（environment）</div>
            <div className="text-sm text-slate-700">可指定自定义环境变量</div>
          </div>
          <div className="bg-orange-50 border-2 border-orange-300 rounded-lg p-4 md:col-span-2">
            <div className="font-bold text-orange-700 mb-2">p（PATH）</div>
            <div className="text-sm text-slate-700">在 PATH 环境变量中搜索可执行文件，无需完整路径</div>
          </div>
        </div>
      </div>

      {/* Key Points */}
      <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 mb-3">关键要点</h4>
        <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
          <li>
            <strong>execve() 是唯一的真正系统调用</strong>，其他都是基于 execve() 的库函数封装。
          </li>
          <li>
            <strong>参数列表 vs 数组</strong>：l 变体用于编译时已知参数数量，v 变体用于动态构建参数。
          </li>
          <li>
            <strong>PATH 搜索</strong>：p 变体会在 PATH 环境变量中搜索可执行文件（如 Shell 行为），
            其他需要完整路径。
          </li>
          <li>
            <strong>环境变量</strong>：e 变体允许自定义环境变量，其他继承父进程环境变量。
          </li>
          <li>
            <strong>Shell 常用 execvp()</strong>：因为 Shell 需要动态参数 + PATH 搜索。
          </li>
        </ul>
      </div>
    </div>
  );
}
