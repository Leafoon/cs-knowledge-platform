"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Database } from "lucide-react";

export default function SyscallParameterPassing() {
  const [selectedMethod, setSelectedMethod] = useState<"register" | "stack" | "memory">("register");

  const methods = {
    register: {
      name: "寄存器传递（主流）",
      description: "参数通过 CPU 寄存器传递，快速高效",
      maxParams: 6,
      advantages: ["速度快（无内存访问）", "硬件优化", "无需额外复制"],
      disadvantages: ["参数数量受限（通常 ≤6）", "大结构体无法传递"],
      example: `// x86-64 寄存器约定
rax = 系统调用号（如 SYS_write = 1）
rdi = 参数1（fd）
rsi = 参数2（buffer 指针）
rdx = 参数3（count）
r10 = 参数4
r8  = 参数5
r9  = 参数6

// 示例：write(1, "hello", 5)
mov rax, 1              ; SYS_write
mov rdi, 1              ; fd = 1 (stdout)
lea rsi, [msg]          ; buffer = &msg
mov rdx, 5              ; count = 5
syscall`,
      validation: `// 内核验证
if (fd < 0 || fd >= NOFILE) return -EBADF;
if (!user_readable(buffer, count)) return -EFAULT;`
    },
    stack: {
      name: "栈传递（x86 32位）",
      description: "参数通过栈传递，兼容性好但较慢",
      maxParams: "无限制",
      advantages: ["支持任意数量参数", "兼容性好"],
      disadvantages: ["速度慢（需访问内存）", "需复制参数到栈"],
      example: `// x86 (32位) 栈传递
push 5                  ; 参数3: count
push msg                ; 参数2: buffer
push 1                  ; 参数1: fd
mov eax, 4              ; SYS_write
int 0x80
add esp, 12             ; 清理栈（3个参数 × 4字节）

// 栈布局（从高地址到低地址）
[esp+8] = 5         ; count
[esp+4] = msg       ; buffer
[esp+0] = 1         ; fd`,
      validation: `// 内核从用户栈读取
int fd = *(int*)(user_esp + 0);
void *buf = *(void**)(user_esp + 4);
size_t count = *(size_t*)(user_esp + 8);`
    },
    memory: {
      name: "内存结构体传递（复杂场景）",
      description: "通过指向结构体的指针传递大量参数",
      maxParams: "无限制",
      advantages: ["支持复杂数据结构", "适合大量参数"],
      disadvantages: ["需验证用户空间指针", "需复制整个结构体到内核"],
      example: `// 用户态：构造参数结构体
struct io_params {
    int fd;
    void *buffer;
    size_t count;
    off_t offset;
};

struct io_params params = {
    .fd = 1,
    .buffer = msg,
    .count = 5,
    .offset = 0
};

// 传递结构体指针
syscall(SYS_pwrite, &params);

// 内核态：复制并验证
struct io_params kparams;
copy_from_user(&kparams, user_ptr, sizeof(kparams));
if (!valid_fd(kparams.fd)) return -EBADF;`,
      validation: `// 内核必须验证
1. 指针在用户空间范围
2. 结构体可读
3. 嵌套指针（如 buffer）也需验证`
    }
  };

  const current = methods[selectedMethod];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-cyan-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <ArrowRight className="w-7 h-7 text-cyan-600" />
        系统调用参数传递方式
      </h3>

      {/* Method Selector */}
      <div className="flex justify-center gap-4 mb-6 flex-wrap">
        <button
          onClick={() => setSelectedMethod("register")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            selectedMethod === "register"
              ? "bg-blue-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          寄存器传递
        </button>
        <button
          onClick={() => setSelectedMethod("stack")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            selectedMethod === "stack"
              ? "bg-green-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          栈传递
        </button>
        <button
          onClick={() => setSelectedMethod("memory")}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            selectedMethod === "memory"
              ? "bg-purple-600 text-white"
              : "bg-slate-200 text-slate-700 hover:bg-slate-300"
          }`}
        >
          内存结构体传递
        </button>
      </div>

      {/* Method Detail */}
      <motion.div
        key={selectedMethod}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-md p-6 mb-6"
      >
        <h4 className="font-bold text-slate-800 text-xl mb-2">{current.name}</h4>
        <p className="text-sm text-slate-600 mb-6">{current.description}</p>

        {/* Characteristics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
            <div className="font-bold text-green-800 mb-2 flex items-center gap-2">
              <Database className="w-5 h-5" />
              优点
            </div>
            <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
              {current.advantages.map((adv, idx) => (
                <li key={idx}>{adv}</li>
              ))}
            </ul>
          </div>
          <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
            <div className="font-bold text-red-800 mb-2">缺点</div>
            <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
              {current.disadvantages.map((dis, idx) => (
                <li key={idx}>{dis}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Code Example */}
        <div className="mb-6">
          <div className="text-sm font-semibold text-slate-700 mb-2">代码示例</div>
          <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
            {current.example}
          </pre>
        </div>

        {/* Validation */}
        <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
          <div className="font-bold text-amber-800 mb-2">内核验证</div>
          <pre className="bg-white p-3 rounded border border-amber-200 text-sm font-mono overflow-x-auto">
            {current.validation}
          </pre>
        </div>
      </motion.div>

      {/* Visual Comparison */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-slate-800 mb-4">传递方式可视化对比</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Register */}
          <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
            <div className="font-bold text-blue-800 mb-3 text-center">寄存器传递</div>
            <div className="space-y-2">
              <div className="bg-white p-2 rounded border border-blue-200 text-xs">
                <div className="font-semibold">rax = SYS_write</div>
              </div>
              <div className="bg-white p-2 rounded border border-blue-200 text-xs">
                <div className="font-semibold">rdi = fd (1)</div>
              </div>
              <div className="bg-white p-2 rounded border border-blue-200 text-xs">
                <div className="font-semibold">rsi = buffer</div>
              </div>
              <div className="bg-white p-2 rounded border border-blue-200 text-xs">
                <div className="font-semibold">rdx = count (5)</div>
              </div>
            </div>
            <div className="mt-3 text-xs text-center text-blue-700">
              ⚡ 最快（无内存访问）
            </div>
          </div>

          {/* Stack */}
          <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
            <div className="font-bold text-green-800 mb-3 text-center">栈传递</div>
            <div className="space-y-2">
              <div className="bg-white p-2 rounded border border-green-200 text-xs">
                <div className="font-semibold">[esp+8] = count (5)</div>
              </div>
              <div className="bg-white p-2 rounded border border-green-200 text-xs">
                <div className="font-semibold">[esp+4] = buffer</div>
              </div>
              <div className="bg-white p-2 rounded border border-green-200 text-xs">
                <div className="font-semibold">[esp+0] = fd (1)</div>
              </div>
              <div className="bg-slate-200 p-2 rounded border border-green-200 text-xs text-center">
                ↓ 用户栈
              </div>
            </div>
            <div className="mt-3 text-xs text-center text-green-700">
              📚 需访问内存
            </div>
          </div>

          {/* Memory Struct */}
          <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
            <div className="font-bold text-purple-800 mb-3 text-center">结构体传递</div>
            <div className="space-y-2">
              <div className="bg-white p-2 rounded border border-purple-200 text-xs">
                <div className="font-semibold">struct io_params</div>
              </div>
              <div className="bg-white p-2 rounded border border-purple-200 text-xs">
                <div className="pl-2">fd: 1</div>
                <div className="pl-2">buffer: 0x...</div>
                <div className="pl-2">count: 5</div>
                <div className="pl-2">offset: 0</div>
              </div>
              <div className="bg-white p-2 rounded border border-purple-200 text-xs text-center">
                rdi = &params
              </div>
            </div>
            <div className="mt-3 text-xs text-center text-purple-700">
              🔄 需复制结构体
            </div>
          </div>
        </div>
      </div>

      {/* Security Considerations */}
      <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded">
        <h4 className="font-bold text-red-800 mb-3">安全考虑（内核必须验证）</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-700">
          <div>
            <div className="font-semibold mb-2">指针验证</div>
            <ul className="space-y-1 list-disc list-inside">
              <li>指针必须在用户空间范围（&lt; TASK_SIZE）</li>
              <li>禁止访问内核空间（防止提权）</li>
              <li>检查页表映射有效性</li>
            </ul>
          </div>
          <div>
            <div className="font-semibold mb-2">参数验证</div>
            <ul className="space-y-1 list-disc list-inside">
              <li>文件描述符范围检查（0 ~ NOFILE-1）</li>
              <li>长度检查（防止整数溢出）</li>
              <li>权限验证（UID/GID）</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
