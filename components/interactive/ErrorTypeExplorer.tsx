"use client";

import React, { useState } from "react";

interface ErrorType {
  name: string;
  phase: string;
  description: string;
  examples: string[];
  detection: string;
  severity: "high" | "medium" | "low";
}

const errorTypes: ErrorType[] = [
  {
    name: "编译时错误",
    phase: "编译期",
    description: "在 TE → TIR → 目标代码 的编译流程中发现的错误",
    examples: [
      "Shape 推导失败: relay.InferType 报错",
      "Dtype 不匹配: Cannot convert float32 to int32",
      "不支持的算子: Operator not registered",
      "Schedule 写法错误: IterVar axis not found",
    ],
    detection: "tvm.lower() / relay.transform.InferType / PassContext",
    severity: "high",
  },
  {
    name: "运行时错误",
    phase: "推理期",
    description: "模型部署后执行推理时出现的错误",
    examples: [
      "CUDA kernel launch 失败: invalid configuration",
      "内存越界: device-side assert triggered",
      "OpenCL build 失败: clBuildProgram error -11",
      "段错误: SIGSEGV in packed_func",
    ],
    detection: "TVM_DEBUG_RUNTIME=1 / cuda-memcheck / DebugExecutor",
    severity: "high",
  },
  {
    name: "性能错误",
    phase: "优化期",
    description: "代码能运行但性能远低于预期",
    examples: [
      "未 vectorize: 循环未使用 SIMD",
      "错误的 tiling: cache miss rate 高",
      "冗余计算: 相同表达式重复求值",
      "同步开销: 过多的 device-host 同步",
    ],
    detection: "Profiling / AutoTVM tuning / Meta Schedule",
    severity: "medium",
  },
  {
    name: "逻辑错误",
    phase: "任意期",
    description: "计算结果不正确，但程序不崩溃",
    examples: [
      "归约轴错误: sum over wrong axis",
      "填充错误: conv2d padding 不对",
      "广播错误: element-wise op 维度不匹配",
      "数值溢出: 大数乘法超出 float32 范围",
    ],
    detection: "数值对比 / DebugExecutor dump / 单元测试",
    severity: "high",
  },
];

const severityColors = { high: "bg-red-600/30 text-red-200", medium: "bg-yellow-600/30 text-yellow-200", low: "bg-green-600/30 text-green-200" };

export function ErrorTypeExplorer() {
  const [active, setActive] = useState(0);
  const et = errorTypes[active];

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-indigo-950/80 via-purple-950/60 to-blue-950/80 backdrop-blur rounded-xl border border-indigo-700/30 shadow-lg my-6">
      <h3 className="text-xl font-bold mb-2 text-indigo-200">错误类型分类探索器</h3>
      <p className="text-sm text-indigo-300/70 mb-4">TVM 编译器中四大类错误的详细分类、检测手段与典型示例。</p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        {errorTypes.map((e, i) => (
          <button
            key={i}
            onClick={() => setActive(i)}
            className={`px-3 py-2 rounded-lg border text-sm transition-colors ${
              active === i
                ? "bg-indigo-600/60 border-indigo-400 text-white"
                : "bg-indigo-900/30 border-indigo-700/40 text-indigo-300 hover:bg-indigo-800/40"
            }`}
          >
            <div className="font-semibold text-xs">{e.name}</div>
            <div className="text-[10px] text-indigo-400 mt-0.5">{e.phase}</div>
          </button>
        ))}
      </div>

      <div className="space-y-3">
        <div className="bg-indigo-900/30 rounded-lg p-4 border border-indigo-800/30">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg font-bold text-indigo-100">{et.name}</span>
            <span className={`px-2 py-0.5 text-[10px] rounded ${severityColors[et.severity]}`}>
              {et.severity === "high" ? "严重" : et.severity === "medium" ? "中等" : "低"}
            </span>
          </div>
          <p className="text-sm text-indigo-200/70">{et.description}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-indigo-950/60 rounded-lg p-3 border border-indigo-800/40">
            <div className="text-xs text-indigo-400 mb-2">典型示例</div>
            <ul className="space-y-1">
              {et.examples.map((ex, i) => (
                <li key={i} className="text-xs text-indigo-200/70 flex items-start gap-1.5">
                  <span className="text-purple-400 mt-0.5">•</span>
                  {ex}
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-indigo-900/20 rounded-lg p-3 border border-indigo-800/20">
            <div className="text-xs text-indigo-400 mb-2">检测手段</div>
            <div className="text-sm text-cyan-300/80 font-mono text-xs">{et.detection}</div>
          </div>
        </div>
      </div>
    </div>
  );
};


