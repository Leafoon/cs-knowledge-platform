'use client';

import { useState } from "react";

interface JsonPart {
  key: string;
  value: string;
  color: string;
  desc: string;
}

const jsonParts: JsonPart[] = [
  { key: "kind", value: '"cuda"', color: "text-violet-400", desc: "必填: 目标平台类型" },
  { key: "arch", value: '"sm_80"', color: "text-indigo-400", desc: "GPU 架构版本 (Ampere)" },
  { key: "max_shared_memory", value: "49152", color: "text-blue-400", desc: "每 block 最大共享内存 (bytes)" },
  { key: "max_threads_per_block", value: "1024", color: "text-cyan-400", desc: "每 block 最大线程数" },
  { key: "thread_warp_size", value: "32", color: "text-teal-400", desc: "Warp 大小" },
];

export default function TargetJSONParser() {
  const [hovered, setHovered] = useState<number | null>(null);
  const [showParsed, setShowParsed] = useState(false);

  const jsonStr = `{
  "kind": "cuda",
  "arch": "sm_80",
  "max_shared_memory": 49152,
  "max_threads_per_block": 1024,
  "thread_warp_size": 32
}`;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        Target JSON 解析
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">
            JSON 输入
          </h4>
          <pre className="bg-slate-900 p-4 rounded-xl text-sm font-mono leading-relaxed">
            {jsonParts.map((part, i) => (
              <div
                key={part.key}
                className={`transition-all cursor-pointer ${
                  hovered === i ? "bg-slate-700/50 -mx-2 px-2 rounded" : ""
                }`}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
              >
                <span className="text-slate-400">{"  "}&quot;</span>
                <span className={part.color}>{part.key}</span>
                <span className="text-slate-400">&quot;: </span>
                <span className="text-amber-300">{part.value}</span>
                <span className="text-slate-500">
                  {i < jsonParts.length - 1 ? "," : ""}
                </span>
              </div>
            ))}
            <span className="text-slate-400">&#125;</span>
          </pre>
        </div>

        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">
            解析结果
          </h4>
          <div className="space-y-2">
            {jsonParts.map((part, i) => (
              <div
                key={part.key}
                className={`flex items-center gap-3 p-2 rounded-lg transition-all ${
                  hovered === i
                    ? "bg-indigo-100 dark:bg-indigo-900/40 scale-[1.02]"
                    : "bg-white dark:bg-slate-800"
                }`}
              >
                <span className={`font-mono text-sm font-bold ${part.color}`}>
                  {part.key}
                </span>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  →
                </span>
                <span className="font-mono text-sm text-amber-600 dark:text-amber-400">
                  {part.value}
                </span>
                <span className="text-xs text-slate-400 dark:text-slate-500 ml-auto">
                  {part.desc}
                </span>
              </div>
            ))}
          </div>

          <button
            onClick={() => setShowParsed(!showParsed)}
            className="mt-4 px-4 py-2 bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg text-sm font-bold transition-colors"
          >
            {showParsed ? "隐藏" : "显示"} Python 调用
          </button>
          {showParsed && (
            <pre className="mt-2 bg-slate-900 text-green-400 p-3 rounded-lg text-xs font-mono">
              {`target = tvm.target.Target(
    '${jsonStr.replace(/\n/g, "\n    ")}'
)
print(target.kind)   # "cuda"
print(target.arch)   # "sm_80"`}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
