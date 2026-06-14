"use client";
import { useState } from "react";

const methods = [
  { method: "GET", desc: "获取资源，不修改服务器状态", safe: true, idempotent: true, hasBody: false, example: "GET /api/users HTTP/1.1", color: "green" },
  { method: "POST", desc: "提交数据，创建新资源或触发处理", safe: false, idempotent: false, hasBody: true, example: 'POST /api/users\n{"name":"Alice","age":25}', color: "blue" },
  { method: "PUT", desc: "替换目标资源（整体更新）", safe: false, idempotent: true, hasBody: true, example: 'PUT /api/users/1\n{"name":"Alice","age":26}', color: "yellow" },
  { method: "DELETE", desc: "删除指定资源", safe: false, idempotent: true, hasBody: false, example: "DELETE /api/users/1 HTTP/1.1", color: "red" },
  { method: "HEAD", desc: "与GET相同但不返回响应体", safe: true, idempotent: true, hasBody: false, example: "HEAD /api/users HTTP/1.1", color: "purple" },
  { method: "OPTIONS", desc: "获取资源支持的通信选项（CORS预检）", safe: true, idempotent: true, hasBody: false, example: "OPTIONS /api/users HTTP/1.1\nAccess-Control-Request-Method: POST", color: "orange" },
  { method: "PATCH", desc: "对资源进行部分修改", safe: false, idempotent: false, hasBody: true, example: 'PATCH /api/users/1\n{"age":27}', color: "teal" },
];

export function HTTPMethodDemo() {
  const [selected, setSelected] = useState(0);
  const m = methods[selected];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">HTTP 方法演示</h3>
      <div className="flex flex-wrap gap-1.5 mb-4">
        {methods.map((m, i) => (
          <button key={m.method} onClick={() => setSelected(i)}
            className={`px-3 py-1.5 rounded text-sm font-mono font-semibold transition-all ${
              i === selected ? `bg-${m.color}-500/20 border border-${m.color}-500 text-${m.color}-400` : "bg-bg-subtle text-text-secondary hover:bg-bg-muted"
            }`}>
            {m.method}
          </button>
        ))}
      </div>
      <div className="p-4 rounded-lg bg-bg-muted border border-border-subtle">
        <div className="flex gap-3 mb-3">
          <span className={`px-2 py-0.5 rounded text-xs ${m.safe ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}>
            {m.safe ? "安全" : "非安全"}
          </span>
          <span className={`px-2 py-0.5 rounded text-xs ${m.idempotent ? "bg-green-500/20 text-green-400" : "bg-yellow-500/20 text-yellow-400"}`}>
            {m.idempotent ? "幂等" : "非幂等"}
          </span>
          <span className={`px-2 py-0.5 rounded text-xs ${m.hasBody ? "bg-blue-500/20 text-blue-400" : "bg-gray-500/20 text-gray-400"}`}>
            {m.hasBody ? "有请求体" : "无请求体"}
          </span>
        </div>
        <p className="text-sm text-text-secondary mb-3">{m.desc}</p>
        <pre className="font-mono text-xs bg-bg-subtle p-3 rounded text-text-muted whitespace-pre-wrap">{m.example}</pre>
      </div>
    </div>
  );
}
export default HTTPMethodDemo;
