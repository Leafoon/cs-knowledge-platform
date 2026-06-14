"use client";
import { useState } from "react";

const categories = [
  {
    name: "1xx 信息", color: "blue", codes: [
      { code: 100, text: "Continue", desc: "继续发送请求体" },
      { code: 101, text: "Switching Protocols", desc: "切换协议（如升级WebSocket）" },
      { code: 103, text: "Early Hints", desc: "预加载提示" },
    ],
  },
  {
    name: "2xx 成功", color: "green", codes: [
      { code: 200, text: "OK", desc: "请求成功" },
      { code: 201, text: "Created", desc: "资源已创建" },
      { code: 204, text: "No Content", desc: "成功但无返回内容" },
      { code: 206, text: "Partial Content", desc: "范围请求成功" },
    ],
  },
  {
    name: "3xx 重定向", color: "yellow", codes: [
      { code: 301, text: "Moved Permanently", desc: "永久重定向" },
      { code: 302, text: "Found", desc: "临时重定向" },
      { code: 304, text: "Not Modified", desc: "资源未修改，使用缓存" },
      { code: 307, text: "Temporary Redirect", desc: "临时重定向（保持方法）" },
      { code: 308, text: "Permanent Redirect", desc: "永久重定向（保持方法）" },
    ],
  },
  {
    name: "4xx 客户端错误", color: "orange", codes: [
      { code: 400, text: "Bad Request", desc: "请求语法错误" },
      { code: 401, text: "Unauthorized", desc: "需要认证" },
      { code: 403, text: "Forbidden", desc: "服务器拒绝执行" },
      { code: 404, text: "Not Found", desc: "资源不存在" },
      { code: 405, text: "Method Not Allowed", desc: "请求方法不允许" },
      { code: 429, text: "Too Many Requests", desc: "请求频率超限" },
    ],
  },
  {
    name: "5xx 服务器错误", color: "red", codes: [
      { code: 500, text: "Internal Server Error", desc: "服务器内部错误" },
      { code: 502, text: "Bad Gateway", desc: "网关收到无效响应" },
      { code: 503, text: "Service Unavailable", desc: "服务暂时不可用" },
      { code: 504, text: "Gateway Timeout", desc: "网关超时" },
    ],
  },
];

export function HTTPStatusCodeExplorer() {
  const [active, setActive] = useState("2xx 成功");
  const cat = categories.find((c) => c.name === active)!;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">HTTP 状态码探索器</h3>
      <div className="flex flex-wrap gap-1.5 mb-4">
        {categories.map((c) => (
          <button key={c.name} onClick={() => setActive(c.name)}
            className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
              active === c.name ? `bg-${c.color}-500/20 border border-${c.color}-500 text-${c.color}-400` : "bg-bg-subtle text-text-secondary"
            }`}>
            {c.name}
          </button>
        ))}
      </div>
      <div className="space-y-2">
        {cat.codes.map((c) => (
          <div key={c.code} className="flex items-center gap-3 p-2.5 rounded bg-bg-muted border border-border-subtle">
            <span className={`font-mono font-bold text-lg text-${cat.color}-400 w-12 text-center`}>{c.code}</span>
            <div>
              <p className="text-sm font-medium text-text-primary">{c.text}</p>
              <p className="text-xs text-text-secondary">{c.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
export default HTTPStatusCodeExplorer;
