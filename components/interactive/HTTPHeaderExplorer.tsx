"use client";
import { useState } from "react";

type HeaderCategory = "request" | "response" | "entity";

const headers: Record<HeaderCategory, { name: string; example: string; desc: string }[]> = {
  request: [
    { name: "Host", example: "example.com:443", desc: "目标主机名和端口（HTTP/1.1必需）" },
    { name: "User-Agent", example: "Mozilla/5.0 (Macintosh; ...)", desc: "客户端软件标识" },
    { name: "Accept", example: "text/html, application/json", desc: "客户端可接受的媒体类型" },
    { name: "Accept-Encoding", example: "gzip, deflate, br", desc: "支持的内容编码" },
    { name: "Authorization", example: "Bearer eyJhbGciOi...", desc: "认证凭据" },
    { name: "Cookie", example: "session=abc123; theme=dark", desc: "之前由服务器设置的Cookie" },
    { name: "Content-Type", example: "application/json", desc: "请求体的媒体类型" },
    { name: "Connection", example: "keep-alive", desc: "连接管理选项" },
    { name: "Cache-Control", example: "no-cache, max-age=3600", desc: "缓存指令" },
  ],
  response: [
    { name: "Content-Type", example: "text/html; charset=utf-8", desc: "响应体的媒体类型" },
    { name: "Content-Length", example: "3426", desc: "响应体字节长度" },
    { name: "Server", example: "nginx/1.24.0", desc: "服务器软件信息" },
    { name: "Set-Cookie", example: "session=xyz; HttpOnly; Secure", desc: "设置Cookie" },
    { name: "Location", example: "https://example.com/new", desc: "重定向目标URL" },
    { name: "Access-Control-Allow-Origin", example: "*", desc: "CORS允许的来源" },
    { name: "Strict-Transport-Security", example: "max-age=31536000", desc: "HSTS安全策略" },
    { name: "ETag", example: "\"33a64df5\"", desc: "资源的版本标识" },
  ],
  entity: [
    { name: "Content-Encoding", example: "gzip", desc: "内容编码方式" },
    { name: "Content-Language", example: "zh-CN", desc: "内容的自然语言" },
    { name: "Last-Modified", example: "Wed, 21 Oct 2025 07:28:00 GMT", desc: "资源最后修改时间" },
    { name: "Expires", example: "Thu, 01 Dec 2025 16:00:00 GMT", desc: "响应过期时间" },
    { name: "Transfer-Encoding", example: "chunked", desc: "传输编码（分块传输等）" },
  ],
};

const categoryLabels: Record<HeaderCategory, string> = { request: "请求头部", response: "响应头部", entity: "实体头部" };

export function HTTPHeaderExplorer() {
  const [category, setCategory] = useState<HeaderCategory>("request");
  const [search, setSearch] = useState("");

  const filtered = headers[category].filter(
    (h) => h.name.toLowerCase().includes(search.toLowerCase()) || h.desc.includes(search)
  );

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">HTTP 头部探索器</h3>
      <div className="flex gap-2 mb-3">
        {(Object.keys(headers) as HeaderCategory[]).map((cat) => (
          <button key={cat} onClick={() => setCategory(cat)}
            className={`px-3 py-1.5 rounded text-sm ${category === cat ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
            {categoryLabels[cat]}
          </button>
        ))}
      </div>
      <input
        type="text" placeholder="搜索头部字段..." value={search} onChange={(e) => setSearch(e.target.value)}
        className="w-full px-3 py-1.5 rounded bg-bg-subtle border border-border-subtle text-sm text-text-primary mb-3 placeholder:text-text-muted"
      />
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {filtered.map((h) => (
          <div key={h.name} className="p-2.5 rounded bg-bg-muted border border-border-subtle">
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm text-blue-400 font-semibold">{h.name}</span>
            </div>
            <p className="text-xs text-text-secondary mt-0.5">{h.desc}</p>
            <p className="font-mono text-xs text-text-muted mt-1 bg-bg-subtle px-2 py-0.5 rounded inline-block">{h.example}</p>
          </div>
        ))}
        {filtered.length === 0 && <p className="text-sm text-text-muted text-center py-4">未找到匹配的头部字段</p>}
      </div>
    </div>
  );
}
export default HTTPHeaderExplorer;
