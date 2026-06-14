"use client";
import { useState } from "react";

interface DomainNode {
  name: string;
  type: "root" | "tld" | "sld" | "sub" | "host";
  children?: DomainNode[];
  expanded?: boolean;
}

const DOMAIN_TREE: DomainNode = {
  name: ". (根域名)",
  type: "root",
  children: [
    { name: ".com", type: "tld", children: [
      { name: "google.com", type: "sld", children: [
        { name: "www.google.com", type: "sub" },
        { name: "mail.google.com", type: "sub" },
        { name: "maps.google.com", type: "sub" },
      ]},
      { name: "github.com", type: "sld", children: [
        { name: "www.github.com", type: "sub" },
      ]},
    ]},
    { name: ".edu", type: "tld", children: [
      { name: "mit.edu", type: "sld", children: [
        { name: "www.mit.edu", type: "sub" },
      ]},
    ]},
    { name: ".cn", type: "tld", children: [
      { name: "baidu.cn", type: "sld", children: [
        { name: "www.baidu.cn", type: "sub" },
      ]},
    ]},
    { name: ".org", type: "tld", children: [
      { name: "wikipedia.org", type: "sld" },
    ]},
  ],
};

const COLORS: Record<string, string> = {
  root: "text-red-500", tld: "text-blue-500", sld: "text-green-500", sub: "text-purple-500", host: "text-orange-500",
};

const BG_COLORS: Record<string, string> = {
  root: "bg-red-100 dark:bg-red-900/30", tld: "bg-blue-100 dark:bg-blue-900/30",
  sld: "bg-green-100 dark:bg-green-900/30", sub: "bg-purple-100 dark:bg-purple-900/30", host: "bg-orange-100 dark:bg-orange-900/30",
};

function TreeNode({ node, depth }: { node: DomainNode; depth: number }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = node.children && node.children.length > 0;

  return (
    <div style={{ marginLeft: depth * 20 }}>
      <div className={`flex items-center gap-2 mb-1 cursor-pointer ${COLORS[node.type]}`}
        onClick={() => hasChildren && setExpanded(!expanded)}>
        {hasChildren && <span className="text-xs w-4">{expanded ? "▼" : "▶"}</span>}
        {!hasChildren && <span className="text-xs w-4">•</span>}
        <span className={`px-2 py-0.5 rounded text-sm font-mono ${BG_COLORS[node.type]}`}>{node.name}</span>
        <span className="text-xs text-text-secondary ml-1">
          {node.type === "root" && "根域名服务器"}
          {node.type === "tld" && "顶级域"}
          {node.type === "sld" && "二级域"}
          {node.type === "sub" && "子域"}
        </span>
      </div>
      {expanded && node.children?.map((child, i) => (
        <TreeNode key={i} node={child} depth={depth + 1} />
      ))}
    </div>
  );
}

export function DomainTreeExplorer() {
  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DNS域名层次树</h3>
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        <TreeNode node={DOMAIN_TREE} depth={0} />
      </div>
      <div className="flex gap-2 flex-wrap text-xs">
        {Object.entries(COLORS).map(([type, color]) => (
          <span key={type} className={`px-2 py-1 rounded ${BG_COLORS[type]} ${color}`}>
            {type === "root" ? "根域" : type === "tld" ? "顶级域(TLD)" : type === "sld" ? "二级域(SLD)" : "子域"}
          </span>
        ))}
      </div>
      <div className="text-xs text-text-secondary mt-3">
        DNS采用层次化命名:根域(.)→顶级域(.com/.cn)→二级域(google.com)→子域(www.google.com)。解析从根服务器开始逐级查询。
      </div>
    </div>
  );
}

export default DomainTreeExplorer;
