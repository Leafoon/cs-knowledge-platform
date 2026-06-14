"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface AppDomain {
  name: string;
  compute: number;
  memory: number;
  io: number;
  realtime: number;
  parallel: number;
  color: string;
}

const domains: AppDomain[] = [
  { name: "科学计算", compute: 95, memory: 70, io: 30, realtime: 20, parallel: 90, color: "#667eea" },
  { name: "工业控制", compute: 40, memory: 20, io: 60, realtime: 95, parallel: 30, color: "#10b981" },
  { name: "网络服务", compute: 50, memory: 80, io: 90, realtime: 60, parallel: 85, color: "#f59e0b" },
  { name: "人工智能", compute: 95, memory: 90, io: 40, realtime: 30, parallel: 95, color: "#ef4444" },
  { name: "嵌入式系统", compute: 20, memory: 15, io: 70, realtime: 90, parallel: 20, color: "#8b5cf6" },
  { name: "大数据分析", compute: 70, memory: 95, io: 95, realtime: 20, parallel: 90, color: "#ec4899" },
];

const axes = [
  { key: "compute" as const, label: "计算能力", angle: 0 },
  { key: "memory" as const, label: "内存需求", angle: 72 },
  { key: "io" as const, label: "I/O 能力", angle: 144 },
  { key: "realtime" as const, label: "实时性", angle: 216 },
  { key: "parallel" as const, label: "并行能力", angle: 288 },
];

const cx = 200;
const cy = 200;
const r = 150;

function polar(angleDeg: number, radius: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
}

function buildPath(domain: AppDomain): string {
  const vals = axes.map((a) => domain[a.key] / 100);
  return axes
    .map((a, i) => {
      const p = polar(a.angle, vals[i] * r);
      return `${i === 0 ? "M" : "L"}${p.x},${p.y}`;
    })
    .join(" ") + " Z";
}

export function ComputerApplicationRadar() {
  const [active, setActive] = useState<string[]>([domains[0].name, domains[3].name]);

  function toggle(name: string) {
    setActive((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
    );
  }

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        计算机应用领域雷达图
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        对比不同应用领域对各项性能指标的需求
      </p>

      <div className="flex flex-wrap gap-2 mb-4">
        {domains.map((d) => (
          <button
            key={d.name}
            onClick={() => toggle(d.name)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              active.includes(d.name)
                ? "text-white border-transparent"
                : "bg-bg-secondary text-text-secondary border-border-subtle"
            }`}
            style={active.includes(d.name) ? { backgroundColor: d.color } : {}}
          >
            {d.name}
          </button>
        ))}
      </div>

      <div className="flex flex-col lg:flex-row gap-6 items-center">
        <svg viewBox="0 0 400 400" className="w-full max-w-[400px]">
          {/* grid rings */}
          {[0.2, 0.4, 0.6, 0.8, 1.0].map((s) => (
            <polygon
              key={s}
              points={axes.map((a) => { const p = polar(a.angle, s * r); return `${p.x},${p.y}`; }).join(" ")}
              fill="none"
              stroke="#374151"
              strokeWidth={0.5}
              opacity={0.5}
            />
          ))}
          {/* axis lines */}
          {axes.map((a) => {
            const p = polar(a.angle, r);
            const lp = polar(a.angle, r + 22);
            return (
              <g key={a.key}>
                <line x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="#374151" strokeWidth={0.5} />
                <text x={lp.x} y={lp.y} textAnchor="middle" dominantBaseline="middle"
                  className="fill-text-secondary" fontSize={10} fontWeight={500}>
                  {a.label}
                </text>
              </g>
            );
          })}
          {/* data polygons */}
          {domains
            .filter((d) => active.includes(d.name))
            .map((d) => (
              <motion.polygon
                key={d.name}
                points={axes.map((a) => { const p = polar(a.angle, (d[a.key] / 100) * r); return `${p.x},${p.y}`; }).join(" ")}
                fill={d.color}
                fillOpacity={0.15}
                stroke={d.color}
                strokeWidth={2}
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.5 }}
                style={{ transformOrigin: `${cx}px ${cy}px` }}
              />
            ))}
          {/* data dots */}
          {domains
            .filter((d) => active.includes(d.name))
            .map((d) =>
              axes.map((a) => {
                const p = polar(a.angle, (d[a.key] / 100) * r);
                return <circle key={`${d.name}-${a.key}`} cx={p.x} cy={p.y} r={3} fill={d.color} />;
              })
            )}
        </svg>

        {/* legend table */}
        <div className="flex-1 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="text-left py-1.5 px-2 text-text-secondary">领域</th>
                {axes.map((a) => (
                  <th key={a.key} className="text-center py-1.5 px-2 text-text-secondary">{a.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {domains.filter((d) => active.includes(d.name)).map((d) => (
                <tr key={d.name} className="border-b border-border-subtle/50">
                  <td className="py-1.5 px-2 font-medium text-text-primary">
                    <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: d.color }} />
                    {d.name}
                  </td>
                  {axes.map((a) => (
                    <td key={a.key} className="text-center py-1.5 px-2 font-mono text-text-primary">
                      {d[a.key]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
