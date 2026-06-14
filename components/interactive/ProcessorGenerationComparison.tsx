"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface Processor {
  id: string;
  name: string;
  year: number;
  wordLen: number;
  transistors: string;
  process: string;
  freq: string;
  cores: number;
  arch: string;
}

const processors: Processor[] = [
  { id: "p1", name: "Intel 4004", year: 1971, wordLen: 4, transistors: "2.3K", process: "10μm", freq: "740KHz", cores: 1, arch: "MCS-4" },
  { id: "p2", name: "Intel 8080", year: 1974, wordLen: 8, transistors: "4.5K", process: "6μm", freq: "2MHz", cores: 1, arch: "8080" },
  { id: "p3", name: "Intel 8086", year: 1978, wordLen: 16, transistors: "29K", process: "3μm", freq: "5MHz", cores: 1, arch: "x86" },
  { id: "p4", name: "80386", year: 1985, wordLen: 32, transistors: "275K", process: "1μm", freq: "16MHz", cores: 1, arch: "IA-32" },
  { id: "p5", name: "Pentium", year: 1993, wordLen: 32, transistors: "3.1M", process: "800nm", freq: "60MHz", cores: 1, arch: "P5" },
  { id: "p6", name: "Pentium 4", year: 2000, wordLen: 32, transistors: "42M", process: "180nm", freq: "1.5GHz", cores: 1, arch: "NetBurst" },
  { id: "p7", name: "Core 2 Duo", year: 2006, wordLen: 64, transistors: "291M", process: "65nm", freq: "2.4GHz", cores: 2, arch: "Core" },
  { id: "p8", name: "i7-3770K", year: 2012, wordLen: 64, transistors: "1.4B", process: "22nm", freq: "3.5GHz", cores: 4, arch: "Ivy Bridge" },
  { id: "p9", name: "Apple M1", year: 2020, wordLen: 64, transistors: "16B", process: "5nm", freq: "3.2GHz", cores: 8, arch: "ARM v8.5" },
  { id: "p10", name: "Apple M4", year: 2024, wordLen: 64, transistors: "28B", process: "3nm", freq: "4.4GHz", cores: 10, arch: "ARM v9.2" },
];

const fields: { key: keyof Processor; label: string }[] = [
  { key: "year", label: "年份" },
  { key: "wordLen", label: "字长(bit)" },
  { key: "transistors", label: "晶体管" },
  { key: "process", label: "制程" },
  { key: "freq", label: "主频" },
  { key: "cores", label: "核心数" },
  { key: "arch", label: "架构" },
];

export function ProcessorGenerationComparison() {
  const [selected, setSelected] = useState<string[]>(["p1", "p10"]);

  function toggle(id: string) {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((s) => s !== id);
      if (prev.length >= 4) return [...prev.slice(1), id];
      return [...prev, id];
    });
  }

  const active = processors.filter((p) => selected.includes(p.id));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        处理器代际对比
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        选择最多 4 款处理器进行参数对比
      </p>

      <div className="flex flex-wrap gap-2 mb-6">
        {processors.map((p) => (
          <button
            key={p.id}
            onClick={() => toggle(p.id)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              selected.includes(p.id)
                ? "bg-accent-primary text-white border-accent-primary"
                : "bg-bg-secondary text-text-secondary border-border-subtle hover:border-accent-primary"
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>

      {active.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="text-left py-2 px-3 text-text-secondary font-medium w-24">特性</th>
                {active.map((p) => (
                  <th key={p.id} className="text-center py-2 px-3 text-text-primary font-semibold">
                    {p.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {fields.map((f, fi) => (
                <motion.tr
                  key={f.key}
                  initial={{ opacity: 0, x: -5 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: fi * 0.03 }}
                  className="border-b border-border-subtle/50"
                >
                  <td className="py-2 px-3 text-text-secondary font-medium text-xs">{f.label}</td>
                  {active.map((p) => (
                    <td key={p.id} className="py-2 px-3 text-center font-mono text-xs text-text-primary">
                      {String(p[f.key])}
                    </td>
                  ))}
                </motion.tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      )}
    </div>
  );
}
