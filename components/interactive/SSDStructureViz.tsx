"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Cpu, Layers, ArrowRight, RefreshCw } from "lucide-react";

interface Block { id: number; erased: boolean; wear: number }

export function SSDStructureViz() {
  const [blocks] = useState<Block[]>(
    Array.from({ length: 16 }, (_, i) => ({ id: i, erased: i % 3 !== 0, wear: Math.floor(Math.random() * 50) }))
  );
  const [selectedBlock, setSelectedBlock] = useState<number | null>(null);
  const [view, setView] = useState<"structure" | "ftl" | "wear">("structure");

  const logicalPages = [0, 1, 2, 3, 4, 5, 6, 7];
  const ftlMap: Record<number, number> = { 0: 3, 1: 7, 2: 1, 3: 12, 4: 5, 5: 15, 6: 9, 7: 0 };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Cpu className="w-5 h-5 text-sky-500" />
        SSD内部结构
      </h3>
      <div className="flex gap-2 mb-4">
        {(["structure", "ftl", "wear"] as const).map(v => (
          <button key={v} onClick={() => setView(v)}
            className={`px-3 py-1.5 rounded text-sm ${view === v ? "bg-sky-500 text-white" : "bg-bg-subtle"}`}>
            {v === "structure" ? "NAND结构" : v === "ftl" ? "FTL映射" : "磨损均衡"}
          </button>
        ))}
      </div>
      {view === "structure" && (
        <div>
          <p className="text-sm text-text-secondary mb-4">
            SSD由多个闪存芯片组成，每个芯片包含多个Block，每个Block包含多个Page（4KB）。
            写入以Page为单位，擦除以Block为单位。
          </p>
          <div className="grid grid-cols-4 gap-2">
            {blocks.map(b => (
              <motion.div key={b.id} whileHover={{ scale: 1.05 }}
                onClick={() => setSelectedBlock(b.id)}
                className={`p-3 rounded border cursor-pointer ${selectedBlock === b.id ? "border-sky-500 bg-sky-500/20" : b.erased ? "border-green-500/50 bg-green-500/10" : "border-border-subtle bg-bg-subtle"}`}>
                <div className="text-xs font-bold">Block {b.id}</div>
                <div className="text-xs text-text-secondary mt-1">
                  {b.erased ? "已擦除（可写）" : "含数据"}
                </div>
                <div className="mt-1 h-1 bg-bg-elevated rounded overflow-hidden">
                  <div className="h-full bg-sky-500 rounded" style={{ width: `${b.wear}%` }} />
                </div>
              </motion.div>
            ))}
          </div>
          <div className="mt-3 text-xs text-text-secondary">
            <span className="text-green-500">■</span> 已擦除 | <span className="text-text-secondary">■</span> 含数据 | 底部条 = 磨损程度
          </div>
        </div>
      )}
      {view === "ftl" && (
        <div>
          <p className="text-sm text-text-secondary mb-4">
            FTL（闪存转换层）将逻辑页地址映射到物理块地址，使SSD能像块设备一样工作。
          </p>
          <div className="flex items-start gap-8">
            <div>
              <div className="text-xs font-bold mb-2 text-center">逻辑页</div>
              {logicalPages.map(p => (
                <motion.div key={p} whileHover={{ x: 5 }}
                  className="px-3 py-1 mb-1 bg-blue-500/20 border border-blue-500 rounded text-sm text-center">
                  LP {p}
                </motion.div>
              ))}
            </div>
            <div className="flex flex-col justify-center gap-1 mt-4">
              {logicalPages.map(p => (
                <ArrowRight key={p} className="w-4 h-4 text-sky-500" />
              ))}
            </div>
            <div>
              <div className="text-xs font-bold mb-2 text-center">物理块</div>
              {logicalPages.map(p => (
                <motion.div key={p} whileHover={{ x: -5 }}
                  className="px-3 py-1 mb-1 bg-amber-500/20 border border-amber-500 rounded text-sm text-center">
                  PB {ftlMap[p]}
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      )}
      {view === "wear" && (
        <div>
          <p className="text-sm text-text-secondary mb-4">
            磨损均衡算法确保各Block的擦除次数均匀分布，延长SSD寿命。
          </p>
          <div className="space-y-1">
            {blocks.map(b => (
              <div key={b.id} className="flex items-center gap-2">
                <span className="w-16 text-xs text-text-secondary">Block {b.id}</span>
                <div className="flex-1 h-4 bg-bg-subtle rounded overflow-hidden">
                  <motion.div initial={{ width: 0 }} animate={{ width: `${b.wear}%` }}
                    className={`h-full rounded ${b.wear > 80 ? "bg-red-500" : b.wear > 50 ? "bg-amber-500" : "bg-green-500"}`} />
                </div>
                <span className="w-10 text-xs text-right">{b.wear}%</span>
              </div>
            ))}
          </div>
          <div className="mt-3 text-xs text-text-secondary">
            <span className="text-green-500">●</span> 低磨损 | <span className="text-amber-500">●</span> 中磨损 | <span className="text-red-500">●</span> 高磨损
          </div>
        </div>
      )}
    </div>
  );
}
