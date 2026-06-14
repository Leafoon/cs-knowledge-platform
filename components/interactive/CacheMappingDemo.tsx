"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Grid3X3, Shuffle, Layers, ArrowRight } from "lucide-react";

type MappingType = "direct" | "fully" | "set-associative";

const MAPPING_INFO: Record<MappingType, { name: string; desc: string; icon: typeof Grid3X3 }> = {
  direct: { name: "直接映射", desc: "每个主存块只能映射到Cache的一个固定行", icon: Grid3X3 },
  fully: { name: "全相联映射", desc: "每个主存块可以映射到Cache的任意行", icon: Shuffle },
  "set-associative": { name: "组相联映射", desc: "主存块映射到Cache的某一组，组内可任意放置", icon: Layers },
};

function getAddressBits(type: MappingType, addr: number) {
  const offsetBits = 2;
  const cacheLines = 8;
  const offset = addr & 0x3;
  if (type === "direct") {
    const index = (addr >> offsetBits) & (cacheLines - 1);
    const tag = addr >> (offsetBits + 3);
    return { tag, index, offset };
  } else if (type === "fully") {
    const tag = addr >> offsetBits;
    return { tag, index: null, offset };
  } else {
    const sets = cacheLines / 2;
    const index = (addr >> offsetBits) & (sets - 1);
    const tag = addr >> (offsetBits + 2);
    return { tag, index, offset };
  }
}

export function CacheMappingDemo() {
  const [mappingType, setMappingType] = useState<MappingType>("direct");
  const [address, setAddress] = useState(42);
  const info = MAPPING_INFO[mappingType];
  const Icon = info.icon;
  const bits = getAddressBits(mappingType, address);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Icon className="w-5 h-5 text-blue-500" />
        Cache映射方式演示
      </h3>
      <div className="flex gap-2 mb-4">
        {(Object.keys(MAPPING_INFO) as MappingType[]).map(t => (
          <button key={t} onClick={() => setMappingType(t)}
            className={`px-3 py-1.5 rounded text-sm ${mappingType === t ? "bg-blue-500 text-white" : "bg-bg-subtle"}`}>
            {MAPPING_INFO[t].name}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-4">{info.desc}</p>
      <div className="flex items-center gap-3 mb-4">
        <label className="text-sm">地址:</label>
        <input type="range" min={0} max={255} value={address} onChange={e => setAddress(+e.target.value)}
          className="flex-1" />
        <span className="font-mono text-sm w-12">{address}</span>
      </div>
      <motion.div key={mappingType} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="bg-bg-subtle rounded p-4 mb-4">
        <div className="text-sm font-mono mb-2">地址分解: {address.toString(2).padStart(8, "0")}</div>
        <div className="flex gap-1 text-xs">
          {bits.tag !== null && (
            <div className="bg-red-500/20 border border-red-500 rounded px-2 py-1">
              Tag: {bits.tag}
            </div>
          )}
          {bits.index !== null && (
            <div className="bg-green-500/20 border border-green-500 rounded px-2 py-1">
              Index: {bits.index}
            </div>
          )}
          <div className="bg-blue-500/20 border border-blue-500 rounded px-2 py-1">
            Offset: {bits.offset}
          </div>
        </div>
      </motion.div>
      <div className="grid grid-cols-8 gap-1">
        {Array.from({ length: 8 }).map((_, i) => {
          const match = mappingType === "direct"
            ? i === bits.index
            : mappingType === "fully"
            ? true
            : i >= (bits.index ?? 0) * 2 && i < (bits.index ?? 0) * 2 + 2;
          return (
            <motion.div key={i} animate={{ scale: match ? 1.1 : 1, opacity: match ? 1 : 0.4 }}
              className={`p-2 rounded text-center text-xs font-mono border ${match ? "border-blue-500 bg-blue-500/20" : "border-border-subtle bg-bg-subtle"}`}>
              行{i}
            </motion.div>
          );
        })}
      </div>
      <div className="mt-3 text-xs text-text-secondary flex items-center gap-1">
        <ArrowRight className="w-3 h-3" />
        {mappingType === "direct" && `地址${address} → Cache行${bits.index}`}
        {mappingType === "fully" && `地址${address} → 可放入任意行`}
        {mappingType === "set-associative" && `地址${address} → 组${bits.index}（行${(bits.index ?? 0) * 2}-${(bits.index ?? 0) * 2 + 1}）`}
      </div>
    </div>
  );
}
