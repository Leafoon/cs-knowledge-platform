"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { MapPin } from "lucide-react";

export function AddressFieldDesigner() {
  const [totalBits, setTotalBits] = useState(16);
  const [addrCount, setAddrCount] = useState(2);
  const [addrBits, setAddrBits] = useState(6);

  const totalAddrBits = addrCount * addrBits;
  const opcodeBits = totalBits - totalAddrBits;
  const maxInstr = opcodeBits > 0 ? Math.pow(2, opcodeBits) : 0;
  const addrSpace = Math.pow(2, addrBits);
  const valid = opcodeBits > 0 && totalAddrBits <= totalBits;

  const presets = [
    { name: "三地址 16位", total: 16, addr: 3, ab: 4 },
    { name: "二地址 16位", total: 16, addr: 2, ab: 6 },
    { name: "一地址 16位", total: 16, addr: 1, ab: 12 },
    { name: "二地址 32位", total: 32, addr: 2, ab: 13 },
  ];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <MapPin className="w-5 h-5 text-pink-500" />
        地址字段设计器
      </h3>

      <div className="flex flex-wrap gap-2 mb-4">
        {presets.map(p => (
          <button key={p.name} onClick={() => { setTotalBits(p.total); setAddrCount(p.addr); setAddrBits(p.ab); }}
            className="px-3 py-1 rounded text-xs bg-bg-surface border border-border-subtle hover:border-blue-400">
            {p.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-xs text-text-muted mb-1">指令总位数</label>
          <select value={totalBits} onChange={e => setTotalBits(Number(e.target.value))}
            className="w-full px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm">
            {[8, 16, 32, 64].map(b => <option key={b} value={b}>{b}位</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">地址个数</label>
          <select value={addrCount} onChange={e => setAddrCount(Number(e.target.value))}
            className="w-full px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm">
            {[0, 1, 2, 3].map(n => <option key={n} value={n}>{n}地址</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">每个地址位数</label>
          <input type="number" value={addrBits} onChange={e => setAddrBits(Math.max(1, Number(e.target.value)))} min={1}
            className="w-full px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        </div>
      </div>

      {valid ? (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="flex h-10 rounded overflow-hidden mb-4 border border-border-subtle">
            <div className="flex items-center justify-center text-xs font-mono text-white bg-blue-500"
              style={{ width: `${(opcodeBits / totalBits) * 100}%` }}>
              OP {opcodeBits}b
            </div>
            {Array.from({ length: addrCount }).map((_, i) => (
              <div key={i} className="flex items-center justify-center text-xs font-mono text-white"
                style={{ width: `${(addrBits / totalBits) * 100}%`, backgroundColor: `hsl(${120 + i * 60}, 60%, 45%)` }}>
                A{i + 1} {addrBits}b
              </div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded bg-bg-surface border border-border-subtle text-center">
              <div className="text-xs text-text-muted">操作码位数</div>
              <div className="text-xl font-mono font-bold text-blue-400">{opcodeBits}</div>
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle text-center">
              <div className="text-xs text-text-muted">最大指令数</div>
              <div className="text-xl font-mono font-bold text-green-400">{maxInstr.toLocaleString()}</div>
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle text-center">
              <div className="text-xs text-text-muted">每地址空间</div>
              <div className="text-xl font-mono font-bold">{addrSpace.toLocaleString()}</div>
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle text-center">
              <div className="text-xs text-text-muted">地址码总位数</div>
              <div className="text-xl font-mono font-bold">{totalAddrBits}</div>
            </div>
          </div>
        </motion.div>
      ) : (
        <div className="p-4 rounded bg-red-500/10 border border-red-500/30 text-sm text-red-400">
          错误：地址码总位数 ({totalAddrBits}) 超过指令总位数 ({totalBits})，请调整参数。
        </div>
      )}
    </div>
  );
}
