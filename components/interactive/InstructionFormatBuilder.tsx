"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Puzzle, Plus, X } from "lucide-react";

interface Field {
  id: number;
  name: string;
  bits: number;
  color: string;
}

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

let nextId = 1;

export function InstructionFormatBuilder() {
  const [totalBits, setTotalBits] = useState(16);
  const [fields, setFields] = useState<Field[]>([
    { id: nextId++, name: "操作码", bits: 4, color: COLORS[0] },
    { id: nextId++, name: "地址码", bits: 12, color: COLORS[1] },
  ]);
  const [newName, setNewName] = useState("");
  const [newBits, setNewBits] = useState(4);

  const usedBits = fields.reduce((s, f) => s + f.bits, 0);
  const remaining = totalBits - usedBits;

  const addField = () => {
    if (!newName || newBits <= 0 || newBits > remaining) return;
    setFields([...fields, { id: nextId++, name: newName, bits: newBits, color: COLORS[fields.length % COLORS.length] }]);
    setNewName("");
    setNewBits(4);
  };

  const removeField = (id: number) => {
    setFields(fields.filter(f => f.id !== id));
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Puzzle className="w-5 h-5 text-indigo-500" />
        指令格式构建器
      </h3>

      <div className="flex items-center gap-4 mb-4">
        <label className="text-sm">指令总位数:</label>
        <select value={totalBits} onChange={e => setTotalBits(Number(e.target.value))}
          className="px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm">
          {[8, 16, 32, 64].map(b => <option key={b} value={b}>{b}位</option>)}
        </select>
        <span className="text-xs text-text-muted">已用 {usedBits}/{totalBits} 位，剩余 {remaining} 位</span>
      </div>

      <div className="flex h-10 rounded overflow-hidden mb-4 border border-border-subtle">
        <AnimatePresence mode="popLayout">
          {fields.map(f => (
            <motion.div
              key={f.id}
              layout
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="flex items-center justify-center text-xs font-mono text-white relative group"
              style={{ width: `${(f.bits / totalBits) * 100}%`, backgroundColor: f.color }}
            >
              <span>{f.name}({f.bits}b)</span>
              <button onClick={() => removeField(f.id)}
                className="absolute right-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <X className="w-3 h-3" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
        {remaining > 0 && (
          <div className="flex items-center justify-center text-xs text-text-muted bg-bg-surface"
            style={{ width: `${(remaining / totalBits) * 100}%` }}>
            空闲 {remaining}b
          </div>
        )}
      </div>

      <div className="flex gap-2 mb-4">
        <input value={newName} onChange={e => setNewName(e.target.value)} placeholder="字段名"
          className="flex-1 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm" />
        <input type="number" value={newBits} onChange={e => setNewBits(Number(e.target.value))} min={1} max={remaining}
          className="w-20 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        <button onClick={addField} disabled={remaining <= 0 || !newName}
          className="px-3 py-1 rounded bg-blue-500 text-white text-sm flex items-center gap-1 disabled:opacity-40 hover:bg-blue-600">
          <Plus className="w-4 h-4" /> 添加
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {fields.map(f => (
          <div key={f.id} className="flex items-center gap-2 p-2 rounded bg-bg-surface border border-border-subtle text-sm">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: f.color }} />
            <span>{f.name}</span>
            <span className="ml-auto font-mono text-text-muted">{f.bits}位</span>
            <span className="text-xs text-text-muted">2^{f.bits}={Math.pow(2, f.bits)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
