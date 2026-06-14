"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { TreePine, Plus, Trash2 } from "lucide-react";

interface HuffmanNode {
  id: number;
  freq: number;
  label: string;
  left?: HuffmanNode;
  right?: HuffmanNode;
}

let nodeId = 0;
function buildHuffmanTree(items: { label: string; freq: number }[]): HuffmanNode | null {
  if (items.length < 2) return null;
  nodeId = 0;
  let nodes: HuffmanNode[] = items.map(it => ({ id: nodeId++, freq: it.freq, label: it.label }));
  while (nodes.length > 1) {
    nodes.sort((a, b) => a.freq - b.freq);
    const left = nodes.shift()!;
    const right = nodes.shift()!;
    nodes.push({ id: nodeId++, freq: left.freq + right.freq, label: "", left, right });
  }
  return nodes[0] || null;
}

function getCodes(node: HuffmanNode | null, prefix = ""): { label: string; code: string; freq: number }[] {
  if (!node) return [];
  if (!node.left && !node.right) return [{ label: node.label, code: prefix || "0", freq: node.freq }];
  return [...getCodes(node.left ?? null, prefix + "0"), ...getCodes(node.right ?? null, prefix + "1")];
}

export function HuffmanCodingViz() {
  const [instructions, setInstructions] = useState([
    { label: "LOAD", freq: 40 },
    { label: "STORE", freq: 20 },
    { label: "ADD", freq: 15 },
    { label: "JMP", freq: 10 },
    { label: "SUB", freq: 8 },
    { label: "MUL", freq: 7 },
  ]);
  const [newLabel, setNewLabel] = useState("");
  const [newFreq, setNewFreq] = useState(5);

  const tree = useMemo(() => buildHuffmanTree(instructions), [instructions]);
  const codes = useMemo(() => {
    const c = getCodes(tree);
    return c.sort((a, b) => a.code.length - b.code.length);
  }, [tree]);

  const fixedBits = Math.ceil(Math.log2(instructions.length || 1));
  const huffmanAvg = codes.length > 0 ? codes.reduce((s, c) => s + c.code.length * c.freq, 0) / instructions.reduce((s, i) => s + i.freq, 0) : 0;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <TreePine className="w-5 h-5 text-green-500" />
        哈夫曼编码可视化
      </h3>

      <div className="flex gap-2 mb-4">
        <input value={newLabel} onChange={e => setNewLabel(e.target.value)} placeholder="指令名"
          className="flex-1 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm" />
        <input type="number" value={newFreq} onChange={e => setNewFreq(Number(e.target.value))} min={1}
          className="w-20 px-2 py-1 rounded border border-border-subtle bg-bg-surface text-sm font-mono" />
        <button onClick={() => { if (newLabel) { setInstructions([...instructions, { label: newLabel, freq: newFreq }]); setNewLabel(""); } }}
          className="px-3 py-1 rounded bg-green-500 text-white text-sm flex items-center gap-1 hover:bg-green-600">
          <Plus className="w-4 h-4" />
        </button>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        {instructions.map((inst, i) => (
          <div key={i} className="flex items-center gap-1 px-2 py-1 rounded bg-bg-surface border border-border-subtle text-sm">
            <span className="font-mono">{inst.label}</span>
            <span className="text-xs text-text-muted">({inst.freq})</span>
            <button onClick={() => setInstructions(instructions.filter((_, j) => j !== i))} className="ml-1 text-text-muted hover:text-red-400">
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>

      {codes.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <table className="w-full text-sm mb-4">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="text-left p-2">指令</th>
                <th className="text-left p-2">频率</th>
                <th className="text-left p-2">哈夫曼编码</th>
                <th className="text-left p-2">码长</th>
              </tr>
            </thead>
            <tbody>
              {codes.map((c, i) => (
                <motion.tr key={c.label} className="border-b border-border-subtle"
                  initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}>
                  <td className="p-2 font-mono">{c.label}</td>
                  <td className="p-2">{c.freq}</td>
                  <td className="p-2 font-mono text-blue-400">{c.code}</td>
                  <td className="p-2">{c.code.length}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded bg-bg-surface border border-border-subtle text-center">
              <div className="text-xs text-text-muted">定长编码</div>
              <div className="text-lg font-mono font-bold">{fixedBits} 位/指令</div>
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle text-center">
              <div className="text-xs text-text-muted">哈夫曼平均码长</div>
              <div className="text-lg font-mono font-bold text-green-400">{huffmanAvg.toFixed(2)} 位/指令</div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
