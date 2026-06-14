"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Search, RotateCcw, AlertTriangle } from "lucide-react";

interface Segment {
  name: string;
  base: number;
  limit: number;
  color: string;
}

const initialSegments: Segment[] = [
  { name: "Code", base: 0x1000, limit: 0x0800, color: "#EF4444" },
  { name: "Data", base: 0x2000, limit: 0x0C00, color: "#F59E0B" },
  { name: "Heap", base: 0x3000, limit: 0x1000, color: "#10B981" },
  { name: "Stack", base: 0x5000, limit: 0x0600, color: "#3B82F6" },
];

const PHYSICAL_MEM_SIZE = 0x7000; // 28KB for visualization

export default function SegmentationVisualizer() {
  const [segments] = useState<Segment[]>(initialSegments);
  const [segSelect, setSegSelect] = useState(0);
  const [offsetInput, setOffsetInput] = useState("0x0200");
  const [step, setStep] = useState(0); // 0=idle, 1=lookup, 2=bounds, 3=compute, 4=done
  const [result, setResult] = useState<{ pa: number; error?: string } | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  const handleTranslate = useCallback(() => {
    const offset = parseInt(offsetInput, 16);
    if (isNaN(offset) || offset < 0) return;

    setIsAnimating(true);
    setStep(1);
    setResult(null);

    setTimeout(() => {
      setStep(2);
      const seg = segments[segSelect];

      setTimeout(() => {
        if (offset >= seg.limit) {
          setStep(4);
          setResult({ pa: 0, error: `Segmentation fault! Offset 0x${offset.toString(16)} exceeds limit 0x${seg.limit.toString(16)}` });
          setIsAnimating(false);
          return;
        }

        setStep(3);
        setTimeout(() => {
          const pa = seg.base + offset;
          setStep(4);
          setResult({ pa });
          setIsAnimating(false);
        }, 600);
      }, 600);
    }, 600);
  }, [offsetInput, segSelect, segments]);

  const handleReset = () => {
    setStep(0);
    setResult(null);
    setIsAnimating(false);
  };

  // Place segments on physical memory bar
  const physicalBlocks = segments.map((seg) => ({
    start: seg.base,
    end: seg.base + seg.limit,
    color: seg.color,
    name: seg.name,
  }));

  // Find gaps (external fragmentation)
  const gaps: { start: number; end: number }[] = [];
  const sorted = [...physicalBlocks].sort((a, b) => a.start - b.start);
  if (sorted.length > 0 && sorted[0].start > 0) {
    gaps.push({ start: 0, end: sorted[0].start });
  }
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].end < sorted[i + 1].start) {
      gaps.push({ start: sorted[i].end, end: sorted[i + 1].start });
    }
  }
  const lastEnd = sorted[sorted.length - 1]?.end || 0;
  if (lastEnd < PHYSICAL_MEM_SIZE) {
    gaps.push({ start: lastEnd, end: PHYSICAL_MEM_SIZE });
  }
  const totalGap = gaps.reduce((s, g) => s + (g.end - g.start), 0);
  const fragPct = ((totalGap / PHYSICAL_MEM_SIZE) * 100).toFixed(1);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-rose-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <div className="flex items-center justify-center gap-3 mb-6">
        <Layers className="w-7 h-7 text-rose-600 dark:text-rose-400" />
        <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100">
          Segmentation Visualizer
        </h2>
      </div>

      {/* Input */}
      <div className="flex flex-wrap gap-4 mb-6 items-end">
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Segment</label>
          <select value={segSelect} onChange={(e) => setSegSelect(Number(e.target.value))}
            disabled={isAnimating}
            className="px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-gray-800 text-slate-800 dark:text-slate-200 text-sm disabled:opacity-50">
            {segments.map((s, i) => (
              <option key={i} value={i}>{s.name} (base=0x{s.base.toString(16)}, limit=0x{s.limit.toString(16)})</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Offset (hex)</label>
          <input type="text" value={offsetInput} onChange={(e) => setOffsetInput(e.target.value)}
            disabled={isAnimating} placeholder="0x0200"
            className="px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-gray-800 text-slate-800 dark:text-slate-200 font-mono w-36 disabled:opacity-50" />
        </div>
        <button onClick={handleTranslate} disabled={isAnimating}
          className="px-4 py-2 bg-rose-600 text-white rounded-lg hover:bg-rose-700 transition flex items-center gap-2 text-sm font-medium disabled:opacity-50">
          <Search className="w-4 h-4" /> Translate
        </button>
        <button onClick={handleReset}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition flex items-center gap-2 text-sm font-medium">
          <RotateCcw className="w-4 h-4" /> Reset
        </button>
      </div>

      {/* Segment Table */}
      <div className="mb-6 bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-gray-700">
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Segment Table</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-100 dark:bg-gray-700">
              <th className="px-3 py-2 text-left text-slate-600 dark:text-slate-300">Segment</th>
              <th className="px-3 py-2 text-left text-slate-600 dark:text-slate-300">Base</th>
              <th className="px-3 py-2 text-left text-slate-600 dark:text-slate-300">Limit</th>
              <th className="px-3 py-2 text-left text-slate-600 dark:text-slate-300">Range</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((seg, i) => (
              <motion.tr key={i}
                className={`${segSelect === i && step >= 1 ? "bg-rose-50 dark:bg-rose-900/20 ring-1 ring-rose-300" : "hover:bg-slate-50 dark:hover:bg-gray-700"}`}
                animate={segSelect === i && step === 1 ? { scale: [1, 1.02, 1] } : {}}
              >
                <td className="px-3 py-2 font-medium">
                  <span className="inline-flex items-center gap-2">
                    <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: seg.color }} />
                    {seg.name}
                  </span>
                </td>
                <td className="px-3 py-2 font-mono text-slate-700 dark:text-slate-200">0x{seg.base.toString(16).toUpperCase().padStart(4, "0")}</td>
                <td className="px-3 py-2 font-mono text-slate-700 dark:text-slate-200">0x{seg.limit.toString(16).toUpperCase().padStart(4, "0")}</td>
                <td className="px-3 py-2 font-mono text-xs text-slate-500">0x{seg.base.toString(16).toUpperCase()} - 0x{(seg.base + seg.limit).toString(16).toUpperCase()}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Translation steps */}
      <div className="flex items-center justify-center gap-2 mb-6">
        {["Lookup Segment", "Check Bounds", "Compute PA"].map((label, i) => (
          <div key={i} className="flex items-center gap-2">
            <motion.div
              className={`px-3 py-1.5 rounded-full text-xs font-medium ${
                step > i + 1 ? "bg-rose-600 text-white" : step === i + 1 ? "bg-rose-200 dark:bg-rose-800 text-rose-700 dark:text-rose-300 ring-2 ring-rose-400" : "bg-slate-200 dark:bg-gray-700 text-slate-500"
              }`}
              animate={step === i + 1 ? { scale: [1, 1.1, 1] } : {}}
              transition={{ duration: 0.5, repeat: step === i + 1 ? Infinity : 0, repeatDelay: 1 }}
            >
              {label}
            </motion.div>
            {i < 2 && <span className="text-slate-400">&rarr;</span>}
          </div>
        ))}
      </div>

      {/* Physical Memory Bar */}
      <div className="mb-6 bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-gray-700">
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Physical Memory</h4>
        <div className="relative h-20 rounded-lg overflow-hidden bg-slate-200 dark:bg-gray-700">
          {/* Segments */}
          {physicalBlocks.map((block, i) => (
            <motion.div key={i}
              className="absolute top-0 h-full flex items-center justify-center text-white text-xs font-mono"
              style={{
                left: `${(block.start / PHYSICAL_MEM_SIZE) * 100}%`,
                width: `${((block.end - block.start) / PHYSICAL_MEM_SIZE) * 100}%`,
                backgroundColor: block.color,
              }}
              initial={{ scaleY: 0 }}
              animate={{ scaleY: 1 }}
              transition={{ delay: i * 0.1 }}
            >
              {block.name}
            </motion.div>
          ))}
          {/* Gaps (external fragmentation) */}
          {gaps.map((gap, i) => (
            <div key={`gap-${i}`}
              className="absolute top-0 h-full bg-slate-300 dark:bg-gray-600 border-dashed border-x border-slate-400 dark:border-gray-500"
              style={{
                left: `${(gap.start / PHYSICAL_MEM_SIZE) * 100}%`,
                width: `${((gap.end - gap.start) / PHYSICAL_MEM_SIZE) * 100}%`,
              }}
            />
          ))}
          {/* Result pointer */}
          {result && !result.error && (
            <motion.div
              className="absolute top-0 w-1 h-full bg-white shadow-lg"
              style={{ left: `${(result.pa / PHYSICAL_MEM_SIZE) * 100}%` }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-white dark:bg-gray-900 text-xs font-mono px-2 py-0.5 rounded shadow text-rose-600 dark:text-rose-400 whitespace-nowrap">
                PA: 0x{result.pa.toString(16).toUpperCase().padStart(4, "0")}
              </div>
            </motion.div>
          )}
        </div>
        <div className="flex justify-between text-xs text-slate-400 mt-1 font-mono">
          <span>0x0000</span>
          <span>0x{PHYSICAL_MEM_SIZE.toString(16).toUpperCase()}</span>
        </div>
      </div>

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className={`p-5 rounded-xl border-2 ${result.error ? "bg-red-50 dark:bg-red-900/20 border-red-400" : "bg-green-50 dark:bg-green-900/20 border-green-400"}`}
          >
            {result.error ? (
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-6 h-6 text-red-500" />
                <div>
                  <div className="text-lg font-bold text-red-700 dark:text-red-300">Segmentation Fault</div>
                  <div className="text-sm text-red-600 dark:text-red-400">{result.error}</div>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white font-bold text-sm">PA</div>
                <div>
                  <div className="text-lg font-bold text-green-700 dark:text-green-300">
                    Physical Address: 0x{result.pa.toString(16).toUpperCase().padStart(4, "0")}
                  </div>
                  <div className="text-sm text-green-600 dark:text-green-400">
                    = base (0x{segments[segSelect].base.toString(16)}) + offset ({offsetInput})
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Fragmentation stats */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
          <div className="text-xs text-amber-600 dark:text-amber-400 mb-1">External Fragmentation</div>
          <div className="text-2xl font-bold text-amber-700 dark:text-amber-300">{fragPct}%</div>
          <div className="text-xs text-slate-500 mt-1">{totalGap} bytes in {gaps.length} free gaps</div>
        </div>
        <div className="p-4 bg-slate-50 dark:bg-gray-800 rounded-lg border border-slate-200 dark:border-gray-700">
          <div className="text-xs text-slate-500 mb-1">Memory Utilization</div>
          <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">{(100 - parseFloat(fragPct)).toFixed(1)}%</div>
          <div className="text-xs text-slate-500 mt-1">Segments consume {(PHYSICAL_MEM_SIZE - totalGap)} bytes</div>
        </div>
      </div>

      {/* Info */}
      <div className="mt-6 p-4 bg-rose-50 dark:bg-rose-900/20 rounded-lg border border-rose-200 dark:border-rose-800">
        <p className="text-sm text-rose-800 dark:text-rose-200">
          <strong>Segmentation:</strong> Memory is divided into variable-sized segments (code, data, heap, stack).
          Each segment has a base address and a limit. The physical address = base + offset, provided offset &lt; limit.
          External fragmentation occurs when free memory is scattered in small chunks between allocated segments.
        </p>
      </div>
    </div>
  );
}
