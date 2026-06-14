"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Layers, RotateCcw, Search } from "lucide-react";

interface SegEntry {
  name: string;
  base: number;
  limit: number;
  pageTableBase: number;
  color: string;
}

interface PageEntry {
  vpn: number;
  pfn: number;
  valid: boolean;
}

const segTable: SegEntry[] = [
  { name: "Code", base: 0x0000, limit: 0x1000, pageTableBase: 0x5000, color: "#EF4444" },
  { name: "Data", base: 0x2000, limit: 0x1000, pageTableBase: 0x6000, color: "#F59E0B" },
  { name: "Heap", base: 0x4000, limit: 0x2000, pageTableBase: 0x7000, color: "#10B981" },
  { name: "Stack", base: 0x8000, limit: 0x0800, pageTableBase: 0x8000, color: "#3B82F6" },
];

function makePageTable(base: number): PageEntry[] {
  return [
    { vpn: 0, pfn: (base / 0x100) + 0x10, valid: true },
    { vpn: 1, pfn: (base / 0x100) + 0x12, valid: true },
    { vpn: 2, pfn: (base / 0x100) + 0x14, valid: true },
    { vpn: 3, pfn: 0, valid: false },
    { vpn: 4, pfn: (base / 0x100) + 0x18, valid: true },
    { vpn: 5, pfn: 0, valid: false },
    { vpn: 6, pfn: (base / 0x100) + 0x1C, valid: true },
    { vpn: 7, pfn: (base / 0x100) + 0x1E, valid: true },
  ];
}

export default function SegmentPageTranslation() {
  const [segIdx, setSegIdx] = useState(0);
  const [pageIdx, setPageIdx] = useState(0);
  const [offset, setOffset] = useState("0x040");
  const [step, setStep] = useState(0); // 0=idle, 1=seg lookup, 2=linear addr, 3=page lookup, 4=done
  const [result, setResult] = useState<{ pa: number; linearAddr: number } | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  const seg = segTable[segIdx];
  const pageTable = makePageTable(seg.pageTableBase);

  const handleTranslate = useCallback(() => {
    const off = parseInt(offset, 16);
    if (isNaN(off) || off < 0) return;

    setIsAnimating(true);
    setResult(null);

    // Step 1: Segment lookup
    setStep(1);
    setTimeout(() => {
      // Step 2: Compute linear address
      setStep(2);
      const linearAddr = seg.base + (pageIdx * 0x100) + off;

      setTimeout(() => {
        // Step 3: Page table lookup
        setStep(3);
        const pte = pageTable.find((e) => e.vpn === pageIdx);

        setTimeout(() => {
          if (!pte || !pte.valid) {
            setStep(4);
            setResult({ pa: -1, linearAddr });
            setIsAnimating(false);
            return;
          }
          const pa = (pte.pfn * 0x100) + (off & 0xFF);
          setStep(4);
          setResult({ pa, linearAddr });
          setIsAnimating(false);
        }, 800);
      }, 800);
    }, 800);
  }, [segIdx, pageIdx, offset, seg, pageTable]);

  const handleReset = () => {
    setStep(0);
    setResult(null);
    setIsAnimating(false);
  };

  const linearAddr = seg.base + (pageIdx * 0x100) + (parseInt(offset, 16) || 0);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <div className="flex items-center justify-center gap-3 mb-6">
        <Layers className="w-7 h-7 text-teal-600 dark:text-teal-400" />
        <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100">
          Segment + Page Two-Level Translation
        </h2>
      </div>

      {/* Input */}
      <div className="flex flex-wrap gap-4 mb-6 items-end">
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Segment</label>
          <select value={segIdx} onChange={(e) => setSegIdx(Number(e.target.value))} disabled={isAnimating}
            className="px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-gray-800 text-slate-800 dark:text-slate-200 text-sm disabled:opacity-50">
            {segTable.map((s, i) => <option key={i} value={i}>{s.name}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Page Index</label>
          <select value={pageIdx} onChange={(e) => setPageIdx(Number(e.target.value))} disabled={isAnimating}
            className="px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-gray-800 text-slate-800 dark:text-slate-200 text-sm disabled:opacity-50">
            {[0, 1, 2, 3, 4, 5, 6, 7].map((p) => <option key={p} value={p}>Page {p}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Offset (hex)</label>
          <input type="text" value={offset} onChange={(e) => setOffset(e.target.value)} disabled={isAnimating}
            className="px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-gray-800 text-slate-800 dark:text-slate-200 font-mono text-sm w-24 disabled:opacity-50" />
        </div>
        <button onClick={handleTranslate} disabled={isAnimating}
          className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition flex items-center gap-2 text-sm font-medium disabled:opacity-50">
          <Search className="w-4 h-4" /> Translate
        </button>
        <button onClick={handleReset}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition flex items-center gap-2 text-sm font-medium">
          <RotateCcw className="w-4 h-4" /> Reset
        </button>
      </div>

      {/* Logical address breakdown */}
      <div className="mb-6 p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-slate-200 dark:border-gray-700">
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Logical Address</h4>
        <div className="flex items-center gap-3 font-mono text-sm">
          <div className={`px-4 py-2 rounded-lg text-center ${step >= 1 ? "bg-rose-100 dark:bg-rose-900/30 ring-2 ring-rose-400" : "bg-slate-100 dark:bg-gray-700"}`}>
            <div className="text-xs text-slate-500 mb-1">Segment</div>
            <div className="font-bold text-slate-800 dark:text-slate-200">{seg.name} [{segIdx}]</div>
          </div>
          <span className="text-slate-400">+</span>
          <div className={`px-4 py-2 rounded-lg text-center ${step >= 2 ? "bg-blue-100 dark:bg-blue-900/30 ring-2 ring-blue-400" : "bg-slate-100 dark:bg-gray-700"}`}>
            <div className="text-xs text-slate-500 mb-1">Page</div>
            <div className="font-bold text-slate-800 dark:text-slate-200">Page {pageIdx}</div>
          </div>
          <span className="text-slate-400">+</span>
          <div className={`px-4 py-2 rounded-lg text-center ${step >= 2 ? "bg-amber-100 dark:bg-amber-900/30 ring-2 ring-amber-400" : "bg-slate-100 dark:bg-gray-700"}`}>
            <div className="text-xs text-slate-500 mb-1">Offset</div>
            <div className="font-bold text-slate-800 dark:text-slate-200">{offset}</div>
          </div>
        </div>
      </div>

      {/* Side by side: Segment Table and Page Table */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Segment Table */}
        <motion.div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-gray-700"
          animate={{ borderColor: step === 1 ? "#14B8A6" : undefined }}>
          <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Segment Table</h4>
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-100 dark:bg-gray-700">
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">Seg</th>
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">Base</th>
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">Limit</th>
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">PT Base</th>
              </tr>
            </thead>
            <tbody>
              {segTable.map((s, i) => (
                <motion.tr key={i}
                  className={segIdx === i && step >= 1 ? "bg-teal-50 dark:bg-teal-900/20 ring-1 ring-teal-300" : "hover:bg-slate-50 dark:hover:bg-gray-700"}>
                  <td className="px-2 py-1.5 font-medium flex items-center gap-1">
                    <span className="w-2 h-2 rounded-sm" style={{ backgroundColor: s.color }} />
                    {s.name}
                  </td>
                  <td className="px-2 py-1.5 font-mono">0x{s.base.toString(16).padStart(4, "0")}</td>
                  <td className="px-2 py-1.5 font-mono">0x{s.limit.toString(16).padStart(4, "0")}</td>
                  <td className="px-2 py-1.5 font-mono">0x{s.pageTableBase.toString(16).padStart(4, "0")}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </motion.div>

        {/* Page Table */}
        <motion.div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-gray-700"
          animate={{ borderColor: step === 3 ? "#14B8A6" : undefined }}>
          <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
            Page Table (for {seg.name}, base: 0x{seg.pageTableBase.toString(16).padStart(4, "0")})
          </h4>
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-100 dark:bg-gray-700">
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">VPN</th>
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">PFN</th>
                <th className="px-2 py-1.5 text-center text-slate-600 dark:text-slate-300">Valid</th>
              </tr>
            </thead>
            <tbody>
              {pageTable.map((pte, i) => (
                <motion.tr key={i}
                  className={pageIdx === pte.vpn && step >= 3 ? "bg-teal-50 dark:bg-teal-900/20 ring-1 ring-teal-300" : "hover:bg-slate-50 dark:hover:bg-gray-700"}>
                  <td className="px-2 py-1.5 font-mono">0x{pte.vpn.toString(16).padStart(2, "0")}</td>
                  <td className="px-2 py-1.5 font-mono">{pte.valid ? `0x${pte.pfn.toString(16).padStart(2, "0")}` : "---"}</td>
                  <td className="px-2 py-1.5 text-center">{pte.valid ? "Y" : "N"}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      </div>

      {/* Steps indicator */}
      <div className="flex items-center justify-center gap-2 mb-6">
        {["Seg Lookup", "Linear Addr", "Page Lookup", "Physical Addr"].map((label, i) => (
          <div key={i} className="flex items-center gap-2">
            <motion.div
              className={`px-3 py-1.5 rounded-full text-xs font-medium ${
                step > i + 1 ? "bg-teal-600 text-white" : step === i + 1 ? "bg-teal-200 dark:bg-teal-800 text-teal-700 dark:text-teal-300 ring-2 ring-teal-400" : "bg-slate-200 dark:bg-gray-700 text-slate-500"
              }`}
              animate={step === i + 1 ? { scale: [1, 1.08, 1] } : {}}
              transition={{ duration: 0.6, repeat: step === i + 1 ? Infinity : 0, repeatDelay: 0.8 }}
            >
              {label}
            </motion.div>
            {i < 3 && <ArrowRight className="w-4 h-4 text-slate-400" />}
          </div>
        ))}
      </div>

      {/* Linear address (intermediate) */}
      {step >= 2 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 text-center">
          <span className="text-sm text-blue-700 dark:text-blue-300">
            Linear Address = base(0x{seg.base.toString(16)}) + page({pageIdx}) * pageSize(0x100) + offset({offset}) = <strong className="font-mono">0x{linearAddr.toString(16).toUpperCase().padStart(4, "0")}</strong>
          </span>
        </motion.div>
      )}

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className={`p-5 rounded-xl border-2 ${result.pa < 0 ? "bg-red-50 dark:bg-red-900/20 border-red-400" : "bg-green-50 dark:bg-green-900/20 border-green-400"}`}>
            {result.pa < 0 ? (
              <div className="text-lg font-bold text-red-700 dark:text-red-300">Page Fault! Invalid page table entry.</div>
            ) : (
              <div>
                <div className="text-lg font-bold text-green-700 dark:text-green-300 mb-1">
                  Physical Address: 0x{result.pa.toString(16).toUpperCase().padStart(4, "0")}
                </div>
                <div className="text-sm text-green-600 dark:text-green-400">
                  Two-step: Logical ({seg.name} + Page {pageIdx} + {offset}) → Linear (0x{result.linearAddr.toString(16)}) → Physical (0x{result.pa.toString(16)})
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info */}
      <div className="mt-6 p-4 bg-teal-50 dark:bg-teal-900/20 rounded-lg border border-teal-200 dark:border-teal-800">
        <p className="text-sm text-teal-800 dark:text-teal-200">
          <strong>Segment + Page Translation:</strong> Used by x86 in protected mode. The logical address is (segment, offset).
          Step 1: The segment selector indexes the segment table to get the segment base. The linear address = segment base + offset.
          Step 2: The linear address is then translated through the page table to get the physical address.
          This two-level scheme combines the protection benefits of segmentation with the flexibility of paging.
        </p>
      </div>
    </div>
  );
}
