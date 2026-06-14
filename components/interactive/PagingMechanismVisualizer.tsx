"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Search, CheckCircle2, XCircle, RotateCcw, ChevronRight } from "lucide-react";

interface PageTableEntry {
  vpn: number;
  pfn: number;
  valid: boolean;
  dirty: boolean;
  referenced: boolean;
}

interface TLBEntry {
  vpn: number;
  pfn: number;
}

const initialPTEs: PageTableEntry[] = [
  { vpn: 0x0, pfn: 0x0A, valid: true, dirty: false, referenced: true },
  { vpn: 0x1, pfn: 0x03, valid: true, dirty: true, referenced: false },
  { vpn: 0x2, pfn: 0x00, valid: false, dirty: false, referenced: false },
  { vpn: 0x3, pfn: 0x1F, valid: true, dirty: false, referenced: true },
  { vpn: 0x4, pfn: 0x0C, valid: true, dirty: true, referenced: true },
  { vpn: 0x5, pfn: 0x00, valid: false, dirty: false, referenced: false },
  { vpn: 0x6, pfn: 0x07, valid: true, dirty: false, referenced: false },
  { vpn: 0x7, pfn: 0x1B, valid: true, dirty: false, referenced: true },
];

const initialTLB: TLBEntry[] = [
  { vpn: 0x0, pfn: 0x0A },
  { vpn: 0x4, pfn: 0x0C },
];

export default function PagingMechanismVisualizer() {
  const [vaInput, setVaInput] = useState("0x04");
  const [ptes, setPTEs] = useState<PageTableEntry[]>(initialPTEs);
  const [tlb, setTLB] = useState<TLBEntry[]>(initialTLB);
  const [step, setStep] = useState(0); // 0=idle, 1=split, 2=tlb check, 3=pt lookup, 4=done
  const [result, setResult] = useState<{ pa: string; hit: boolean } | null>(null);
  const [highlightVPN, setHighlightVPN] = useState<number | null>(null);
  const [showEditor, setShowEditor] = useState(false);
  const [newVPN, setNewVPN] = useState("");
  const [newPFN, setNewPFN] = useState("");
  const [newValid, setNewValid] = useState(true);

  const PAGE_SIZE = 16; // 4-bit offset for demonstration
  const OFFSET_BITS = 4;

  const handleTranslate = useCallback(() => {
    const va = parseInt(vaInput, 16);
    if (isNaN(va) || va < 0 || va > 0xFF) {
      alert("Enter a valid hex address (0x00 - 0xFF)");
      return;
    }

    const vpn = va >> OFFSET_BITS;
    const offset = va & (PAGE_SIZE - 1);
    setHighlightVPN(vpn);
    setStep(1);
    setResult(null);

    // Step 1: show split
    setTimeout(() => {
      // Step 2: TLB check
      setStep(2);
      const tlbHit = tlb.find((e) => e.vpn === vpn);

      setTimeout(() => {
        if (tlbHit) {
          const pa = (tlbHit.pfn << OFFSET_BITS) | offset;
          setResult({ pa: `0x${pa.toString(16).toUpperCase().padStart(4, "0")}`, hit: true });
          setStep(4);
        } else {
          // Step 3: Page table lookup
          setStep(3);
          const pte = ptes.find((e) => e.vpn === vpn);

          setTimeout(() => {
            if (pte && pte.valid) {
              const pa = (pte.pfn << OFFSET_BITS) | offset;
              setResult({ pa: `0x${pa.toString(16).toUpperCase().padStart(4, "0")}`, hit: false });
              // Add to TLB
              setTLB((prev) => [...prev.filter((e) => e.vpn !== vpn), { vpn, pfn: pte.pfn }]);
            } else {
              setResult({ pa: "PAGE FAULT", hit: false });
            }
            setStep(4);
          }, 800);
        }
      }, 800);
    }, 800);
  }, [vaInput, tlb, ptes]);

  const handleReset = () => {
    setStep(0);
    setResult(null);
    setHighlightVPN(null);
    setPTEs(initialPTEs);
    setTLB(initialTLB);
  };

  const handleAddEntry = () => {
    const vpn = parseInt(newVPN, 16);
    const pfn = parseInt(newPFN, 16);
    if (isNaN(vpn) || isNaN(pfn)) return;
    setPTEs((prev) => {
      const existing = prev.findIndex((e) => e.vpn === vpn);
      if (existing >= 0) {
        const updated = [...prev];
        updated[existing] = { ...updated[existing], pfn, valid: newValid };
        return updated;
      }
      return [...prev, { vpn, pfn, valid: newValid, dirty: false, referenced: false }];
    });
    setNewVPN("");
    setNewPFN("");
  };

  const va = parseInt(vaInput, 16);
  const vpn = isNaN(va) ? 0 : va >> OFFSET_BITS;
  const offset = isNaN(va) ? 0 : va & (PAGE_SIZE - 1);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <div className="flex items-center justify-center gap-3 mb-6">
        <Layers className="w-7 h-7 text-violet-600 dark:text-violet-400" />
        <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100">
          Paging Mechanism Visualizer
        </h2>
      </div>

      {/* Input */}
      <div className="flex flex-wrap gap-4 mb-6 items-end">
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            Virtual Address (hex, 8-bit VA, {OFFSET_BITS}-bit offset)
          </label>
          <input type="text" value={vaInput} onChange={(e) => setVaInput(e.target.value)}
            placeholder="0x04" disabled={step > 0 && step < 4}
            className="px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-gray-800 text-slate-800 dark:text-slate-200 font-mono w-40 disabled:opacity-50" />
        </div>
        <button onClick={handleTranslate} disabled={step > 0 && step < 4}
          className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition flex items-center gap-2 text-sm font-medium disabled:opacity-50">
          <Search className="w-4 h-4" /> Translate
        </button>
        <button onClick={handleReset}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition flex items-center gap-2 text-sm font-medium">
          <RotateCcw className="w-4 h-4" /> Reset
        </button>
        <button onClick={() => setShowEditor(!showEditor)}
          className="px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition text-sm font-medium">
          {showEditor ? "Hide Editor" : "Edit Page Table"}
        </button>
      </div>

      {/* Address breakdown */}
      <motion.div
        className="mb-6 p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-slate-200 dark:border-gray-700"
        animate={{ borderColor: step >= 1 ? "#7C3AED" : undefined }}
      >
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Virtual Address Breakdown</h4>
        <div className="flex items-center gap-2 font-mono text-sm">
          <div className={`px-4 py-2 rounded-lg text-center ${step >= 1 ? "bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300 ring-2 ring-violet-400" : "bg-slate-100 dark:bg-gray-700 text-slate-500"}`}>
            <div className="text-xs mb-1">VPN ({OFFSET_BITS} bits)</div>
            <div className="font-bold">0x{vpn.toString(16).toUpperCase().padStart(2, "0")}</div>
          </div>
          <span className="text-slate-400">|</span>
          <div className={`px-4 py-2 rounded-lg text-center ${step >= 1 ? "bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 ring-2 ring-amber-400" : "bg-slate-100 dark:bg-gray-700 text-slate-500"}`}>
            <div className="text-xs mb-1">Offset</div>
            <div className="font-bold">0x{offset.toString(16).toUpperCase().padStart(1, "0")}</div>
          </div>
        </div>
      </motion.div>

      {/* TLB and Page Table side by side */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* TLB */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-gray-700">
          <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4 text-yellow-500" /> TLB ({tlb.length} entries)
          </h4>
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-100 dark:bg-gray-700">
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">VPN</th>
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">PFN</th>
              </tr>
            </thead>
            <tbody>
              {tlb.map((entry, i) => (
                <motion.tr key={i}
                  className={`${step === 2 && entry.vpn === highlightVPN ? "bg-yellow-100 dark:bg-yellow-900/30" : "hover:bg-slate-50 dark:hover:bg-gray-700"}`}
                  animate={step === 2 && entry.vpn === highlightVPN ? { scale: [1, 1.05, 1] } : {}}
                  transition={{ duration: 0.5 }}
                >
                  <td className="px-2 py-1.5 font-mono text-slate-700 dark:text-slate-200">0x{entry.vpn.toString(16).toUpperCase().padStart(2, "0")}</td>
                  <td className="px-2 py-1.5 font-mono text-slate-700 dark:text-slate-200">0x{entry.pfn.toString(16).toUpperCase().padStart(2, "0")}</td>
                </motion.tr>
              ))}
              {tlb.length === 0 && (
                <tr><td colSpan={2} className="px-2 py-4 text-center text-slate-400">Empty</td></tr>
              )}
            </tbody>
          </table>
          {step === 2 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className={`mt-2 text-xs font-medium ${tlb.find((e) => e.vpn === highlightVPN) ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
              {tlb.find((e) => e.vpn === highlightVPN) ? "TLB HIT" : "TLB MISS - consulting page table..."}
            </motion.div>
          )}
        </div>

        {/* Page Table */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-gray-700">
          <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Page Table</h4>
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-100 dark:bg-gray-700">
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">VPN</th>
                <th className="px-2 py-1.5 text-left text-slate-600 dark:text-slate-300">PFN</th>
                <th className="px-2 py-1.5 text-center text-slate-600 dark:text-slate-300">V</th>
                <th className="px-2 py-1.5 text-center text-slate-600 dark:text-slate-300">D</th>
                <th className="px-2 py-1.5 text-center text-slate-600 dark:text-slate-300">R</th>
              </tr>
            </thead>
            <tbody>
              {ptes.sort((a, b) => a.vpn - b.vpn).map((pte, i) => (
                <motion.tr key={i}
                  className={`${step >= 3 && pte.vpn === highlightVPN && pte.valid ? "bg-violet-100 dark:bg-violet-900/30 ring-2 ring-violet-400" : step >= 3 && pte.vpn === highlightVPN && !pte.valid ? "bg-red-100 dark:bg-red-900/30 ring-2 ring-red-400" : "hover:bg-slate-50 dark:hover:bg-gray-700"}`}
                  animate={step >= 3 && pte.vpn === highlightVPN ? { scale: [1, 1.03, 1] } : {}}
                  transition={{ duration: 0.5 }}
                >
                  <td className="px-2 py-1.5 font-mono text-slate-700 dark:text-slate-200">0x{pte.vpn.toString(16).toUpperCase().padStart(2, "0")}</td>
                  <td className="px-2 py-1.5 font-mono text-slate-700 dark:text-slate-200">{pte.valid ? `0x${pte.pfn.toString(16).toUpperCase().padStart(2, "0")}` : "---"}</td>
                  <td className="px-2 py-1.5 text-center">{pte.valid ? <CheckCircle2 className="w-3.5 h-3.5 text-green-500 mx-auto" /> : <XCircle className="w-3.5 h-3.5 text-red-500 mx-auto" />}</td>
                  <td className="px-2 py-1.5 text-center">{pte.dirty ? "D" : "-"}</td>
                  <td className="px-2 py-1.5 text-center">{pte.referenced ? "R" : "-"}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Translation steps indicator */}
      <div className="flex items-center justify-center gap-2 mb-6">
        {["Split VA", "TLB Check", "Page Table", "Physical Addr"].map((label, i) => (
          <div key={i} className="flex items-center gap-2">
            <motion.div
              className={`px-3 py-1.5 rounded-full text-xs font-medium ${step > i ? "bg-violet-600 text-white" : step === i + 1 ? "bg-violet-200 dark:bg-violet-800 text-violet-700 dark:text-violet-300 ring-2 ring-violet-400" : "bg-slate-200 dark:bg-gray-700 text-slate-500 dark:text-slate-400"}`}
              animate={step === i + 1 ? { scale: [1, 1.1, 1] } : {}}
              transition={{ duration: 0.5, repeat: step === i + 1 ? Infinity : 0, repeatDelay: 1 }}
            >
              {label}
            </motion.div>
            {i < 3 && <ChevronRight className="w-4 h-4 text-slate-400" />}
          </div>
        ))}
      </div>

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className={`p-5 rounded-xl border-2 ${result.pa === "PAGE FAULT" ? "bg-red-50 dark:bg-red-900/20 border-red-400" : result.hit ? "bg-yellow-50 dark:bg-yellow-900/20 border-yellow-400" : "bg-green-50 dark:bg-green-900/20 border-green-400"}`}
          >
            <div className="flex items-center gap-3">
              {result.pa === "PAGE FAULT" ? <XCircle className="w-6 h-6 text-red-500" /> :
                result.hit ? <Zap className="w-6 h-6 text-yellow-500" /> :
                  <CheckCircle2 className="w-6 h-6 text-green-500" />}
              <div>
                <div className="text-lg font-bold text-slate-800 dark:text-gray-100">
                  {result.pa === "PAGE FAULT" ? "Page Fault!" : `Physical Address: ${result.pa}`}
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400">
                  {result.hit ? "Resolved via TLB (fast path)" : result.pa === "PAGE FAULT" ? "VPN not found in page table - OS must load from disk" : "Resolved via page table walk (slow path), TLB updated"}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Page Table Editor */}
      <AnimatePresence>
        {showEditor && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
            className="mt-6 p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-slate-200 dark:border-gray-700 overflow-hidden">
            <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Page Table Editor</h4>
            <div className="flex flex-wrap gap-3 items-end">
              <div>
                <label className="block text-xs text-slate-500 mb-1">VPN (hex)</label>
                <input type="text" value={newVPN} onChange={(e) => setNewVPN(e.target.value)} placeholder="0x0"
                  className="px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded bg-white dark:bg-gray-900 text-slate-800 dark:text-slate-200 font-mono text-sm w-20" />
              </div>
              <div>
                <label className="block text-xs text-slate-500 mb-1">PFN (hex)</label>
                <input type="text" value={newPFN} onChange={(e) => setNewPFN(e.target.value)} placeholder="0x0"
                  className="px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded bg-white dark:bg-gray-900 text-slate-800 dark:text-slate-200 font-mono text-sm w-20" />
              </div>
              <label className="flex items-center gap-1 text-sm text-slate-600 dark:text-slate-400">
                <input type="checkbox" checked={newValid} onChange={(e) => setNewValid(e.target.checked)} /> Valid
              </label>
              <button onClick={handleAddEntry}
                className="px-3 py-1.5 bg-violet-600 text-white rounded text-sm font-medium hover:bg-violet-700 transition">
                Add/Update
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info */}
      <div className="mt-6 p-4 bg-violet-50 dark:bg-violet-900/20 rounded-lg border border-violet-200 dark:border-violet-800">
        <p className="text-sm text-violet-800 dark:text-violet-200">
          <strong>Paging Mechanism:</strong> The virtual address is split into a Virtual Page Number (VPN) and an offset.
          The TLB is checked first (1 cycle). On a miss, the page table is walked to find the Physical Frame Number (PFN).
          If the PTE is valid, the physical address = PFN | offset. If invalid, a page fault occurs and the OS must handle it.
        </p>
      </div>
    </div>
  );
}

function Zap(props: React.SVGProps<SVGSVGElement> & { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}
