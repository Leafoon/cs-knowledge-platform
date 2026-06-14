"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, ChevronRight, ChevronLeft, RotateCcw, Play } from "lucide-react";

interface PTLevel {
  name: string;
  fullName: string;
  indexBits: string;
  bitRange: string;
  color: string;
  entries: { idx: number; value: string; present: boolean }[];
}

interface PTEFields {
  present: boolean;
  rw: boolean;
  user: boolean;
  pwt: boolean;
  pcd: boolean;
  accessed: boolean;
  dirty: boolean;
  ps: boolean;
  pfn: string;
}

function makeFakePTE(idx: number, level: number): PTEFields {
  return {
    present: idx < 400,
    rw: level === 0,
    user: true,
    pwt: false,
    pcd: false,
    accessed: idx < 200,
    dirty: level === 0 && idx < 100,
    ps: level === 1 && idx > 300,
    pfn: `0x${(0x1000 + idx * 0x10 + level * 0x100000).toString(16).toUpperCase().slice(0, 6)}`,
  };
}

export default function X86PageTableWalker() {
  const [vaInput, setVaInput] = useState("0x00007FFF12345678");
  const [currentStep, setCurrentStep] = useState(-1); // -1=idle, 0-3=levels, 4=done
  const [isAnimating, setIsAnimating] = useState(false);

  const parseVA = (hex: string) => {
    const val = BigInt(hex);
    return {
      pml4: Number((val >> 39n) & 0x1FFn),
      pdpt: Number((val >> 30n) & 0x1FFn),
      pd: Number((val >> 21n) & 0x1FFn),
      pt: Number((val >> 12n) & 0x1FFn),
      offset: Number(val & 0xFFFn),
      bits47_39: ((val >> 39n) & 0x1FFn).toString(2).padStart(9, "0"),
      bits38_30: ((val >> 30n) & 0x1FFn).toString(2).padStart(9, "0"),
      bits29_21: ((val >> 21n) & 0x1FFn).toString(2).padStart(9, "0"),
      bits20_12: ((val >> 12n) & 0x1FFn).toString(2).padStart(9, "0"),
      bits11_0: (val & 0xFFFn).toString(2).padStart(12, "0"),
    };
  };

  let parsed = { pml4: 0, pdpt: 0, pd: 0, pt: 0, offset: 0, bits47_39: "000000000", bits38_30: "000000000", bits29_21: "000000000", bits20_12: "000000000", bits11_0: "000000000000" };
  try { parsed = parseVA(vaInput); } catch {}

  const levels: PTLevel[] = [
    { name: "PML4", fullName: "Page Map Level 4", indexBits: "47:39", bitRange: "9 bits", color: "purple",
      entries: [
        { idx: parsed.pml4, value: makeFakePTE(parsed.pml4, 3).pfn, present: true },
        { idx: (parsed.pml4 + 1) % 512, value: "0x000000", present: false },
        { idx: (parsed.pml4 + 50) % 512, value: makeFakePTE(parsed.pml4 + 50, 3).pfn, present: true },
      ].sort((a, b) => a.idx - b.idx),
    },
    { name: "PDPT", fullName: "Page Directory Pointer Table", indexBits: "38:30", bitRange: "9 bits", color: "blue",
      entries: [
        { idx: parsed.pdpt, value: makeFakePTE(parsed.pdpt, 2).pfn, present: true },
        { idx: (parsed.pdpt + 1) % 512, value: "0x000000", present: false },
        { idx: (parsed.pdpt + 100) % 512, value: makeFakePTE(parsed.pdpt + 100, 2).pfn, present: true },
      ].sort((a, b) => a.idx - b.idx),
    },
    { name: "PD", fullName: "Page Directory", indexBits: "29:21", bitRange: "9 bits", color: "green",
      entries: [
        { idx: parsed.pd, value: makeFakePTE(parsed.pd, 1).pfn, present: true },
        { idx: (parsed.pd + 1) % 512, value: "0x000000", present: false },
        { idx: (parsed.pd + 200) % 512, value: makeFakePTE(parsed.pd + 200, 1).pfn, present: true },
      ].sort((a, b) => a.idx - b.idx),
    },
    { name: "PT", fullName: "Page Table", indexBits: "20:12", bitRange: "9 bits", color: "orange",
      entries: [
        { idx: parsed.pt, value: makeFakePTE(parsed.pt, 0).pfn, present: true },
        { idx: (parsed.pt + 1) % 512, value: "0x000000", present: false },
        { idx: (parsed.pt + 50) % 512, value: makeFakePTE(parsed.pt + 50, 0).pfn, present: true },
      ].sort((a, b) => a.idx - b.idx),
    },
  ];

  const indices = [parsed.pml4, parsed.pdpt, parsed.pd, parsed.pt];

  const finalPFN = "0x0C000";
  const physAddr = `0x0C000${parsed.offset.toString(16).toUpperCase().padStart(3, "0")}`;

  const startWalk = useCallback(async () => {
    setIsAnimating(true);
    setCurrentStep(0);
    for (let i = 1; i <= 4; i++) {
      await new Promise((r) => setTimeout(r, 1200));
      setCurrentStep(i);
    }
    setIsAnimating(false);
  }, []);

  const stepNext = () => {
    if (currentStep < 4) setCurrentStep((s) => s + 1);
  };
  const stepPrev = () => {
    if (currentStep > 0) setCurrentStep((s) => s - 1);
  };
  const handleReset = () => {
    setCurrentStep(-1);
    setIsAnimating(false);
  };

  const colorMap: Record<string, string> = {
    purple: "from-purple-500 to-purple-600",
    blue: "from-blue-500 to-blue-600",
    green: "from-green-500 to-green-600",
    orange: "from-orange-500 to-orange-600",
  };
  const bgColorMap: Record<string, string> = {
    purple: "bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800",
    blue: "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800",
    green: "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800",
    orange: "bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800",
  };
  const textColorMap: Record<string, string> = {
    purple: "text-purple-700 dark:text-purple-300",
    blue: "text-blue-700 dark:text-blue-300",
    green: "text-green-700 dark:text-green-300",
    orange: "text-orange-700 dark:text-orange-300",
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <div className="flex items-center justify-center gap-3 mb-6">
        <Cpu className="w-7 h-7 text-orange-600 dark:text-orange-400" />
        <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100">
          x86-64 Four-Level Page Table Walk
        </h2>
      </div>

      {/* CR3 Register */}
      <div className="mb-4 p-3 bg-slate-800 dark:bg-slate-900 rounded-lg flex items-center gap-4">
        <span className="text-amber-400 font-mono text-sm font-bold">CR3</span>
        <span className="text-slate-300 font-mono text-sm">= 0x00000000_00500000 (PML4 base)</span>
      </div>

      {/* Input */}
      <div className="flex flex-wrap gap-4 mb-6 items-end">
        <div className="flex-1 min-w-64">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">48-bit Virtual Address</label>
          <input type="text" value={vaInput} onChange={(e) => setVaInput(e.target.value)}
            disabled={isAnimating}
            className="w-full px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-gray-800 text-slate-800 dark:text-slate-200 font-mono disabled:opacity-50" />
        </div>
        <button onClick={startWalk} disabled={isAnimating}
          className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition flex items-center gap-2 text-sm font-medium disabled:opacity-50">
          <Play className="w-4 h-4" /> Auto Walk
        </button>
        <button onClick={stepPrev} disabled={currentStep <= 0}
          className="px-3 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition disabled:opacity-30">
          <ChevronLeft className="w-4 h-4" />
        </button>
        <button onClick={stepNext} disabled={currentStep >= 4}
          className="px-3 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition disabled:opacity-30">
          <ChevronRight className="w-4 h-4" />
        </button>
        <button onClick={handleReset}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition flex items-center gap-2 text-sm font-medium">
          <RotateCcw className="w-4 h-4" /> Reset
        </button>
      </div>

      {/* VA bit breakdown */}
      <div className="mb-6 p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-slate-200 dark:border-gray-700">
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Virtual Address Bits</h4>
        <div className="grid grid-cols-5 gap-2 font-mono text-xs text-center">
          {[["PML4", "47:39", parsed.bits47_39, parsed.pml4, "purple"],
            ["PDPT", "38:30", parsed.bits38_30, parsed.pdpt, "blue"],
            ["PD", "29:21", parsed.bits29_21, parsed.pd, "green"],
            ["PT", "20:12", parsed.bits20_12, parsed.pt, "orange"],
            ["Offset", "11:0", parsed.bits11_0, parsed.offset, "gray"]].map(([name, bits, binVal, decVal, color], i) => (
            <div key={i} className={`p-2 rounded-lg ${currentStep >= i && currentStep < 5 ? `ring-2 ring-${color === "gray" ? "slate" : color}-400` : ""} ${color === "gray" ? "bg-slate-100 dark:bg-gray-700" : `bg-${color === "purple" ? "purple" : color === "blue" ? "blue" : color === "green" ? "green" : "orange"}-50 dark:bg-${color === "purple" ? "purple" : color === "blue" ? "blue" : color === "green" ? "green" : "orange"}-900/20`}`}>
              <div className="text-slate-500 dark:text-slate-400 mb-1">{name} ({bits as string})</div>
              <div className="font-bold text-slate-800 dark:text-slate-100">{binVal as string}</div>
              <div className="text-slate-500 dark:text-slate-400">= {decVal as number}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Four levels walkthrough */}
      <div className="space-y-3 mb-6">
        {levels.map((level, i) => {
          const isActive = currentStep === i;
          const isDone = currentStep > i;
          const entry = level.entries.find((e) => e.idx === indices[i]);
          const pte = makeFakePTE(indices[i], 3 - i);

          return (
            <motion.div
              key={level.name}
              className={`p-4 rounded-xl border transition-all ${
                isActive
                  ? `bg-gradient-to-r ${colorMap[level.color]} text-white shadow-lg border-transparent`
                  : isDone
                    ? `${bgColorMap[level.color]} border`
                    : "bg-white dark:bg-gray-800 border-slate-200 dark:border-gray-700 opacity-50"
              }`}
              animate={{ scale: isActive ? 1.01 : 1, opacity: isDone || isActive ? 1 : 0.5 }}
              transition={{ duration: 0.3 }}
            >
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h4 className={`font-bold ${isActive ? "text-white" : isDone ? textColorMap[level.color] : "text-slate-500"}`}>
                    {level.name} ({level.fullName})
                  </h4>
                  <span className={`text-xs ${isActive ? "text-white/80" : "text-slate-400"}`}>
                    Index bits: {level.indexBits} ({level.bitRange}) | Index: {indices[i]}
                  </span>
                </div>
                {isDone && <span className="text-xs bg-white/20 px-2 py-1 rounded">Done</span>}
                {isActive && <span className="text-xs bg-white/30 px-2 py-1 rounded animate-pulse">Walking...</span>}
              </div>

              {/* Entries display */}
              <div className="grid grid-cols-3 gap-2 mt-2">
                {level.entries.map((e, j) => (
                  <div key={j}
                    className={`p-2 rounded text-xs font-mono ${
                      e.idx === indices[i]
                        ? isActive ? "bg-white/30 ring-2 ring-white" : "bg-white dark:bg-gray-900 ring-2 ring-orange-400"
                        : e.present
                          ? isActive ? "bg-white/10" : "bg-green-50 dark:bg-green-900/20"
                          : isActive ? "bg-white/10" : "bg-red-50 dark:bg-red-900/20"
                    }`}>
                    <div className={isActive ? "text-white" : "text-slate-500"}>[{e.idx}]</div>
                    <div className={`font-bold ${isActive ? "text-white" : "text-slate-800 dark:text-slate-200"}`}>
                      {e.value}
                    </div>
                    <div className={isActive ? "text-white/70" : e.present ? "text-green-600 dark:text-green-400" : "text-red-500"}>
                      {e.present ? "Present" : "Not present"}
                    </div>
                  </div>
                ))}
              </div>

              {/* PTE bit fields when active */}
              {isActive && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}
                  className="mt-3 grid grid-cols-4 gap-1 text-xs">
                  {[
                    ["P", pte.present], ["R/W", pte.rw], ["U/S", pte.user], ["PWT", pte.pwt],
                    ["PCD", pte.pcd], ["A", pte.accessed], ["D", pte.dirty], ["PS", pte.ps],
                  ].map(([label, val], k) => (
                    <div key={k} className="bg-white/15 rounded px-2 py-1 text-center">
                      <span className="text-white/60">{label as string}: </span>
                      <span className="font-bold text-white">{val ? "1" : "0"}</span>
                    </div>
                  ))}
                  <div className="col-span-4 bg-white/15 rounded px-2 py-1 text-center">
                    <span className="text-white/60">PFN: </span>
                    <span className="font-bold text-white">{pte.pfn}</span>
                  </div>
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Final physical address */}
      <AnimatePresence>
        {currentStep >= 4 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="p-5 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl shadow-lg text-white"
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center font-bold">PA</div>
              <h4 className="text-lg font-bold">Translation Complete</h4>
            </div>
            <div className="grid grid-cols-3 gap-4 mt-3">
              <div className="bg-white/15 rounded-lg p-3">
                <div className="text-xs text-white/70">PFN</div>
                <div className="font-mono font-bold text-lg">{finalPFN}</div>
              </div>
              <div className="bg-white/15 rounded-lg p-3">
                <div className="text-xs text-white/70">Offset</div>
                <div className="font-mono font-bold text-lg">0x{parsed.offset.toString(16).toUpperCase().padStart(3, "0")}</div>
              </div>
              <div className="bg-white/15 rounded-lg p-3">
                <div className="text-xs text-white/70">Physical Address</div>
                <div className="font-mono font-bold text-lg">{physAddr}</div>
              </div>
            </div>
            <div className="mt-3 text-sm text-white/80">
              = PFN ({finalPFN}) &lt;&lt; 12 | offset (0x{parsed.offset.toString(16).toUpperCase().padStart(3, "0")})
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info */}
      <div className="mt-6 p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
        <p className="text-sm text-orange-800 dark:text-orange-200">
          <strong>x86-64 Paging:</strong> Uses 4-level page tables (PML4 → PDPT → PD → PT).
          Each level indexes 9 bits (512 entries per table), with a 12-bit page offset for 4KB pages.
          The CPU starts from CR3 (PML4 base), follows each pointer through the hierarchy, and
          combines the final PFN with the offset to produce the physical address.
          Each memory access in the walk costs a potential cache miss, making TLB hits critical for performance.
        </p>
      </div>
    </div>
  );
}
