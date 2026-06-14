"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, ChevronRight, ChevronLeft, RotateCcw, ArrowRight } from "lucide-react";

interface InstrLine {
  line: number;
  code: string;
  comment: string;
  effect?: string;
  writesTo?: string[];
  readsFrom?: string[];
}

const SWTCH_CODE: InstrLine[] = [
  { line: 1, code: ".globl swtch", comment: "Context switch function", effect: "Function entry point" },
  { line: 2, code: "swtch:", comment: "swtch(struct context *old, struct context *new)", effect: "Label: a0=old context, a1=new context" },
  { line: 3, code: "sd ra, 0(a0)", comment: "Save old->ra", effect: "Store CPU ra to old->ra", writesTo: ["old.ra"], readsFrom: ["ra"] },
  { line: 4, code: "sd sp, 8(a0)", comment: "Save old->sp", effect: "Store CPU sp to old->sp", writesTo: ["old.sp"], readsFrom: ["sp"] },
  { line: 5, code: "sd s0, 16(a0)", comment: "Save old->s0", effect: "Store CPU s0 to old->s0", writesTo: ["old.s0"], readsFrom: ["s0"] },
  { line: 6, code: "sd s1, 24(a0)", comment: "Save old->s1", effect: "Store CPU s1 to old->s1", writesTo: ["old.s1"], readsFrom: ["s1"] },
  { line: 7, code: "sd s2, 32(a0)", comment: "Save old->s2", effect: "Store CPU s2 to old->s2", writesTo: ["old.s2"], readsFrom: ["s2"] },
  { line: 8, code: "sd s3, 40(a0)", comment: "Save old->s3", effect: "Store CPU s3 to old->s3", writesTo: ["old.s3"], readsFrom: ["s3"] },
  { line: 9, code: "sd s4, 48(a0)", comment: "Save old->s4", effect: "Store CPU s4 to old->s4", writesTo: ["old.s4"], readsFrom: ["s4"] },
  { line: 10, code: "sd s5, 56(a0)", comment: "Save old->s5", effect: "Store CPU s5 to old->s5", writesTo: ["old.s5"], readsFrom: ["s5"] },
  { line: 11, code: "sd s6, 64(a0)", comment: "Save old->s6", effect: "Store CPU s6 to old->s6", writesTo: ["old.s6"], readsFrom: ["s6"] },
  { line: 12, code: "sd s7, 72(a0)", comment: "Save old->s7", effect: "Store CPU s7 to old->s7", writesTo: ["old.s7"], readsFrom: ["s7"] },
  { line: 13, code: "sd s8, 80(a0)", comment: "Save old->s8", effect: "Store CPU s8 to old->s8", writesTo: ["old.s8"], readsFrom: ["s8"] },
  { line: 14, code: "sd s9, 88(a0)", comment: "Save old->s9", effect: "Store CPU s9 to old->s9", writesTo: ["old.s9"], readsFrom: ["s9"] },
  { line: 15, code: "sd s10, 96(a0)", comment: "Save old->s10", effect: "Store CPU s10 to old->s10", writesTo: ["old.s10"], readsFrom: ["s10"] },
  { line: 16, code: "sd s11, 104(a0)", comment: "Save old->s11", effect: "Store CPU s11 to old->s11", writesTo: ["old.s11"], readsFrom: ["s11"] },
  { line: 17, code: "", comment: "--- switch to new context ---", effect: "Now load new context's registers into CPU" },
  { line: 18, code: "ld ra, 0(a1)", comment: "Load new->ra", effect: "Load new->ra into CPU ra", writesTo: ["ra"], readsFrom: ["new.ra"] },
  { line: 19, code: "ld sp, 8(a1)", comment: "Load new->sp", effect: "Load new->sp into CPU sp", writesTo: ["sp"], readsFrom: ["new.sp"] },
  { line: 20, code: "ld s0, 16(a1)", comment: "Load new->s0", effect: "Load new->s0 into CPU s0", writesTo: ["s0"], readsFrom: ["new.s0"] },
  { line: 21, code: "ld s1, 24(a1)", comment: "Load new->s1", effect: "Load new->s1 into CPU s1", writesTo: ["s1"], readsFrom: ["new.s1"] },
  { line: 22, code: "ld s2, 32(a1)", comment: "Load new->s2", effect: "Load new->s2 into CPU s2", writesTo: ["s2"], readsFrom: ["new.s2"] },
  { line: 23, code: "ld s3, 40(a1)", comment: "Load new->s3", effect: "Load new->s3 into CPU s3", writesTo: ["s3"], readsFrom: ["new.s3"] },
  { line: 24, code: "ld s4, 48(a1)", comment: "Load new->s4", effect: "Load new->s4 into CPU s4", writesTo: ["s4"], readsFrom: ["new.s4"] },
  { line: 25, code: "ld s5, 56(a1)", comment: "Load new->s5", effect: "Load new->s5 into CPU s5", writesTo: ["s5"], readsFrom: ["new.s5"] },
  { line: 26, code: "ld s6, 64(a1)", comment: "Load new->s6", effect: "Load new->s6 into CPU s6", writesTo: ["s6"], readsFrom: ["new.s6"] },
  { line: 27, code: "ld s7, 72(a1)", comment: "Load new->s7", effect: "Load new->s7 into CPU s7", writesTo: ["s7"], readsFrom: ["new.s7"] },
  { line: 28, code: "ld s8, 80(a1)", comment: "Load new->s8", effect: "Load new->s8 into CPU s8", writesTo: ["s8"], readsFrom: ["new.s8"] },
  { line: 29, code: "ld s9, 88(a1)", comment: "Load new->s9", effect: "Load new->s9 into CPU s9", writesTo: ["s9"], readsFrom: ["new.s9"] },
  { line: 30, code: "ld s10, 96(a1)", comment: "Load new->s10", effect: "Load new->s10 into CPU s10", writesTo: ["s10"], readsFrom: ["new.s10"] },
  { line: 31, code: "ld s11, 104(a1)", comment: "Load new->s11", effect: "Load new->s11 into CPU s11", writesTo: ["s11"], readsFrom: ["new.s11"] },
  { line: 32, code: "ret", comment: "Return to new kernel context", effect: "Jump to new->ra (returns to caller of swtch for new process)" },
];

const ALL_REGS = ["ra", "sp", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"];

const INIT_OLD: Record<string, string> = {
  ra: "0x80002340", sp: "0x80010000", s0: "0x00000001", s1: "0x00000002",
  s2: "0x000000aa", s3: "0x000000bb", s4: "0x000000cc", s5: "0x000000dd",
  s6: "0x000000ee", s7: "0x000000ff", s8: "0x11111111", s9: "0x22222222",
  s10: "0x33333333", s11: "0x44444444",
};

const INIT_NEW: Record<string, string> = {
  ra: "0x80004560", sp: "0x80014000", s0: "0x00000009", s1: "0x0000000a",
  s2: "0x00000011", s3: "0x00000022", s4: "0x00000033", s5: "0x00000044",
  s6: "0x00000055", s7: "0x00000066", s8: "0x55555555", s9: "0x66666666",
  s10: "0x77777777", s11: "0x88888888",
};

export default function Xv6SwtchCodeViewer() {
  const [pc, setPc] = useState(0);
  const [cpuRegs, setCpuRegs] = useState<Record<string, string>>({ ...INIT_OLD });
  const [oldCtx, setOldCtx] = useState<Record<string, string>>({});
  const [newCtx] = useState<Record<string, string>>({ ...INIT_NEW });

  const step = useCallback(() => {
    if (pc >= SWTCH_CODE.length - 1) return;
    const nextPc = pc + 1;
    const instr = SWTCH_CODE[nextPc];

    if (instr.writesTo && instr.readsFrom) {
      const dest = instr.writesTo[0];
      const src = instr.readsFrom[0];

      if (dest.startsWith("old.")) {
        const reg = dest.slice(4);
        setOldCtx((prev) => ({ ...prev, [reg]: cpuRegs[reg] || "---" }));
      } else if (src.startsWith("new.")) {
        const reg = src.slice(4);
        const val = INIT_NEW[reg] || "---";
        setCpuRegs((prev) => ({ ...prev, [reg]: val }));
      } else {
        setCpuRegs((prev) => ({ ...prev, [dest]: INIT_NEW[dest] || "0x00000000" }));
      }
    }

    setPc(nextPc);
  }, [pc, cpuRegs]);

  const reset = useCallback(() => {
    setPc(0);
    setCpuRegs({ ...INIT_OLD });
    setOldCtx({});
  }, []);

  const currentLine = SWTCH_CODE[pc];
  const isSaving = pc >= 2 && pc <= 15;
  const isLoading = pc >= 17 && pc <= 30;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center flex items-center justify-center gap-2">
        <Terminal className="w-7 h-7 text-amber-600 dark:text-amber-400" />
        xv6 swtch.S Code Viewer
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Assembly code */}
        <div className="lg:col-span-2 bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
          <div className="flex items-center gap-2 px-4 py-2 bg-gray-800 border-b border-gray-700">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="ml-2 text-xs text-gray-400 font-mono">swtch.S</span>
          </div>
          <div className="p-3 max-h-[520px] overflow-y-auto">
            {SWTCH_CODE.map((instr, idx) => {
              const isCurrent = idx === pc;
              const isExecuted = idx < pc;
              const isEmpty = instr.code === "";
              return (
                <motion.div
                  key={idx}
                  animate={{
                    backgroundColor: isCurrent
                      ? "rgba(250, 204, 21, 0.2)"
                      : "transparent",
                  }}
                  className={`flex items-start gap-2 px-2 py-0.5 rounded font-mono text-xs leading-relaxed ${
                    isCurrent ? "ring-1 ring-yellow-400" : ""
                  }`}
                >
                  <span className="w-6 text-right text-gray-600 select-none shrink-0">
                    {instr.line}
                  </span>
                  {isCurrent && (
                    <ChevronRight className="w-3 h-3 text-yellow-400 shrink-0 mt-0.5" />
                  )}
                  {!isCurrent && <span className="w-3 shrink-0" />}
                  <span
                    className={`${
                      isEmpty
                        ? "text-gray-500 italic"
                        : isExecuted
                        ? "text-gray-500"
                        : isCurrent
                        ? "text-yellow-200"
                        : "text-green-300"
                    }`}
                  >
                    {instr.code || instr.comment}
                  </span>
                  {!isEmpty && (
                    <span className="text-gray-600 ml-auto shrink-0 hidden sm:inline">
                      # {instr.comment}
                    </span>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Right panel: register state */}
        <div className="space-y-4">
          {/* Effect description */}
          <AnimatePresence mode="wait">
            <motion.div
              key={pc}
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 5 }}
              className={`p-3 rounded-lg text-sm border-l-4 ${
                isSaving
                  ? "bg-blue-50 border-blue-500 dark:bg-blue-900/30 dark:border-blue-400"
                  : isLoading
                  ? "bg-green-50 border-green-500 dark:bg-green-900/30 dark:border-green-400"
                  : "bg-slate-50 border-slate-400 dark:bg-gray-800 dark:border-gray-500"
              }`}
            >
              <p className="font-semibold text-slate-800 dark:text-gray-100 text-xs">
                {currentLine.effect || "Ready"}
              </p>
            </motion.div>
          </AnimatePresence>

          {/* Old context */}
          <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700">
            <h4 className="text-xs font-bold text-blue-700 dark:text-blue-300 mb-2">
              old context (Process A)
            </h4>
            <div className="space-y-0.5">
              {ALL_REGS.map((r) => (
                <div
                  key={r}
                  className="flex justify-between text-xs font-mono px-1 py-0.5 rounded"
                >
                  <span className="text-blue-800 dark:text-blue-300">{r}</span>
                  <span className="text-slate-600 dark:text-gray-400">
                    {oldCtx[r] || "---"}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* CPU registers */}
          <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700">
            <h4 className="text-xs font-bold text-amber-700 dark:text-amber-300 mb-2">
              CPU Registers
            </h4>
            <div className="space-y-0.5">
              {ALL_REGS.map((r) => (
                <motion.div
                  key={r}
                  animate={{
                    backgroundColor:
                      currentLine.writesTo?.includes(r) || currentLine.readsFrom?.includes(r)
                        ? "rgba(250, 204, 21, 0.3)"
                        : "transparent",
                  }}
                  className="flex justify-between text-xs font-mono px-1 py-0.5 rounded"
                >
                  <span className="text-amber-800 dark:text-amber-300">{r}</span>
                  <span className="text-slate-600 dark:text-gray-400">
                    {cpuRegs[r]}
                  </span>
                </motion.div>
              ))}
            </div>
          </div>

          {/* New context */}
          <div className="p-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-700">
            <h4 className="text-xs font-bold text-green-700 dark:text-green-300 mb-2">
              new context (Process B)
            </h4>
            <div className="space-y-0.5">
              {ALL_REGS.map((r) => (
                <div
                  key={r}
                  className="flex justify-between text-xs font-mono px-1 py-0.5 rounded"
                >
                  <span className="text-green-800 dark:text-green-300">{r}</span>
                  <span className="text-slate-600 dark:text-gray-400">
                    {INIT_NEW[r]}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Data flow indicator */}
      <div className="my-3 flex items-center justify-center gap-2 text-xs font-semibold">
        {isSaving && (
          <>
            <span className="text-blue-600 dark:text-blue-400">CPU Registers</span>
            <ArrowRight className="w-4 h-4 text-blue-500" />
            <span className="text-blue-600 dark:text-blue-400">old context</span>
          </>
        )}
        {isLoading && (
          <>
            <span className="text-green-600 dark:text-green-400">new context</span>
            <ArrowRight className="w-4 h-4 text-green-500" />
            <span className="text-green-600 dark:text-green-400">CPU Registers</span>
          </>
        )}
        {!isSaving && !isLoading && (
          <span className="text-slate-500 dark:text-gray-400">
            Step through to see register data flow
          </span>
        )}
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3 mt-2">
        <button
          onClick={reset}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 flex items-center gap-1 text-sm"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
        <button
          onClick={() => setPc((p) => Math.max(0, p - 1))}
          disabled={pc === 0}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 flex items-center gap-1 text-sm"
        >
          <ChevronLeft className="w-4 h-4" />
          Prev
        </button>
        <button
          onClick={step}
          disabled={pc >= SWTCH_CODE.length - 1}
          className="px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 disabled:opacity-50 flex items-center gap-1 text-sm"
        >
          Step
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      <p className="text-center text-xs text-slate-500 dark:text-gray-400 mt-3">
        Step {pc + 1} / {SWTCH_CODE.length} &mdash; {isSaving ? "Saving old context" : isLoading ? "Loading new context" : pc === 0 ? "Start" : "Done"}
      </p>
    </div>
  );
}
