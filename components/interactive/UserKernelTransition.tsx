"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeftRight, ArrowRight, ArrowLeft, ChevronRight,
  RotateCcw, Cpu, Shield, HardDrive,
} from "lucide-react";

interface Reg { name: string; val: string; group: string }

const U_REGS: Reg[] = [
  { name: "ra", val: "0x00401234", group: "gpr" }, { name: "sp", val: "0x7fff8000", group: "gpr" },
  { name: "gp", val: "0x00500000", group: "gpr" }, { name: "tp", val: "0x00000000", group: "gpr" },
  { name: "a0", val: "0x00000001", group: "arg" }, { name: "a1", val: "0x00000002", group: "arg" },
  { name: "a2", val: "0x00000003", group: "arg" }, { name: "a3", val: "0x00000004", group: "arg" },
  { name: "a4", val: "0x00000005", group: "arg" }, { name: "a5", val: "0x00000006", group: "arg" },
  { name: "a6", val: "0x00000007", group: "arg" }, { name: "a7", val: "0x00000008", group: "arg" },
  { name: "s0", val: "0x00001000", group: "sav" }, { name: "s1", val: "0x00001001", group: "sav" },
  { name: "s2", val: "0x00001002", group: "sav" }, { name: "s3", val: "0x00001003", group: "sav" },
  { name: "s4", val: "0x00001004", group: "sav" }, { name: "s5", val: "0x00001005", group: "sav" },
  { name: "s6", val: "0x00001006", group: "sav" }, { name: "s7", val: "0x00001007", group: "sav" },
  { name: "s8", val: "0x00001008", group: "sav" }, { name: "s9", val: "0x00001009", group: "sav" },
  { name: "s10", val: "0x0000100a", group: "sav" }, { name: "s11", val: "0x0000100b", group: "sav" },
  { name: "t0", val: "0x00002000", group: "tmp" }, { name: "t1", val: "0x00002001", group: "tmp" },
  { name: "t2", val: "0x00002002", group: "tmp" }, { name: "t3", val: "0x00002003", group: "tmp" },
  { name: "t4", val: "0x00002004", group: "tmp" }, { name: "t5", val: "0x00002005", group: "tmp" },
  { name: "t6", val: "0x00002006", group: "tmp" },
  { name: "sepc", val: "0x00401238", group: "csr" }, { name: "sstatus", val: "0x00000020", group: "csr" },
];

const K_META: Reg[] = [
  { name: "kernel_satp", val: "0x80000001", group: "kmeta" },
  { name: "kernel_sp", val: "0x80010000", group: "kmeta" },
  { name: "kernel_trap", val: "0x80002000", group: "kmeta" },
  { name: "kernel_hartid", val: "0x00000000", group: "kmeta" },
];

interface Step { label: string; regs: string[]; desc: string; note?: string }

const ENTRY: Step[] = [
  { label: "Kernel Metadata", regs: ["kernel_satp","kernel_sp","kernel_trap","kernel_hartid"],
    desc: "uservec loads kernel metadata from trapframe into CPU: page table, stack, trap handler, hart ID.",
    note: "Loaded FROM trapframe (read), unlike other regs saved INTO it." },
  { label: "Save ra, sp, gp, tp", regs: ["ra","sp","gp","tp"],
    desc: "First general-purpose registers stored into the trapframe." },
  { label: "Save a0-a7 (sscratch swap)", regs: ["a0","a1","a2","a3","a4","a5","a6","a7"],
    desc: "a0 was swapped with sscratch on trap entry (a0 = trapframe ptr). Original user a0 already saved; a1-a7 saved normally.",
    note: "The a0/sscratch swap lets uservec find the trapframe without clobbering any register." },
  { label: "Save s0-s11", regs: ["s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11"],
    desc: "Callee-saved registers stored — preserves user function local state." },
  { label: "Save t0-t6", regs: ["t0","t1","t2","t3","t4","t5","t6"],
    desc: "Temporaries saved. Though caller-saved by ABI, a trap can interrupt anywhere so all must be preserved." },
  { label: "Save sepc, sstatus", regs: ["sepc","sstatus"],
    desc: "sepc (trap PC, set by hardware) and sstatus stored. Then jumps to usertrap() in C.",
    note: "usertrap() handles the trap: syscall, device interrupt, page fault, etc." },
];

const RETURN: Step[] = [
  { label: "Restore sepc, sstatus", regs: ["sepc","sstatus"],
    desc: "userret loads sepc and sstatus from trapframe. sepc is the sret return address." },
  { label: "Restore t0-t6", regs: ["t0","t1","t2","t3","t4","t5","t6"],
    desc: "Temporaries restored from trapframe to CPU." },
  { label: "Restore s0-s11", regs: ["s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11"],
    desc: "Callee-saved registers restored from trapframe." },
  { label: "Restore a0-a7 (sscratch swap)", regs: ["a0","a1","a2","a3","a4","a5","a6","a7"],
    desc: "a0-a7 restored. a0 swapped back with sscratch to recover original user a0.",
    note: "sscratch is zeroed for the next trap entry." },
  { label: "Restore ra, sp, gp, tp", regs: ["ra","sp","gp","tp"],
    desc: "ra, sp, gp, tp restored. CPU now holds full user register set." },
  { label: "sret to User Mode", regs: [],
    desc: "sret switches to U-mode, sets PC=sepc, restores sstatus. User execution resumes.",
    note: "Trapframe retains a stale copy (reused on next trap entry)." },
];

const G_COLORS: Record<string, {t: string; bg: string}> = {
  kmeta: { t: "text-orange-600 dark:text-orange-400", bg: "bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800" },
  gpr:   { t: "text-blue-600 dark:text-blue-400",     bg: "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800" },
  arg:   { t: "text-emerald-600 dark:text-emerald-400", bg: "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800" },
  sav:   { t: "text-purple-600 dark:text-purple-400",   bg: "bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800" },
  tmp:   { t: "text-pink-600 dark:text-pink-400",       bg: "bg-pink-50 dark:bg-pink-900/20 border-pink-200 dark:border-pink-800" },
  csr:   { t: "text-amber-600 dark:text-amber-400",     bg: "bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800" },
};

const tfSections = [
  { label: "ra, sp, gp, tp", names: ["ra","sp","gp","tp"] },
  { label: "a0-a7", names: ["a0","a1","a2","a3","a4","a5","a6","a7"] },
  { label: "s0-s11", names: ["s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11"] },
  { label: "t0-t6", names: ["t0","t1","t2","t3","t4","t5","t6"] },
  { label: "sepc, sstatus", names: ["sepc","sstatus"] },
];

const groupOf = (name: string) => U_REGS.find(r => r.name === name)?.group ?? "kmeta";

export default function UserKernelTransition() {
  const [mode, setMode] = useState<"entry" | "return">("entry");
  const [si, setSi] = useState(0);
  const [done, setDone] = useState<Set<number>>(new Set());
  const [anim, setAnim] = useState<string[]>([]);

  const steps = mode === "entry" ? ENTRY : RETURN;
  const cur = steps[si];

  const reset = useCallback(() => { setSi(0); setDone(new Set()); setAnim([]); }, []);

  const switchMode = useCallback((m: "entry"|"return") => {
    if (m !== mode) { setMode(m); setSi(0); setDone(new Set()); setAnim([]); }
  }, [mode]);

  const next = useCallback(() => {
    if (si >= steps.length) return;
    setAnim(cur.regs);
    setTimeout(() => {
      setDone(p => { const n = new Set(p); n.add(si); return n; });
      setAnim([]);
      if (si < steps.length - 1) setSi(i => i + 1);
    }, 800);
  }, [si, steps.length, cur]);

  const goTo = useCallback((i: number) => { if (i <= si || done.has(i)) { setSi(i); setAnim([]); } }, [si, done]);

  useEffect(() => { reset(); }, [mode, reset]);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center flex items-center justify-center gap-2">
        <ArrowLeftRight className="w-7 h-7 text-blue-600 dark:text-blue-400" />
        User / Kernel Mode Transition (RISC-V xv6)
      </h2>

      {/* Mode toggle */}
      <div className="flex justify-center gap-2 mb-5">
        {(["entry","return"] as const).map(m => (
          <button key={m} onClick={() => switchMode(m)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
              mode === m ? (m === "entry" ? "bg-red-600 text-white" : "bg-green-600 text-white")
                : "bg-slate-200 text-slate-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-slate-300 dark:hover:bg-gray-600"
            }`}>
            {m === "entry" ? "Trap Entry (uservec)" : "Trap Return (userret)"}
          </button>
        ))}
      </div>

      {/* Step description */}
      <AnimatePresence mode="wait">
        <motion.div key={`${mode}-${si}`} initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} transition={{ duration: 0.25 }}
          className={`mb-5 p-4 rounded-lg border-l-4 ${mode === "entry" ? "bg-red-50 border-red-500 dark:bg-red-900/30 dark:border-red-400" : "bg-green-50 border-green-500 dark:bg-green-900/30 dark:border-green-400"}`}>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-slate-200 dark:bg-gray-700 text-slate-700 dark:text-gray-300">{si+1}/{steps.length}</span>
            <h3 className="font-bold text-slate-800 dark:text-gray-100">{cur.label}</h3>
          </div>
          <p className="text-sm text-slate-600 dark:text-gray-300 mt-1">{cur.desc}</p>
          {cur.note && <p className="text-xs text-amber-700 dark:text-amber-400 mt-2 italic">{cur.note}</p>}
        </motion.div>
      </AnimatePresence>

      {/* Three-column layout */}
      <div className="grid grid-cols-3 gap-4 mb-5 items-start">
        {/* Left: User Mode */}
        <div className="w-full p-3 rounded-lg border-2 border-blue-400 bg-blue-50 dark:bg-blue-900/30 dark:border-blue-500">
          <div className="flex items-center gap-2 mb-3">
            <Cpu className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span className="font-bold text-blue-800 dark:text-blue-200 text-sm">User Mode (U-mode)</span>
          </div>
          <div className="grid grid-cols-2 gap-1">
            {U_REGS.map(r => {
              const active = cur.regs.includes(r.name) || anim.includes(r.name);
              const c = G_COLORS[r.group];
              return (
                <motion.div key={r.name} animate={{ scale: active ? 1.04 : 1, opacity: active ? 1 : 0.6 }} transition={{ duration: 0.25 }}
                  className={`flex justify-between text-xs font-mono px-1.5 py-0.5 rounded border ${active ? `${c.bg}` : "bg-white/50 dark:bg-gray-800/50 border-transparent"}`}>
                  <span className={c.t}>{r.name}</span>
                  <span className="text-slate-600 dark:text-gray-400 text-[10px]">{r.val}</span>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Center: Trapframe */}
        <div className="flex flex-col items-center">
          <div className="w-full p-3 rounded-lg border-2 border-amber-400 bg-amber-50 dark:bg-amber-900/20 dark:border-amber-500">
            <div className="flex items-center justify-center gap-2 mb-3">
              <HardDrive className="w-5 h-5 text-amber-600 dark:text-amber-400" />
              <span className="font-bold text-amber-800 dark:text-amber-200 text-sm">Trapframe (p-&gt;trapframe)</span>
            </div>
            <div className="mb-2">
              <p className="text-[10px] font-semibold text-orange-600 dark:text-orange-400 mb-0.5 uppercase tracking-wide">Kernel Metadata</p>
              {K_META.map(r => {
                const active = cur.regs.includes(r.name) || anim.includes(r.name);
                return (
                  <motion.div key={r.name} animate={{ scale: active ? 1.05 : 1, opacity: active ? 1 : 0.5 }} transition={{ duration: 0.3 }}
                    className="flex justify-between text-xs font-mono px-1.5 py-0.5 rounded bg-orange-50/50 dark:bg-orange-900/10">
                    <span className="text-orange-600 dark:text-orange-400">{r.name}</span>
                    <span className="text-slate-600 dark:text-gray-400 text-[10px]">{r.val}</span>
                  </motion.div>
                );
              })}
            </div>
            {tfSections.map(sec => {
              const g = groupOf(sec.names[0]);
              const c = G_COLORS[g];
              return (
                <div key={sec.label} className="mb-1.5">
                  <p className={`text-[10px] font-semibold mb-0.5 uppercase tracking-wide ${c.t}`}>{sec.label}</p>
                  <div className="grid grid-cols-2 gap-0.5">
                    {sec.names.map(name => {
                      const active = cur.regs.includes(name) || anim.includes(name);
                      const saved = done.has(steps.findIndex(s => s.regs.includes(name)));
                      return (
                        <motion.div key={name} animate={{ scale: active ? 1.05 : 1, opacity: active ? 1 : saved ? 0.9 : 0.4 }} transition={{ duration: 0.3 }}
                          className={`flex justify-between text-[10px] font-mono px-1 py-0.5 rounded ${active ? `${c.bg} font-bold` : saved ? "bg-green-50 dark:bg-green-900/20" : "bg-slate-100/50 dark:bg-gray-800/30"}`}>
                          <span className={active ? c.t : "text-slate-500 dark:text-gray-500"}>{name}</span>
                          {saved && !active && <span className="text-green-500 text-[9px]">saved</span>}
                        </motion.div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
          <AnimatePresence mode="wait">
            <motion.div key={mode} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.8 }}
              className={`mt-2 flex items-center gap-1 text-xs font-bold px-3 py-1 rounded-full ${mode === "entry" ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300" : "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300"}`}>
              {mode === "entry" ? <><ArrowRight className="w-3.5 h-3.5" />uservec: CPU → Trapframe</> : <><ArrowLeft className="w-3.5 h-3.5" />userret: Trapframe → CPU</>}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Right: Kernel Mode */}
        <div className="w-full p-3 rounded-lg border-2 border-red-400 bg-red-50 dark:bg-red-900/30 dark:border-red-500">
          <div className="flex items-center gap-2 mb-3">
            <Shield className="w-5 h-5 text-red-600 dark:text-red-400" />
            <span className="font-bold text-red-800 dark:text-red-200 text-sm">Kernel Mode (S-mode)</span>
          </div>
          <div className="mb-3">
            <p className="text-[10px] font-semibold text-orange-600 dark:text-orange-400 mb-1 uppercase tracking-wide">Kernel Page Table &amp; Stack</p>
            {K_META.map(r => {
              const active = cur.regs.includes(r.name);
              return (
                <motion.div key={r.name} animate={{ scale: active ? 1.04 : 1, opacity: active ? 1 : 0.7 }}
                  className="flex justify-between text-xs font-mono px-1.5 py-0.5 rounded bg-orange-50 dark:bg-orange-900/20">
                  <span className="text-orange-600 dark:text-orange-400">{r.name}</span>
                  <span className="text-slate-600 dark:text-gray-400 text-[10px]">{r.val}</span>
                </motion.div>
              );
            })}
          </div>
          <div className="mb-3 p-2 rounded bg-red-100/50 dark:bg-red-900/20">
            <p className="text-[10px] font-semibold text-red-700 dark:text-red-300 mb-1">Kernel Stack Active</p>
            <div className="text-[10px] font-mono space-y-0.5 text-red-600 dark:text-red-400">
              <p>sp → 0x80010000</p><p>frame[0] = ra</p><p>frame[1] = a0</p><p>...</p>
            </div>
          </div>
          <div className="p-2 rounded bg-slate-100 dark:bg-gray-800/50">
            <p className="text-[10px] font-semibold text-slate-600 dark:text-gray-400 mb-1">Control Flow</p>
            <div className="text-[10px] font-mono space-y-0.5 text-slate-500 dark:text-gray-500">
              {mode === "entry" ? (
                <><p className="text-red-600 dark:text-red-400 font-bold">trap → uservec()</p><p>→ usertrap()</p><p>→ devintr() / syscall()</p><p>→ usertrapret()</p><p>→ userret()</p><p className="text-green-600 dark:text-green-400">→ sret (back to U-mode)</p></>
              ) : (
                <><p>trap → usertrap()</p><p>→ ...</p><p className="text-green-600 dark:text-green-400 font-bold">usertrapret() → userret()</p><p className="text-green-600 dark:text-green-400">→ sret (back to U-mode)</p></>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Step dots */}
      <div className="flex justify-center gap-1.5 mb-4">
        {steps.map((_, i) => (
          <button key={i} onClick={() => goTo(i)}
            className={`w-8 h-8 rounded-full text-xs font-bold transition-all duration-200 ${
              i === si ? (mode === "entry" ? "bg-red-600 text-white scale-110" : "bg-green-600 text-white scale-110")
                : done.has(i) ? "bg-green-200 text-green-800 dark:bg-green-800 dark:text-green-200"
                : "bg-slate-200 text-slate-500 dark:bg-gray-700 dark:text-gray-400 hover:bg-slate-300 dark:hover:bg-gray-600"
            }`}>{i+1}</button>
        ))}
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3">
        <button onClick={reset} className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 flex items-center gap-1 text-sm">
          <RotateCcw className="w-4 h-4" />Reset
        </button>
        <button onClick={next} disabled={si >= steps.length - 1 && done.has(si)}
          className={`px-5 py-2 text-white rounded-lg flex items-center gap-1 text-sm font-semibold disabled:opacity-40 ${mode === "entry" ? "bg-red-600 hover:bg-red-700" : "bg-green-600 hover:bg-green-700"}`}>
          Next Step<ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
