"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// ── 类型 ────────────────────────────────────────────────────────────────────────
type Op = "insert-head" | "insert-tail" | "insert-after" | "delete-head" | "delete-node" | "traverse";

interface LLNode {
  id: number;
  val: number;
  dummy?: boolean;
}

interface Step {
  nodes: LLNode[];
  highlightIds: number[];
  label: string;
  phase: "normal" | "highlight" | "success" | "remove";
  dummyShown: boolean;
}

let uid = 100;

function buildInitialNodes(): LLNode[] {
  return [
    { id: uid++, val: 1 },
    { id: uid++, val: 2 },
    { id: uid++, val: 3 },
    { id: uid++, val: 4 },
  ];
}

function buildSteps(nodes: LLNode[], op: Op, targetVal: number, useDummy: boolean): Step[] {
  const steps: Step[] = [];
  const ns = nodes.map((n) => ({ ...n }));
  const dummy: LLNode = { id: 0, val: -1, dummy: true };

  const snap = (n: LLNode[], hl: number[], label: string, phase: Step["phase"] = "normal") =>
    steps.push({ nodes: n.map((x) => ({ ...x })), highlightIds: hl, label, phase, dummyShown: useDummy });

  if (op === "insert-head") {
    const newNode: LLNode = { id: uid++, val: targetVal };
    if (useDummy) {
      snap([dummy, ...ns], [dummy.id], "哨兵节点：dummy.next 当前指向原头节点", "highlight");
      snap([dummy, newNode, ...ns], [newNode.id], `新节点 ${targetVal} 插入：newNode.next = dummy.next（原链表头）`, "highlight");
      snap([dummy, newNode, ...ns], [dummy.id, newNode.id], `dummy.next = newNode，完成头插`, "success");
    } else {
      snap(ns, [], "准备头插，当前链表状态", "normal");
      snap([newNode, ...ns], [newNode.id], `新节点 ${targetVal} 的 next = 原 head`, "highlight");
      snap([newNode, ...ns], [newNode.id], `head = 新节点，头插完成`, "success");
    }
    return steps;
  }

  if (op === "insert-tail") {
    const newNode: LLNode = { id: uid++, val: targetVal };
    const all = useDummy ? [dummy, ...ns] : ns;
    for (let i = (useDummy ? 1 : 0); i < all.length; i++) {
      snap(all, [all[i].id], `遍历：当前节点 ${all[i].dummy ? "dummy" : all[i].val}，next=${i < all.length - 1 ? all[i + 1].val : "null"}`, "highlight");
    }
    const final = useDummy ? [dummy, ...ns, newNode] : [...ns, newNode];
    snap(final, [newNode.id], `找到尾节点，执行 tail.next = 新节点 ${targetVal}，尾插完成`, "success");
    return steps;
  }

  if (op === "insert-after") {
    // 在第 2 个节点后插入
    const newNode: LLNode = { id: uid++, val: targetVal };
    const insertIdx = Math.min(1, ns.length - 1);
    const all = useDummy ? [dummy, ...ns] : ns;
    const refNode = all[useDummy ? insertIdx + 1 : insertIdx];
    for (let i = 0; i <= (useDummy ? insertIdx + 1 : insertIdx); i++) {
      snap(all, [all[i].id], `遍历到位置 ${i}，节点值=${all[i].dummy ? "dummy" : all[i].val}`, "highlight");
    }
    snap(all, [refNode.id], `找到目标节点 ${refNode.val}，准备在其后插入 ${targetVal}`, "highlight");
    const inserted = [...all.slice(0, (useDummy ? insertIdx + 2 : insertIdx + 1)), newNode, ...all.slice(useDummy ? insertIdx + 2 : insertIdx + 1)];
    snap(inserted, [newNode.id], `newNode.next = prev.next；prev.next = newNode，插入完成`, "success");
    return steps;
  }

  if (op === "delete-head") {
    if (ns.length === 0) {
      snap(ns, [], "链表为空，无法删除", "normal");
      return steps;
    }
    const all = useDummy ? [dummy, ...ns] : ns;
    snap(all, [all[useDummy ? 1 : 0].id], `定位头节点 ${ns[0].val}`, "highlight");
    const removed = useDummy ? [dummy, ...ns.slice(1)] : ns.slice(1);
    snap(removed, useDummy ? [dummy.id] : [], `${useDummy ? "dummy.next" : "head"} = head.next，头节点 ${ns[0].val} 已移除`, "remove");
    return steps;
  }

  if (op === "delete-node") {
    if (ns.length === 0) { snap(ns, [], "链表为空", "normal"); return steps; }
    const delIdx = Math.min(1, ns.length - 1);
    const all = useDummy ? [dummy, ...ns] : ns;
    for (let i = 0; i < (useDummy ? delIdx + 1 : delIdx); i++) {
      snap(all, [all[i].id], `遍历：节点 ${all[i].dummy ? "dummy" : all[i].val}`, "highlight");
    }
    const prevNode = useDummy ? all[delIdx] : all[delIdx - 1 < 0 ? 0 : delIdx - 1];
    const delNode = all[useDummy ? delIdx + 1 : delIdx];
    snap(all, [prevNode.id, delNode.id], `prev=${prevNode.dummy ? "dummy" : prevNode.val}，curr（待删）=${delNode.val}`, "highlight");
    const afterDel = useDummy ? [dummy, ...ns.filter((_, i) => i !== delIdx)] : ns.filter((_, i) => i !== delIdx);
    snap(afterDel, [prevNode.id], `prev.next = curr.next，节点 ${delNode.val} 删除`, "remove");
    return steps;
  }

  if (op === "traverse") {
    const all = useDummy ? [dummy, ...ns] : ns;
    for (let i = 0; i < all.length; i++) {
      snap(all, [all[i].id], `访问节点 ${all[i].dummy ? "dummy(哨兵)" : all[i].val}，next=${i < all.length - 1 ? (all[i + 1].dummy ? "dummy" : all[i + 1].val) : "null"}`, "highlight");
    }
    snap(all, [], "遍历完成，curr = null，退出循环", "success");
    return steps;
  }

  return steps;
}

const OP_LABELS: Record<Op, string> = {
  "insert-head":  "头插",
  "insert-tail":  "尾插",
  "insert-after": "中间插（位置2后）",
  "delete-head":  "删头节点",
  "delete-node":  "删中间节点",
  "traverse":     "遍历",
};

export default function LinkedListOperations() {
  const [op, setOp] = useState<Op>("insert-head");
  const [useDummy, setUseDummy] = useState(true);
  const [targetVal, setTargetVal] = useState(99);
  const [nodes, setNodes] = useState<LLNode[]>(buildInitialNodes);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const steps = React.useMemo(
    () => buildSteps(nodes, op, targetVal, useDummy),
    [nodes.map((n) => n.id + "_" + n.val).join(","), op, targetVal, useDummy]
  );

  const cur = steps[Math.min(step, steps.length - 1)];

  const applyOp = () => {
    const last = steps[steps.length - 1];
    if (last) setNodes(last.nodes.filter((n) => !n.dummy));
    setStep(0);
    setPlaying(false);
  };

  const startPlay = useCallback(() => {
    if (step >= steps.length - 1) setStep(0);
    setPlaying(true);
  }, [step, steps.length]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setStep((s) => { if (s >= steps.length - 1) { setPlaying(false); return s; } return s + 1; });
      }, 900);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, steps.length]);

  const reset = () => { setNodes(buildInitialNodes()); setStep(0); setPlaying(false); };

  const PHASE_COLORS: Record<Step["phase"], { node: string; border: string }> = {
    normal:    { node: "bg-bg-secondary border-border-subtle text-text-primary", border: "" },
    highlight: { node: "bg-blue-500/20 border-blue-400/60 text-blue-700 dark:text-blue-300", border: "" },
    success:   { node: "bg-emerald-500/20 border-emerald-400/60 text-emerald-700 dark:text-emerald-300", border: "" },
    remove:    { node: "bg-bg-secondary border-border-subtle text-text-primary", border: "" },
  };

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-4">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-indigo-500/15 dark:bg-indigo-500/20 flex items-center justify-center text-xl">🔗</div>
        <div>
          <h3 className="font-bold text-text-primary text-base">链表操作步进演示</h3>
          <p className="text-xs text-text-secondary">观察头插/尾插/中间插/删除时指针如何变化，可开关哨兵节点对比</p>
        </div>
      </div>

      {/* 操作选择 */}
      <div className="flex flex-wrap gap-2 border-t border-border-subtle pt-3">
        {(Object.keys(OP_LABELS) as Op[]).map((o) => (
          <button key={o} onClick={() => { setOp(o); setStep(0); setPlaying(false); }}
            className={`px-3 py-1.5 rounded-xl border text-xs font-medium transition-all ${op === o
              ? "bg-indigo-500/20 border-indigo-400/60 text-indigo-700 dark:text-indigo-300"
              : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}>
            {OP_LABELS[o]}
          </button>
        ))}
      </div>

      {/* 控制条 */}
      <div className="flex flex-wrap gap-3 items-center">
        <label className="flex items-center gap-2 text-xs text-text-secondary cursor-pointer">
          <input type="checkbox" checked={useDummy} onChange={(e) => { setUseDummy(e.target.checked); setStep(0); }}
            className="w-4 h-4 rounded accent-indigo-500" />
          显示哨兵节点（Dummy Head）
        </label>
        {(op === "insert-head" || op === "insert-tail" || op === "insert-after") && (
          <label className="flex items-center gap-2 text-xs text-text-secondary">
            插入值：
            <input type="number" value={targetVal} onChange={(e) => { setTargetVal(Number(e.target.value)); setStep(0); }}
              className="w-16 bg-bg-tertiary border border-border-subtle rounded-lg px-2 py-1 text-xs font-mono text-text-primary outline-none" />
          </label>
        )}
        <div className="ml-auto flex items-center gap-2">
          <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center">‹</button>
          <button onClick={playing ? () => setPlaying(false) : startPlay}
            className="px-3 py-1 rounded-lg bg-indigo-500/15 hover:bg-indigo-500/25 text-indigo-700 dark:text-indigo-300 text-xs font-medium transition-colors">
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))} disabled={step >= steps.length - 1}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center">›</button>
          <button onClick={applyOp} className="px-2 py-1 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-700 dark:text-emerald-300 text-xs transition-colors">应用</button>
          <button onClick={reset} className="px-2 py-1 rounded-lg bg-bg-tertiary hover:bg-border-subtle text-text-secondary text-xs transition-colors">重置</button>
        </div>
      </div>

      {/* 进度 */}
      <div className="h-1 bg-bg-tertiary rounded-full overflow-hidden">
        <div className="h-full bg-indigo-500 rounded-full transition-all duration-300"
          style={{ width: `${((step + 1) / steps.length) * 100}%` }} />
      </div>

      {/* 步骤说明 */}
      <div className={`rounded-lg border border-border-subtle px-3 py-2 text-xs font-medium transition-colors
        ${cur.phase === "success" ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300" :
          cur.phase === "highlight" ? "bg-blue-500/10 text-blue-700 dark:text-blue-300" :
          cur.phase === "remove" ? "bg-rose-500/10 text-rose-700 dark:text-rose-300" :
          "bg-bg-tertiary text-text-secondary"}`}>
        步骤 {step + 1}/{steps.length}：{cur?.label}
      </div>

      {/* 链表可视化 */}
      <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4 overflow-x-auto">
        <div className="flex items-center gap-1 min-w-max">
          {cur?.nodes.map((n, i) => {
            const isHl = cur.highlightIds.includes(n.id);
            const baseColor = isHl
              ? (cur.phase === "success" ? "bg-emerald-500/20 border-emerald-400/60 text-emerald-700 dark:text-emerald-300" :
                 cur.phase === "remove" ? "bg-rose-500/20 border-rose-400/60 text-rose-700 dark:text-rose-300" :
                 "bg-blue-500/20 border-blue-400/60 text-blue-700 dark:text-blue-300")
              : n.dummy ? "bg-violet-500/10 border-violet-400/40 text-violet-700 dark:text-violet-300"
              : "bg-bg-secondary border-border-subtle text-text-primary";

            return (
              <React.Fragment key={n.id}>
                <div className={`flex flex-col items-center transition-all duration-300`}>
                  <div className={`rounded-lg border px-3 py-2 text-sm font-bold font-mono min-w-[48px] text-center transition-all duration-300 ${baseColor}`}>
                    {n.dummy ? "D" : n.val}
                  </div>
                  <span className="text-[9px] text-text-tertiary mt-0.5">{n.dummy ? "dummy" : `val=${n.val}`}</span>
                </div>
                {i < cur.nodes.length - 1 && (
                  <div className="flex items-center gap-0.5 text-text-tertiary">
                    <span className="text-[10px]">→</span>
                  </div>
                )}
              </React.Fragment>
            );
          })}
          <div className="flex items-center gap-0.5 ml-1">
            <span className="text-xs text-text-tertiary">→ null</span>
          </div>
        </div>
      </div>

      {/* 哨兵说明 */}
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary px-3 py-2.5 text-xs text-text-secondary space-y-1">
        <span className="font-semibold text-text-primary">
          {useDummy ? "✅ 已启用哨兵节点（Dummy Head）" : "⚠️ 未启用哨兵节点"}
        </span>
        <div>
          {useDummy
            ? "所有节点的处理逻辑统一：prev.next = curr，无需对 head=null 或 head 被删做特殊判断。"
            : "需要单独处理 head 边界：头插修改 head，头删更新 head，代码分支更多。"}
        </div>
      </div>
    </div>
  );
}
