"use client";
import React, { useState, useMemo } from "react";

/** TreeSerializeDemo — 序列化/反序列化过程动画：前序+null → token数组 → 重建 */

interface TNode { val: number | null; left: TNode | null; right: TNode | null; id: number; }
interface LayoutNode extends TNode { x: number; y: number; left: LayoutNode | null; right: LayoutNode | null; }

let _uid = 0;
function n(val: number | null, l: TNode | null = null, r: TNode | null = null): TNode {
  return { val, left: l, right: r, id: _uid++ };
}

const PRESETS: Record<string, TNode> = (() => {
  _uid = 0; const t1 = n(1, n(2), n(3, n(4), n(5)));
  _uid = 0; const t2 = n(1, n(2, n(3), null), n(4, null, n(5)));
  _uid = 0; const t3 = n(1, n(2, n(3, n(4), null), null), null);
  return { "标准例题": t1, "不完整树": t2, "退化左链": t3 };
})();

function layout(node: TNode | null, depth: number, xMin: number, xMax: number): LayoutNode | null {
  if (!node) return null;
  const x = (xMin + xMax) / 2, y = depth * 65 + 35;
  return { ...node, x, y, left: layout(node.left, depth + 1, xMin, (xMin + xMax) / 2), right: layout(node.right, depth + 1, (xMin + xMax) / 2, xMax) };
}

// ─────────────── 步骤生成（序列化）───────────────────
interface SerStep {
  nodeId: number | null; // 当前访问的节点 id（null = 空节点）
  tokens: string[];      // 已生成的 token 序列
  highlightTokenIdx: number; // 新加入 token 的位置
  description: string;
  phase: 'serialize' | 'done';
}

function buildSerializeSteps(root: TNode | null): SerStep[] {
  const steps: SerStep[] = [];
  const tokens: string[] = [];

  function dfs(node: TNode | null) {
    if (!node) {
      const idx = tokens.length;
      tokens.push('#');
      steps.push({
        nodeId: null, tokens: [...tokens], highlightTokenIdx: idx,
        description: `空节点 → 写入 "#"（token[${idx}]）`,
        phase: 'serialize',
      });
      return;
    }
    const idx = tokens.length;
    tokens.push(String(node.val));
    steps.push({
      nodeId: node.id, tokens: [...tokens], highlightTokenIdx: idx,
      description: `访问节点 ${node.val} → 写入 "${node.val}"（token[${idx}]）`,
      phase: 'serialize',
    });
    dfs(node.left);
    dfs(node.right);
  }

  dfs(root);
  steps.push({ nodeId: null, tokens: [...tokens], highlightTokenIdx: -1, description: `✅ 序列化完成！共 ${tokens.length} 个 token（${tokens.filter(t => t === '#').length} 个 null，${tokens.filter(t => t !== '#').length} 个节点）`, phase: 'done' });
  return steps;
}

// ─────────────── 步骤生成（反序列化）─────────────────
interface DeserStep {
  consumedUpTo: number;   // 已消费到第几个 token（0-based exclusive）
  builtNodes: number[];   // 已构建的节点 id 列表
  builtNulls: string[];   // 已构建的 null 标记
  description: string;
  phase: 'deserialize' | 'done';
}

function buildDeserializeSteps(tokens: string[]): DeserStep[] {
  const steps: DeserStep[] = [];
  let cursor = 0;
  const builtNodes: number[] = [], builtNulls: string[] = [];

  function build(): void {
    const idx = cursor++;
    const tok = tokens[idx];
    if (tok === '#') {
      builtNulls.push(`null@${idx}`);
      steps.push({ consumedUpTo: cursor, builtNodes: [...builtNodes], builtNulls: [...builtNulls], description: `消费 token[${idx}]="${tok}" → 构建空节点（null）`, phase: 'deserialize' });
      return;
    }
    const nodeLabel = `节点(${tok})@${idx}`;
    steps.push({ consumedUpTo: cursor, builtNodes: [...builtNodes], builtNulls: [...builtNulls], description: `消费 token[${idx}]="${tok}" → 创建节点 ${tok}，递归构建左子树…`, phase: 'deserialize' });
    build();
    builtNodes.push(idx);
    steps.push({ consumedUpTo: cursor, builtNodes: [...builtNodes], builtNulls: [...builtNulls], description: `节点 ${tok} 左子树构建完成，递归构建右子树…`, phase: 'deserialize' });
    build();
  }

  if (tokens.length > 0) build();
  steps.push({ consumedUpTo: cursor, builtNodes: [...builtNodes], builtNulls: [...builtNulls], description: `✅ 反序列化完成！已还原原始树结构`, phase: 'done' });
  return steps;
}

// ─────────────── SVG 树渲染 ─────────────────────────
function TreeSVG({ tree, activeId }: { tree: LayoutNode | null; activeId: number | null }) {
  function edges(node: LayoutNode | null): React.ReactNode[] {
    if (!node) return [];
    const res: React.ReactNode[] = [];
    if (node.left) res.push(<line key={`el${node.id}`} x1={node.x} y1={node.y} x2={node.left.x} y2={node.left.y} stroke="#374151" strokeWidth={1.5} />);
    if (node.right) res.push(<line key={`er${node.id}`} x1={node.x} y1={node.y} x2={node.right.x} y2={node.right.y} stroke="#374151" strokeWidth={1.5} />);
    return [...res, ...edges(node.left), ...edges(node.right)];
  }
  function nodes(node: LayoutNode | null): React.ReactNode[] {
    if (!node) return [];
    const isActive = node.id === activeId;
    return [
      <circle key={`c${node.id}`} cx={node.x} cy={node.y} r={17} fill={isActive ? '#2563eb' : '#1e293b'} stroke={isActive ? '#93c5fd' : '#4b5563'} strokeWidth={isActive ? 2.5 : 1.5} />,
      <text key={`v${node.id}`} x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="middle" fill="white" fontSize={12} fontWeight="bold">{node.val}</text>,
      ...nodes(node.left), ...nodes(node.right),
    ];
  }
  return (
    <svg viewBox="0 0 280 200" className="w-full" style={{ maxHeight: 180 }}>
      {edges(tree)}{nodes(tree)}
    </svg>
  );
}

// ─────────────── 主组件 ─────────────────────────────
export default function TreeSerializeDemo() {
  const [preset, setPreset] = useState("标准例题");
  const [mode, setMode] = useState<'serialize' | 'deserialize'>('serialize');
  const [stepIdx, setStepIdx] = useState(0);

  const tree = useMemo(() => layout(PRESETS[preset], 0, 0, 280), [preset]);
  const serSteps = useMemo(() => buildSerializeSteps(PRESETS[preset]), [preset]);
  const tokens = serSteps[serSteps.length - 2]?.tokens ?? [];
  const deserSteps = useMemo(() => buildDeserializeSteps(tokens), [tokens]);
  const steps = mode === 'serialize' ? serSteps : deserSteps;
  const step = steps[Math.min(stepIdx, steps.length - 1)];

  const reset = () => setStepIdx(0);

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 text-sm font-mono">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-base font-bold text-text-primary">📦 序列化 / 反序列化动画</h3>
          <p className="text-xs text-text-tertiary">前序遍历 + null 标记 → 唯一确定树结构</p>
        </div>
        <div className="flex gap-1 rounded-lg overflow-hidden border border-border-subtle">
          {(['serialize', 'deserialize'] as const).map(m => (
            <button key={m} onClick={() => { setMode(m); reset(); }}
              className={`px-3 py-1.5 text-xs transition-colors ${mode === m ? "bg-blue-600 text-white" : "bg-bg-tertiary text-text-secondary hover:bg-bg-secondary"}`}>
              {m === 'serialize' ? '📤 序列化' : '📥 反序列化'}
            </button>
          ))}
        </div>
      </div>

      {/* 预设 */}
      <div className="flex gap-2 flex-wrap">
        {Object.keys(PRESETS).map(p => (
          <button key={p} onClick={() => { setPreset(p); reset(); }}
            className={`px-2 py-1 rounded text-xs border transition-colors ${preset === p ? "bg-blue-600 text-white border-blue-600" : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"}`}>
            {p}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 树 */}
        <div className="bg-bg-tertiary rounded-lg border border-border-subtle p-3">
          <div className="text-xs text-text-tertiary mb-1">原始树（前序：根→左→右）</div>
          <TreeSVG tree={tree} activeId={mode === 'serialize' ? ('nodeId' in step ? (step as SerStep).nodeId : null) : null} />
        </div>

        {/* Token 序列 */}
        <div className="bg-bg-tertiary rounded-lg border border-border-subtle p-3">
          <div className="text-xs text-text-tertiary mb-2">
            {mode === 'serialize' ? '序列化输出（token 序列）' : '反序列化输入（逐步消费）'}
          </div>
          <div className="flex flex-wrap gap-1">
            {mode === 'serialize'
              ? (step as SerStep).tokens.map((tok, i) => (
                  <div key={i} className={`flex flex-col items-center`}>
                    <div className={`w-8 h-8 rounded flex items-center justify-center text-xs font-bold border transition-all ${
                      i === (step as SerStep).highlightTokenIdx
                        ? tok === '#' ? "bg-gray-600 text-white border-gray-400" : "bg-blue-600 text-white border-blue-400"
                        : tok === '#' ? "bg-gray-800 text-gray-500 border-gray-700" : "bg-bg-secondary text-text-primary border-border-subtle"
                    }`}>{tok}</div>
                    <span className="text-[9px] text-text-tertiary">{i}</span>
                  </div>
                ))
              : tokens.map((tok, i) => {
                  const consumed = (step as DeserStep).consumedUpTo;
                  return (
                    <div key={i} className="flex flex-col items-center">
                      <div className={`w-8 h-8 rounded flex items-center justify-center text-xs font-bold border transition-all ${
                        i === consumed - 1
                          ? tok === '#' ? "bg-gray-600 text-white border-gray-400" : "bg-green-600 text-white border-green-400"
                          : i < consumed
                          ? "bg-bg-secondary/50 text-text-tertiary border-border-subtle opacity-50"
                          : tok === '#' ? "bg-gray-800 text-gray-500 border-gray-700" : "bg-bg-secondary text-text-primary border-border-subtle"
                      }`}>{tok}</div>
                      <span className="text-[9px] text-text-tertiary">{i}</span>
                    </div>
                  );
                })
            }
          </div>
          {mode === 'serialize' && tokens.length > 0 && (stepIdx === steps.length - 1) && (
            <div className="mt-2 text-xs text-text-tertiary break-all">
              完整字符串：<span className="text-blue-300">&quot;{tokens.join(',')}&quot;</span>
            </div>
          )}
        </div>
      </div>

      {/* 步骤说明 */}
      <div className={`rounded-lg p-3 border min-h-[56px] transition-all ${
        step.phase === 'done' ? "bg-green-500/10 border-green-500/40" : "bg-bg-tertiary border-border-subtle"
      }`}>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-text-tertiary">步骤 {stepIdx + 1} / {steps.length}</span>
          <span className="text-text-tertiary">{mode === 'serialize' ? '前序 DFS' : 'token 消费'}</span>
        </div>
        <p className={`text-sm ${step.phase === 'done' ? "text-green-300" : "text-text-primary"}`}>
          {step.description}
        </p>
      </div>

      {/* 关键洞察 */}
      <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle text-xs text-text-secondary space-y-1">
        <div className="font-bold text-text-primary">为什么需要 null 标记？</div>
        <div>纯前序（如 <span className="text-amber-300">1,2,3</span>）无法区分左偏树和右偏树。加上 <span className="text-gray-400">#</span> 后，每棵不同的树产生唯一的序列。</div>
        <div className="text-text-tertiary">节点数 = n，null 标记数 = n+1（满二叉树叶子性质），总 token 数 = 2n+1</div>
      </div>

      {/* 控制 */}
      <div className="flex gap-2 justify-center">
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle text-xs hover:border-blue-400">↩ 重置</button>
        <button onClick={() => setStepIdx(i => Math.max(0, i - 1))} disabled={stepIdx === 0} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle text-xs disabled:opacity-40 hover:border-blue-400">← 上一步</button>
        <button onClick={() => setStepIdx(i => Math.min(steps.length - 1, i + 1))} disabled={stepIdx >= steps.length - 1} className="px-4 py-2 rounded-lg bg-blue-600 text-white text-xs disabled:opacity-40 hover:bg-blue-700">下一步 →</button>
      </div>
    </div>
  );
}
