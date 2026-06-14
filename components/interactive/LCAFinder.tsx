"use client";
import React, { useState, useMemo } from "react";

/** LCAFinder — 最低公共祖先查找过程可视化：路径高亮 + 汇聚点标记 */

// ─────────────── 数据结构 ───────────────────────────
interface TNode { val: number; left: TNode | null; right: TNode | null; id: number; }
interface LayoutNode extends TNode { x: number; y: number; left: LayoutNode | null; right: LayoutNode | null; }

let _uid = 0;
function n(val: number, l: TNode | null = null, r: TNode | null = null): TNode {
  return { val, left: l, right: r, id: _uid++ };
}

const TREES: Record<string, { tree: TNode; pairs: [number, number][] }> = {
  "标准例题": {
    tree: (() => { _uid = 0; return n(3, n(5, n(6), n(2, n(7), n(4))), n(1, n(0), n(8))); })(),
    pairs: [[7, 4], [5, 1], [6, 4], [7, 8]],
  },
  "完美二叉树": {
    tree: (() => { _uid = 0; return n(1, n(2, n(4), n(5)), n(3, n(6), n(7))); })(),
    pairs: [[4, 7], [4, 5], [2, 3]],
  },
  "链状退化": {
    tree: (() => { _uid = 0; return n(1, n(2, n(3, n(4, n(5), null), null), null), null); })(),
    pairs: [[3, 5], [2, 4]],
  },
};

function layout(node: TNode | null, depth: number, xMin: number, xMax: number): LayoutNode | null {
  if (!node) return null;
  const x = (xMin + xMax) / 2, y = depth * 65 + 35;
  return { ...node, x, y, left: layout(node.left, depth + 1, xMin, (xMin + xMax) / 2), right: layout(node.right, depth + 1, (xMin + xMax) / 2, xMax) };
}

// ─────────────── 步骤生成 ───────────────────────────
interface LCAStep {
  activeId: number | null;
  pFoundSet: Set<number>;
  qFoundSet: Set<number>;
  lcaId: number | null;
  returned: Map<number, 'none' | 'p' | 'q' | 'both' | 'lca'>;
  description: string;
  phase: 'search' | 'found' | 'done';
}

function buildLCASteps(root: TNode | null, pId: number, qId: number): LCAStep[] {
  const steps: LCAStep[] = [];
  const returned = new Map<number, 'none' | 'p' | 'q' | 'both' | 'lca'>();
  let lcaId: number | null = null;

  function dfs(node: TNode | null, pFound: Set<number>, qFound: Set<number>): 'none' | TNode {
    if (!node) return 'none';

    const pMatch = node.id === pId;
    const qMatch = node.id === qId;

    steps.push({
      activeId: node.id, pFoundSet: new Set(pFound), qFoundSet: new Set(qFound), lcaId,
      returned: new Map(returned),
      description: pMatch
        ? `进入节点 ${node.val}：找到 P！${qFound.size > 0 ? '已找到 Q，当前节点就是 LCA！' : '继续递归右子树以找 Q'}`
        : qMatch
        ? `进入节点 ${node.val}：找到 Q！${pFound.size > 0 ? '已找到 P，当前节点就是 LCA！' : '继续递归右子树以找 P'}`
        : `进入节点 ${node.val}，向左右子树搜索 P 和 Q`,
      phase: 'search',
    });

    if (pMatch || qMatch) {
      returned.set(node.id, pMatch ? 'p' : 'q');
      if (pMatch) pFound.add(node.id);
      if (qMatch) qFound.add(node.id);
      return node;  // 找到目标节点，直接返回
    }

    const left = dfs(node.left, pFound, qFound);
    const right = dfs(node.right, pFound, qFound);

    if (left !== 'none' && right !== 'none') {
      // 两侧都找到了，当前节点是 LCA
      lcaId = node.id;
      returned.set(node.id, 'lca');
      steps.push({
        activeId: node.id, pFoundSet: new Set(pFound), qFoundSet: new Set(qFound), lcaId,
        returned: new Map(returned),
        description: `✅ 节点 ${node.val} 的左右子树分别找到 P 和 Q，${node.val} 就是最低公共祖先！`,
        phase: 'found',
      });
      return node;
    }

    const result = left !== 'none' ? left : right;
    if (result !== 'none') {
      returned.set(node.id, returned.get(result.id) ?? 'none');
    } else {
      returned.set(node.id, 'none');
    }

    steps.push({
      activeId: node.id, pFoundSet: new Set(pFound), qFoundSet: new Set(qFound), lcaId,
      returned: new Map(returned),
      description: result !== 'none'
        ? `节点 ${node.val} 返回：在${left !== 'none' ? '左' : '右'}子树中找到了目标，继续向上传递`
        : `节点 ${node.val} 返回：左右子树均未找到，返回 null`,
      phase: 'search',
    });

    return result === 'none' ? 'none' : result;
  }

  dfs(root, new Set(), new Set());
  return steps;
}

// ─────────────── SVG 树渲染 ─────────────────────────
function TreeSVG({ tree, step, pId, qId }: { tree: LayoutNode | null; step: LCAStep; pId: number; qId: number }) {
  const W = 340, H = 240;

  function edges(node: LayoutNode | null): React.ReactNode[] {
    if (!node) return [];
    const res: React.ReactNode[] = [];
    if (node.left) res.push(<line key={`el${node.id}`} x1={node.x} y1={node.y} x2={node.left.x} y2={node.left.y} stroke="#374151" strokeWidth={1.5} />);
    if (node.right) res.push(<line key={`er${node.id}`} x1={node.x} y1={node.y} x2={node.right.x} y2={node.right.y} stroke="#374151" strokeWidth={1.5} />);
    return [...res, ...edges(node.left), ...edges(node.right)];
  }

  function getColor(node: LayoutNode) {
    const ret = step.returned.get(node.id);
    if (node.id === step.lcaId) return { fill: '#f59e0b', stroke: '#fbbf24', text: '#000' };
    if (node.id === pId) return { fill: '#3b82f6', stroke: '#93c5fd', text: '#fff' };
    if (node.id === qId) return { fill: '#a855f7', stroke: '#d8b4fe', text: '#fff' };
    if (node.id === step.activeId) return { fill: '#0891b2', stroke: '#67e8f9', text: '#fff' };
    if (ret === 'p') return { fill: '#1e40af', stroke: '#60a5fa', text: '#93c5fd' };
    if (ret === 'q') return { fill: '#6b21a8', stroke: '#c084fc', text: '#d8b4fe' };
    if (ret === 'none') return { fill: '#111827', stroke: '#4b5563', text: '#6b7280' };
    return { fill: '#1e293b', stroke: '#4b5563', text: '#e2e8f0' };
  }

  function nodes(node: LayoutNode | null): React.ReactNode[] {
    if (!node) return [];
    const c = getColor(node);
    const isActive = node.id === step.activeId || node.id === pId || node.id === qId || node.id === step.lcaId;
    return [
      <circle key={`c${node.id}`} cx={node.x} cy={node.y} r={18} fill={c.fill} stroke={c.stroke} strokeWidth={isActive ? 2.5 : 1.5} />,
      <text key={`v${node.id}`} x={node.x} y={node.y + 1} textAnchor="middle" dominantBaseline="middle" fill={c.text} fontSize={12} fontWeight="bold">{node.val}</text>,
      // P/Q/LCA 标签
      node.id === pId && <text key={`lp${node.id}`} x={node.x - 22} y={node.y - 4} fill="#93c5fd" fontSize={10} fontWeight="bold">P</text>,
      node.id === qId && <text key={`lq${node.id}`} x={node.x + 14} y={node.y - 4} fill="#d8b4fe" fontSize={10} fontWeight="bold">Q</text>,
      node.id === step.lcaId && <text key={`ll${node.id}`} x={node.x} y={node.y - 26} textAnchor="middle" fill="#fbbf24" fontSize={10} fontWeight="bold">LCA</text>,
      ...nodes(node.left), ...nodes(node.right),
    ].filter(Boolean);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 240 }}>
      {edges(tree)}
      {nodes(tree)}
    </svg>
  );
}

// ─────────────── 主组件 ─────────────────────────────
export default function LCAFinder() {
  const [treeName, setTreeName] = useState("标准例题");
  const [pairIdx, setPairIdx] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);

  const { tree: rawTree, pairs } = TREES[treeName];
  const tree = useMemo(() => layout(rawTree, 0, 0, 340), [rawTree]);
  const [pId, qId] = pairs[pairIdx % pairs.length];
  const steps = useMemo(() => buildLCASteps(rawTree, pId, qId), [rawTree, pId, qId]);
  const step = steps[Math.min(stepIdx, steps.length - 1)];

  const reset = () => setStepIdx(0);

  // 找到 P/Q 对应节点的 val
  function findVal(node: TNode | null, id: number): number | null {
    if (!node) return null;
    if (node.id === id) return node.val;
    return findVal(node.left, id) ?? findVal(node.right, id);
  }
  const pVal = findVal(rawTree, pId);
  const qVal = findVal(rawTree, qId);

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 text-sm font-mono">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-base font-bold text-text-primary">🔍 最低公共祖先（LCA）可视化</h3>
          <p className="text-xs text-text-tertiary">后序遍历：归并两侧子树的搜索结果</p>
        </div>
        <div className="flex gap-2">
          <div className="flex items-center gap-1.5 px-2 py-1 bg-blue-500/10 border border-blue-500/30 rounded text-xs text-blue-300">
            <span className="w-2 h-2 rounded-full bg-blue-500" />P={pVal}
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded text-xs text-purple-300">
            <span className="w-2 h-2 rounded-full bg-purple-500" />Q={qVal}
          </div>
          {step.lcaId !== null && (
            <div className="flex items-center gap-1.5 px-2 py-1 bg-amber-500/10 border border-amber-500/30 rounded text-xs text-amber-300">
              <span className="w-2 h-2 rounded-full bg-amber-500" />LCA={findVal(rawTree, step.lcaId)}
            </div>
          )}
        </div>
      </div>

      {/* 预设树选择 */}
      <div className="flex gap-2 flex-wrap">
        {Object.keys(TREES).map(t => (
          <button key={t} onClick={() => { setTreeName(t); setPairIdx(0); reset(); }}
            className={`px-2 py-1 rounded text-xs border transition-colors ${treeName === t ? "bg-blue-600 text-white border-blue-600" : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"}`}>
            {t}
          </button>
        ))}
      </div>

      {/* P/Q 对选择 */}
      <div className="flex gap-2 flex-wrap items-center">
        <span className="text-xs text-text-tertiary">选择节点对：</span>
        {pairs.map(([p, q], i) => {
          const pv = findVal(rawTree, p), qv = findVal(rawTree, q);
          return (
            <button key={i} onClick={() => { setPairIdx(i); reset(); }}
              className={`px-2 py-1 rounded text-xs border transition-colors ${pairIdx === i ? "bg-purple-600 text-white border-purple-600" : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-purple-400"}`}>
              P={pv}, Q={qv}
            </button>
          );
        })}
      </div>

      {/* 树可视化 */}
      <div className="bg-bg-tertiary rounded-lg border border-border-subtle p-3">
        <TreeSVG tree={tree} step={step} pId={pId} qId={qId} />
        <div className="flex gap-4 text-xs text-text-secondary mt-2">
          <span><span className="inline-block w-3 h-3 rounded-full bg-blue-500 mr-1" />P节点</span>
          <span><span className="inline-block w-3 h-3 rounded-full bg-purple-500 mr-1" />Q节点</span>
          <span><span className="inline-block w-3 h-3 rounded-full bg-amber-500 mr-1" />LCA</span>
          <span><span className="inline-block w-3 h-3 rounded-full bg-cyan-600 mr-1" />当前搜索</span>
        </div>
      </div>

      {/* 步骤说明 */}
      <div className={`rounded-lg p-3 border min-h-[56px] transition-colors ${
        step.phase === 'found' ? "bg-amber-500/10 border-amber-500/40" :
        step.phase === 'search' ? "bg-bg-tertiary border-border-subtle" : "bg-bg-tertiary border-border-subtle"
      }`}>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-text-tertiary">步骤 {stepIdx + 1} / {steps.length}</span>
          <span className={`text-xs font-bold ${step.phase === 'found' ? "text-amber-300" : "text-text-tertiary"}`}>
            {step.phase === 'found' ? '✅ 找到 LCA！' : '🔍 搜索中…'}
          </span>
        </div>
        <p className={`text-sm ${step.phase === 'found' ? "text-amber-300" : "text-text-primary"}`}>
          {step.description}
        </p>
      </div>

      {/* 算法伪代码 */}
      <div className="bg-bg-tertiary rounded p-3 border border-border-subtle text-xs text-text-secondary space-y-0.5">
        <div className="font-bold text-text-primary mb-1">后序递归框架</div>
        <div><span className="text-blue-400">if</span> root <span className="text-text-tertiary">is None or</span> root <span className="text-text-tertiary">is</span> p <span className="text-text-tertiary">or</span> root <span className="text-text-tertiary">is</span> q: <span className="text-green-400">return</span> root</div>
        <div>left = lca(root.left, p, q)</div>
        <div>right = lca(root.right, p, q)</div>
        <div><span className="text-blue-400">if</span> left <span className="text-blue-400">and</span> right: <span className="text-amber-400">return root</span> <span className="text-text-tertiary"># p、q 各在两侧</span></div>
        <div><span className="text-green-400">return</span> left <span className="text-blue-400">if</span> left <span className="text-blue-400">else</span> right</div>
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
