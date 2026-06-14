"use client";

import { useState, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, Trash2, RotateCcw, AlertTriangle, CheckCircle, ArrowRight } from "lucide-react";

type NodeType = "process" | "resource";
type EdgeType = "request" | "assignment";

interface GraphNode {
  id: string;
  type: NodeType;
  x: number;
  y: number;
  instances: number;
}

interface GraphEdge {
  id: string;
  from: string;
  to: string;
  type: EdgeType;
}

const NODE_RADIUS = 30;
const SVG_W = 700;
const SVG_H = 420;

export default function ResourceAllocationGraph() {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [nextPid, setNextPid] = useState(1);
  const [nextRid, setNextRid] = useState(1);
  const [edgeMode, setEdgeMode] = useState(false);
  const [edgeFrom, setEdgeFrom] = useState<string | null>(null);
  const [edgeType, setEdgeType] = useState<EdgeType>("request");
  const [highlightedCycle, setHighlightedCycle] = useState<string[]>([]);
  const [log, setLog] = useState<string[]>([]);

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [msg, ...prev].slice(0, 10));
  }, []);

  const addProcess = useCallback(() => {
    const id = `P${nextPid}`;
    const angle = (nodes.filter((n) => n.type === "process").length * Math.PI * 2) / 5;
    const cx = SVG_W * 0.3 + Math.cos(angle) * 100;
    const cy = SVG_H / 2 + Math.sin(angle) * 100;
    setNodes((prev) => [...prev, { id, type: "process", x: cx, y: cy, instances: 1 }]);
    setNextPid((p) => p + 1);
    addLog(`Added process ${id}`);
  }, [nextPid, nodes, addLog]);

  const addResource = useCallback((instances: number = 1) => {
    const id = `R${nextRid}`;
    const angle = (nodes.filter((n) => n.type === "resource").length * Math.PI * 2) / 5;
    const cx = SVG_W * 0.7 + Math.cos(angle) * 100;
    const cy = SVG_H / 2 + Math.sin(angle) * 100;
    setNodes((prev) => [...prev, { id, type: "resource", x: cx, y: cy, instances }]);
    setNextRid((r) => r + 1);
    addLog(`Added resource ${id} (${instances} instance${instances > 1 ? "s" : ""})`);
  }, [nextRid, nodes, addLog]);

  const removeNode = useCallback((nodeId: string) => {
    setNodes((prev) => prev.filter((n) => n.id !== nodeId));
    setEdges((prev) => prev.filter((e) => e.from !== nodeId && e.to !== nodeId));
    setHighlightedCycle([]);
    addLog(`Removed ${nodeId}`);
  }, [addLog]);

  const handleNodeClick = useCallback((nodeId: string) => {
    if (!edgeMode) return;
    if (!edgeFrom) {
      setEdgeFrom(nodeId);
      addLog(`Selected ${nodeId} as edge source`);
    } else {
      if (edgeFrom === nodeId) {
        setEdgeFrom(null);
        return;
      }
      const sourceNode = nodes.find((n) => n.id === edgeFrom);
      const targetNode = nodes.find((n) => n.id === nodeId);
      if (!sourceNode || !targetNode) return;

      // Validate: request = process->resource, assignment = resource->process
      let actualType = edgeType;
      if (sourceNode.type === "process" && targetNode.type === "resource") {
        actualType = "request";
      } else if (sourceNode.type === "resource" && targetNode.type === "process") {
        actualType = "assignment";
      } else {
        addLog("Invalid: edges must go between process and resource");
        setEdgeFrom(null);
        return;
      }

      const edgeId = `${edgeFrom}->${nodeId}`;
      if (edges.some((e) => e.id === edgeId)) {
        addLog("Edge already exists");
        setEdgeFrom(null);
        return;
      }

      setEdges((prev) => [...prev, { id: edgeId, from: edgeFrom, to: nodeId, type: actualType }]);
      addLog(`${actualType === "request" ? "Request" : "Assignment"}: ${edgeFrom} -> ${nodeId}`);
      setEdgeFrom(null);
      setHighlightedCycle([]);
    }
  }, [edgeMode, edgeFrom, edgeType, nodes, edges, addLog]);

  // Cycle detection using DFS
  const detectCycle = useCallback(() => {
    const adj = new Map<string, string[]>();
    for (const n of nodes) adj.set(n.id, []);
    for (const e of edges) {
      const list = adj.get(e.from);
      if (list) list.push(e.to);
    }

    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = new Map<string, number>();
    const parent = new Map<string, string | null>();
    for (const n of nodes) {
      color.set(n.id, WHITE);
      parent.set(n.id, null);
    }

    let cycle: string[] = [];

    const dfs = (u: string): boolean => {
      color.set(u, GRAY);
      const neighbors = adj.get(u) || [];
      for (const v of neighbors) {
        if (color.get(v) === GRAY) {
          // Found cycle, reconstruct
          const path: string[] = [v, u];
          let curr = u;
          while (curr !== v) {
            const p = parent.get(curr);
            if (!p) break;
            path.push(p);
            curr = p;
          }
          cycle = path.reverse();
          return true;
        }
        if (color.get(v) === WHITE) {
          parent.set(v, u);
          if (dfs(v)) return true;
        }
      }
      color.set(u, BLACK);
      return false;
    };

    for (const n of nodes) {
      if (color.get(n.id) === WHITE) {
        if (dfs(n.id)) break;
      }
    }

    return cycle;
  }, [nodes, edges]);

  const runCycleCheck = useCallback(() => {
    const cycle = detectCycle();
    if (cycle.length > 0) {
      setHighlightedCycle(cycle);
      addLog(`Cycle detected: ${cycle.join(" -> ")}`);
    } else {
      setHighlightedCycle([]);
      addLog("No cycle found - no deadlock");
    }
  }, [detectCycle, addLog]);

  const hasDeadlock = highlightedCycle.length > 0;

  const getEdgePath = useCallback((from: GraphNode, to: GraphNode) => {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist === 0) return { x1: from.x, y1: from.y, x2: to.x, y2: to.y };
    const nx = dx / dist;
    const ny = dy / dist;
    const r1 = from.type === "resource" ? 22 : NODE_RADIUS;
    const r2 = to.type === "resource" ? 22 : NODE_RADIUS;
    return {
      x1: from.x + nx * r1,
      y1: from.y + ny * r1,
      x2: to.x - nx * r2,
      y2: to.y - ny * r2,
    };
  }, []);

  const isEdgeInCycle = useCallback((edge: GraphEdge) => {
    if (highlightedCycle.length < 2) return false;
    for (let i = 0; i < highlightedCycle.length - 1; i++) {
      if (highlightedCycle[i] === edge.from && highlightedCycle[i + 1] === edge.to) return true;
    }
    // Check wrap-around
    if (highlightedCycle[0] === edge.from && highlightedCycle[highlightedCycle.length - 1] === edge.to) return true;
    return false;
  }, [highlightedCycle]);

  const isNodeInCycle = useCallback((nodeId: string) => {
    return highlightedCycle.includes(nodeId);
  }, [highlightedCycle]);

  const handleReset = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setNextPid(1);
    setNextRid(1);
    setEdgeMode(false);
    setEdgeFrom(null);
    setHighlightedCycle([]);
    setLog([]);
  }, []);

  const loadPreset = useCallback(() => {
    handleReset();
    const presetNodes: GraphNode[] = [
      { id: "P1", type: "process", x: 180, y: 120, instances: 1 },
      { id: "P2", type: "process", x: 180, y: 300, instances: 1 },
      { id: "R1", type: "resource", x: 450, y: 120, instances: 1 },
      { id: "R2", type: "resource", x: 450, y: 300, instances: 1 },
    ];
    const presetEdges: GraphEdge[] = [
      { id: "R1->P1", from: "R1", to: "P1", type: "assignment" },
      { id: "P1->R2", from: "P1", to: "R2", type: "request" },
      { id: "R2->P2", from: "R2", to: "P2", type: "assignment" },
      { id: "P2->R1", from: "P2", to: "R1", type: "request" },
    ];
    setNodes(presetNodes);
    setEdges(presetEdges);
    setNextPid(3);
    setNextRid(3);
    setHighlightedCycle([]);
    setLog(["Loaded deadlock preset: P1->R2->P2->R1->P1 cycle"]);
  }, [handleReset]);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Resource Allocation Graph
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Controls Panel */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-3">Add Nodes</h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={addProcess}
                className="w-full px-3 py-2 rounded-lg bg-blue-500 text-white text-sm font-medium hover:bg-blue-600 transition-colors flex items-center justify-center gap-2"
              >
                <Plus className="w-4 h-4" /> Add Process
              </button>
              <button
                onClick={() => addResource(1)}
                className="w-full px-3 py-2 rounded-lg bg-emerald-500 text-white text-sm font-medium hover:bg-emerald-600 transition-colors flex items-center justify-center gap-2"
              >
                <Plus className="w-4 h-4" /> Add Resource (1)
              </button>
              <button
                onClick={() => addResource(2)}
                className="w-full px-3 py-2 rounded-lg bg-teal-500 text-white text-sm font-medium hover:bg-teal-600 transition-colors flex items-center justify-center gap-2"
              >
                <Plus className="w-4 h-4" /> Add Resource (2)
              </button>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-3">Add Edges</h3>
            <div className="flex flex-col gap-2">
              <div className="flex gap-2">
                {(["request", "assignment"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => {
                      setEdgeType(t);
                      setEdgeMode(true);
                      setEdgeFrom(null);
                    }}
                    className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-colors ${
                      edgeMode && edgeType === t
                        ? "bg-violet-500 text-white"
                        : "bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-gray-300 hover:bg-slate-300 dark:hover:bg-gray-600"
                    }`}
                  >
                    {t === "request" ? "P -> R" : "R -> P"}
                  </button>
                ))}
              </div>
              {edgeMode && (
                <p className="text-xs text-violet-600 dark:text-violet-400">
                  {edgeFrom ? `Click target node (${edgeFrom} -> ?)` : "Click source node"}
                </p>
              )}
              {edgeMode && (
                <button
                  onClick={() => { setEdgeMode(false); setEdgeFrom(null); }}
                  className="w-full px-2 py-1 rounded text-xs bg-slate-300 dark:bg-gray-600 text-slate-600 dark:text-gray-300 hover:bg-slate-400 dark:hover:bg-gray-500"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-3">Analysis</h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={runCycleCheck}
                className="w-full px-3 py-2 rounded-lg bg-amber-500 text-white text-sm font-medium hover:bg-amber-600 transition-colors"
              >
                Detect Cycles
              </button>
              <button
                onClick={loadPreset}
                className="w-full px-3 py-2 rounded-lg bg-indigo-500 text-white text-sm font-medium hover:bg-indigo-600 transition-colors"
              >
                Load Deadlock Example
              </button>
              <button
                onClick={handleReset}
                className="w-full px-3 py-2 rounded-lg bg-slate-400 text-white text-sm font-medium hover:bg-slate-500 transition-colors flex items-center justify-center gap-2"
              >
                <RotateCcw className="w-4 h-4" /> Reset
              </button>
            </div>
          </div>

          {/* Status */}
          <AnimatePresence>
            {hasDeadlock && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="p-3 bg-red-100 dark:bg-red-900/30 border-2 border-red-400 dark:border-red-600 rounded-xl"
              >
                <div className="flex items-center gap-2 mb-1">
                  <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" />
                  <span className="font-bold text-red-700 dark:text-red-300 text-sm">DEADLOCK</span>
                </div>
                <p className="text-xs text-red-600 dark:text-red-400">
                  Cycle: {highlightedCycle.join(" -> ")}
                </p>
              </motion.div>
            )}
            {highlightedCycle.length === 0 && edges.length > 0 && log[0]?.includes("No cycle") && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="p-3 bg-green-100 dark:bg-green-900/30 border-2 border-green-400 dark:border-green-600 rounded-xl"
              >
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                  <span className="font-bold text-green-700 dark:text-green-300 text-sm">NO DEADLOCK</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Graph Canvas */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
            <svg
              viewBox={`0 0 ${SVG_W} ${SVG_H}`}
              className="w-full h-auto"
              style={{ minHeight: 300 }}
            >
              <defs>
                <marker
                  id="arrow-request"
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#6366f1" />
                </marker>
                <marker
                  id="arrow-assignment"
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981" />
                </marker>
                <marker
                  id="arrow-red"
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444" />
                </marker>
              </defs>

              {/* Grid */}
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e2e8f0" strokeWidth="0.5" className="dark:hidden" />
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#374151" strokeWidth="0.5" className="hidden dark:block" />
              </pattern>
              <rect width="100%" height="100%" fill="url(#grid)" rx="8" />

              {/* Edges */}
              {edges.map((edge) => {
                const fromNode = nodes.find((n) => n.id === edge.from);
                const toNode = nodes.find((n) => n.id === edge.to);
                if (!fromNode || !toNode) return null;
                const { x1, y1, x2, y2 } = getEdgePath(fromNode, toNode);
                const inCycle = isEdgeInCycle(edge);
                const color = inCycle ? "#ef4444" : edge.type === "request" ? "#6366f1" : "#10b981";
                const markerEnd = inCycle ? "url(#arrow-red)" : edge.type === "request" ? "url(#arrow-request)" : "url(#arrow-assignment)";

                return (
                  <motion.line
                    key={edge.id}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 1 }}
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={color}
                    strokeWidth={inCycle ? 3 : 2}
                    markerEnd={markerEnd}
                    strokeDasharray={inCycle ? "6 3" : undefined}
                    className={inCycle ? "animate-pulse" : ""}
                  />
                );
              })}

              {/* Nodes */}
              {nodes.map((node) => {
                const inCycle = isNodeInCycle(node.id);
                return (
                  <g
                    key={node.id}
                    onClick={() => handleNodeClick(node.id)}
                    className={edgeMode ? "cursor-pointer" : "cursor-default"}
                  >
                    {node.type === "process" ? (
                      <>
                        <motion.circle
                          initial={{ r: 0 }}
                          animate={{ r: NODE_RADIUS }}
                          cx={node.x}
                          cy={node.y}
                          fill={inCycle ? "#fee2e2" : "#dbeafe"}
                          stroke={inCycle ? "#ef4444" : "#3b82f6"}
                          strokeWidth={inCycle ? 3 : 2}
                          className={`dark:fill-blue-900 ${inCycle ? "animate-pulse" : ""}`}
                        />
                        <text
                          x={node.x}
                          y={node.y + 1}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          className="text-sm font-bold fill-blue-800 dark:fill-blue-200 pointer-events-none"
                        >
                          {node.id}
                        </text>
                      </>
                    ) : (
                      <>
                        <motion.rect
                          initial={{ width: 0, height: 0 }}
                          animate={{ width: 44, height: 44 }}
                          x={node.x - 22}
                          y={node.y - 22}
                          rx={6}
                          fill={inCycle ? "#fee2e2" : "#d1fae5"}
                          stroke={inCycle ? "#ef4444" : "#10b981"}
                          strokeWidth={inCycle ? 3 : 2}
                          className={inCycle ? "animate-pulse" : ""}
                        />
                        {node.instances > 1 && (
                          <>
                            <motion.rect
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 0.5 }}
                              x={node.x - 18}
                              y={node.y - 18}
                              width={44}
                              height={44}
                              rx={6}
                              fill="none"
                              stroke={inCycle ? "#ef4444" : "#10b981"}
                              strokeWidth={1.5}
                              strokeDasharray="4 2"
                            />
                          </>
                        )}
                        <text
                          x={node.x}
                          y={node.y - 3}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          className="text-sm font-bold fill-emerald-800 dark:fill-emerald-200 pointer-events-none"
                        >
                          {node.id}
                        </text>
                        {node.instances > 1 && (
                          <text
                            x={node.x}
                            y={node.y + 12}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="text-[10px] fill-emerald-600 dark:fill-emerald-400 pointer-events-none"
                          >
                            x{node.instances}
                          </text>
                        )}
                      </>
                    )}
                    {/* Remove button */}
                    {!edgeMode && (
                      <g
                        onClick={(e) => { e.stopPropagation(); removeNode(node.id); }}
                        className="cursor-pointer opacity-0 hover:opacity-100 transition-opacity"
                      >
                        <circle cx={node.x + (node.type === "process" ? 22 : 26)} cy={node.y - (node.type === "process" ? 22 : 22)} r={8} fill="#ef4444" />
                        <text
                          x={node.x + (node.type === "process" ? 22 : 26)}
                          y={node.y - (node.type === "process" ? 22 : 22)}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          className="text-[10px] fill-white pointer-events-none font-bold"
                        >
                          x
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}

              {nodes.length === 0 && (
                <text x={SVG_W / 2} y={SVG_H / 2} textAnchor="middle" className="text-sm fill-slate-400 dark:fill-gray-500">
                  Add processes and resources to begin
                </text>
              )}
            </svg>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap items-center gap-4 mt-3 justify-center text-xs text-slate-600 dark:text-gray-400">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-blue-400 inline-block" /> Process
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-emerald-400 inline-block" /> Resource
            </span>
            <span className="flex items-center gap-1">
              <span className="w-6 h-0.5 bg-indigo-500 inline-block" /> Request
            </span>
            <span className="flex items-center gap-1">
              <span className="w-6 h-0.5 bg-emerald-500 inline-block" /> Assignment
            </span>
            <span className="flex items-center gap-1">
              <span className="w-6 h-0.5 bg-red-500 inline-block border-dashed" /> Cycle
            </span>
          </div>
        </div>

        {/* Log Panel */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-sm font-semibold text-slate-700 dark:text-gray-300 mb-3 flex items-center gap-2">
            <ArrowRight className="w-4 h-4" /> Activity Log
          </h3>
          <div className="space-y-1.5 max-h-96 overflow-y-auto">
            {log.length === 0 && (
              <p className="text-xs text-slate-400 dark:text-gray-500 italic">Add nodes and edges to begin...</p>
            )}
            {log.map((entry, i) => (
              <motion.div
                key={`${entry}-${i}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className={`text-xs p-2 rounded ${
                  i === 0
                    ? "bg-violet-50 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 font-medium"
                    : "text-slate-500 dark:text-gray-400"
                }`}
              >
                {entry}
              </motion.div>
            ))}
          </div>

          {/* Stats */}
          <div className="mt-4 p-3 bg-slate-50 dark:bg-gray-900 rounded-lg">
            <h4 className="text-xs font-semibold text-slate-600 dark:text-gray-400 mb-2">Graph Stats</h4>
            <div className="grid grid-cols-2 gap-2 text-xs text-slate-600 dark:text-gray-400">
              <div>Processes: <span className="font-bold text-blue-600 dark:text-blue-400">{nodes.filter((n) => n.type === "process").length}</span></div>
              <div>Resources: <span className="font-bold text-emerald-600 dark:text-emerald-400">{nodes.filter((n) => n.type === "resource").length}</span></div>
              <div>Requests: <span className="font-bold text-indigo-600 dark:text-indigo-400">{edges.filter((e) => e.type === "request").length}</span></div>
              <div>Assignments: <span className="font-bold text-emerald-600 dark:text-emerald-400">{edges.filter((e) => e.type === "assignment").length}</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
