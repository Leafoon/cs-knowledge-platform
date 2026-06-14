"use client";

import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Flame, ZoomIn, ZoomOut, RotateCcw, Info } from "lucide-react";

interface Frame {
  name: string;
  value: number;
  depth: number;
  color: string;
  children?: Frame[];
}

const COLORS = [
  "#ef4444",
  "#f97316",
  "#f59e0b",
  "#eab308",
  "#84cc16",
  "#22c55e",
  "#14b8a6",
  "#06b6d4",
  "#3b82f6",
  "#6366f1",
  "#8b5cf6",
  "#a855f7",
  "#ec4899",
];

const buildFlameData = (): Frame => ({
  name: "main",
  value: 100,
  depth: 0,
  color: COLORS[0],
  children: [
    {
      name: "process_request",
      value: 65,
      depth: 1,
      color: COLORS[1],
      children: [
        {
          name: "parse_input",
          value: 20,
          depth: 2,
          color: COLORS[2],
          children: [
            { name: "tokenize", value: 12, depth: 3, color: COLORS[3] },
            { name: "validate", value: 8, depth: 3, color: COLORS[4] },
          ],
        },
        {
          name: "compute",
          value: 30,
          depth: 2,
          color: COLORS[5],
          children: [
            {
              name: "matrix_multiply",
              value: 22,
              depth: 3,
              color: COLORS[6],
              children: [
                { name: "dot_product", value: 15, depth: 4, color: COLORS[7] },
                { name: "cache_miss_handler", value: 7, depth: 4, color: COLORS[8] },
              ],
            },
            { name: "hash_lookup", value: 8, depth: 3, color: COLORS[9] },
          ],
        },
        {
          name: "serialize_output",
          value: 15,
          depth: 2,
          color: COLORS[10],
          children: [
            { name: "json_encode", value: 10, depth: 3, color: COLORS[11] },
            { name: "compress", value: 5, depth: 3, color: COLORS[12] },
          ],
        },
      ],
    },
    {
      name: "handle_io",
      value: 35,
      depth: 1,
      color: COLORS[4],
      children: [
        {
          name: "read_from_disk",
          value: 20,
          depth: 2,
          color: COLORS[6],
          children: [
            { name: "file_read", value: 12, depth: 3, color: COLORS[8] },
            { name: "page_fault", value: 8, depth: 3, color: COLORS[10] },
          ],
        },
        {
          name: "write_to_disk",
          value: 15,
          depth: 2,
          color: COLORS[9],
          children: [
            { name: "file_write", value: 10, depth: 3, color: COLORS[11] },
            { name: "fsync", value: 5, depth: 3, color: COLORS[1] },
          ],
        },
      ],
    },
  ],
});

function collectFrames(frame: Frame, startX: number, width: number): Array<Frame & { x: number; w: number }> {
  const result: Array<Frame & { x: number; w: number }> = [];
  const childFrames = frame.children || [];

  let cx = startX;
  const childrenTotal = childFrames.reduce((s, c) => s + c.value, 0);
  const selfWidth = Math.max(0, width - childrenTotal * (width / frame.value));

  for (const child of childFrames) {
    const cw = (child.value / frame.value) * width;
    result.push({ ...child, x: cx, w: cw });
    result.push(...collectFrames(child, cx, cw));
    cx += cw;
  }

  return result;
}

export default function FlameGraphInteractive() {
  const root = useMemo(() => buildFlameData(), []);
  const allFrames = useMemo(() => {
    const frames = collectFrames(root, 0, 100);
    frames.unshift({ ...root, x: 0, w: 100 });
    return frames;
  }, [root]);

  const maxDepth = useMemo(
    () => Math.max(...allFrames.map((f) => f.depth)),
    [allFrames]
  );

  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [zoomStack, setZoomStack] = useState<Frame[]>([root]);

  const currentRoot = zoomStack[zoomStack.length - 1];

  const visibleFrames = useMemo(() => {
    const frames = collectFrames(currentRoot, 0, 100);
    frames.unshift({ ...currentRoot, x: 0, w: 100 });
    return frames;
  }, [currentRoot]);

  const visibleMaxDepth = useMemo(
    () => Math.max(...visibleFrames.map((f) => f.depth)),
    [visibleFrames]
  );

  const rowHeight = 28;
  const svgHeight = (visibleMaxDepth + 1) * rowHeight + 40;
  const svgWidth = 900;

  const zoomInto = (name: string) => {
    const find = (f: Frame): Frame | null => {
      if (f.name === name) return f;
      for (const c of f.children || []) {
        const r = find(c);
        if (r) return r;
      }
      return null;
    };
    const target = find(root);
    if (target && target.children && target.children.length > 0) {
      setZoomStack([...zoomStack, target]);
      setSelected(null);
    }
  };

  const zoomOut = () => {
    if (zoomStack.length > 1) {
      setZoomStack(zoomStack.slice(0, -1));
      setSelected(null);
    }
  };

  const resetZoom = () => {
    setZoomStack([root]);
    setSelected(null);
  };

  const selectedFrame = visibleFrames.find((f) => f.name === selected);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-slate-900 dark:to-slate-800 rounded-xl shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
          <Flame className="w-7 h-7 text-orange-500" />
          交互式火焰图
        </h3>
        <div className="flex gap-2">
          <button
            onClick={zoomOut}
            disabled={zoomStack.length <= 1}
            className="px-3 py-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg text-sm font-semibold flex items-center gap-1 disabled:opacity-40"
          >
            <ZoomOut className="w-4 h-4" /> 缩小
          </button>
          <button
            onClick={resetZoom}
            className="px-3 py-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg text-sm font-semibold flex items-center gap-1"
          >
            <RotateCcw className="w-4 h-4" /> 重置
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2 mb-3 text-xs text-slate-500 dark:text-slate-400">
        <Info className="w-4 h-4" />
        <span>
          悬停查看详情，点击可聚焦（Zoom In）。X 轴宽度 = 采样占比，Y 轴 = 调用栈深度。
        </span>
      </div>

      {zoomStack.length > 1 && (
        <div className="mb-3 flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
          <span>路径:</span>
          {zoomStack.map((f, i) => (
            <React.Fragment key={i}>
              {i > 0 && <span>/</span>}
              <button
                onClick={() => setZoomStack(zoomStack.slice(0, i + 1))}
                className="text-indigo-600 dark:text-indigo-400 hover:underline font-mono"
              >
                {f.name}
              </button>
            </React.Fragment>
          ))}
        </div>
      )}

      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-inner overflow-x-auto">
        <svg
          width={svgWidth}
          height={svgHeight}
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full"
        >
          {visibleFrames.map((frame, i) => {
            const x = (frame.x / 100) * svgWidth;
            const w = Math.max((frame.w / 100) * svgWidth, 2);
            const y = frame.depth * rowHeight + 20;
            const isHovered = hovered === `${frame.name}-${i}`;
            const isSelected = selected === frame.name;
            const opacity = hovered
              ? isHovered
                ? 1
                : 0.5
              : 1;

            return (
              <g key={`${frame.name}-${i}`}>
                <motion.rect
                  x={x}
                  y={y}
                  width={w}
                  height={rowHeight - 2}
                  rx={2}
                  fill={frame.color}
                  opacity={opacity}
                  stroke={isSelected ? "#fff" : "none"}
                  strokeWidth={isSelected ? 2 : 0}
                  initial={{ opacity: 0, scaleY: 0 }}
                  animate={{ opacity, scaleY: 1 }}
                  transition={{ delay: i * 0.01, duration: 0.3 }}
                  onMouseEnter={() => setHovered(`${frame.name}-${i}`)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => {
                    setSelected(frame.name);
                    zoomInto(frame.name);
                  }}
                  className="cursor-pointer"
                  style={{ transformOrigin: `${x + w / 2}px ${y + rowHeight / 2}px` }}
                />
                {w > 40 && (
                  <text
                    x={x + 4}
                    y={y + rowHeight / 2 + 1}
                    fill="#fff"
                    fontSize="11"
                    fontFamily="monospace"
                    dominantBaseline="middle"
                    className="pointer-events-none select-none"
                  >
                    {frame.name.length > w / 7
                      ? frame.name.slice(0, Math.floor(w / 7)) + "..."
                      : frame.name}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <AnimatePresence>
        {hovered && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-4 p-4 bg-white dark:bg-slate-800 rounded-lg shadow-md"
          >
            {(() => {
              const idx = parseInt(hovered.split("-").pop() || "0");
              const frame = visibleFrames[idx];
              if (!frame) return null;
              return (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                  <div>
                    <span className="text-slate-500 dark:text-slate-400 text-xs">函数</span>
                    <p className="font-mono font-bold text-slate-800 dark:text-slate-100">
                      {frame.name}
                    </p>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400 text-xs">采样占比</span>
                    <p className="font-bold text-orange-600">{frame.value}%</p>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400 text-xs">调用深度</span>
                    <p className="font-bold text-blue-600">{frame.depth}</p>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400 text-xs">子函数数</span>
                    <p className="font-bold text-purple-600">
                      {frame.children?.length ?? 0}
                    </p>
                  </div>
                </div>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>

      {selectedFrame && (
        <div className="mt-3 p-3 bg-indigo-50 dark:bg-indigo-900/30 rounded-lg text-sm">
          <span className="font-semibold text-indigo-700 dark:text-indigo-300">
            已聚焦: {selectedFrame.name}
          </span>
          <span className="text-slate-500 dark:text-slate-400 ml-2">
            (占 {selectedFrame.value}% 采样)
          </span>
        </div>
      )}

      <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-500 dark:text-slate-400">
        <span>深度层:</span>
        {Array.from({ length: visibleMaxDepth + 1 }, (_, i) => (
          <span
            key={i}
            className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-700 font-mono"
          >
            D{i}
          </span>
        ))}
      </div>
    </div>
  );
}
