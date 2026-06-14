"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layout, Monitor, Server } from "lucide-react";

interface MemoryRegion {
  name: string;
  color: string;
  darkColor: string;
  startPct: number; // percentage from top (0-100)
  heightPct: number;
  permissions: string;
  growth: string;
  description: string;
  addrRange32: string;
  addrRange64: string;
}

const regions32: MemoryRegion[] = [
  { name: "Kernel Space", color: "bg-gray-700", darkColor: "bg-gray-600", startPct: 0, heightPct: 8, permissions: "Supervisor R/W", growth: "Fixed", description: "Operating system kernel code and data. Not accessible from user mode.", addrRange32: "0xC0000000 - 0xFFFFFFFF", addrRange64: "0xFFFF800000000000 - 0xFFFFFFFFFFFFFFFF" },
  { name: "Stack", color: "bg-blue-500", darkColor: "bg-blue-600", startPct: 10, heightPct: 22, permissions: "User R/W", growth: "Downward", description: "Function call frames, local variables, return addresses. Grows toward lower addresses.", addrRange32: "0xBFFF0000 - 0xBFFFFFFF", addrRange64: "0x00007FFFFFFFE000 - 0x00007FFFFFFFF000" },
  { name: "Memory-Mapped Files / Shared Libraries", color: "bg-teal-500", darkColor: "bg-teal-600", startPct: 34, heightPct: 16, permissions: "User R/W/R-X", growth: "Downward", description: "Shared libraries (.so/.dll), mmap'd files. Dynamically loaded at runtime.", addrRange32: "0x40000000 - 0xBFFEFFFF", addrRange64: "0x00007F0000000000 - 0x00007FFFFFFFDFFF" },
  { name: "Heap", color: "bg-green-500", darkColor: "bg-green-600", startPct: 52, heightPct: 18, permissions: "User R/W", growth: "Upward", description: "Dynamic memory allocation (malloc/new). Managed by the runtime heap allocator.", addrRange32: "0x08048000 - 0x3FFFFFFF", addrRange64: "0x0000555555554000 - 0x00007EFFFFFFFF" },
  { name: "Data Segment (.data/.bss)", color: "bg-amber-500", darkColor: "bg-amber-600", startPct: 72, heightPct: 10, permissions: "User R/W", growth: "Fixed", description: "Global and static variables. .data has initialized values, .bss is zero-initialized.", addrRange32: "0x0804A000 - 0x0806AFFF", addrRange64: "0x0000555555558000 - 0x000055555555AFFF" },
  { name: "Code Segment (.text)", color: "bg-red-500", darkColor: "bg-red-600", startPct: 84, heightPct: 12, permissions: "User R-X", growth: "Fixed", description: "Executable instructions. Read-only and executable to prevent code injection attacks.", addrRange32: "0x08048000 - 0x08049FFF", addrRange64: "0x0000555555554000 - 0x0000555555557FFF" },
];

export default function AddressSpaceLayout() {
  const [is64Bit, setIs64Bit] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState<MemoryRegion | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const regions = regions32;

  const addressLabels32 = ["0xFFFFFFFF", "0xC0000000", "0xBFFF0000", "0x40000000", "0x0804A000", "0x08048000", "0x00000000"];
  const addressLabels64 = ["0xFFFFFFFF_FFFFFFFF", "0xFFFF8000_00000000", "0x00007FFF_FFFFFFFF", "0x00007F00_00000000", "0x00005555_5555A000", "0x00005555_55554000", "0x00000000_00000000"];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <div className="flex items-center justify-center gap-3 mb-6">
        <Layout className="w-7 h-7 text-indigo-600 dark:text-indigo-400" />
        <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100">
          Process Address Space Layout
        </h2>
      </div>

      {/* Toggle */}
      <div className="flex justify-center mb-6">
        <div className="flex bg-slate-200 dark:bg-gray-700 rounded-lg p-1">
          <button
            onClick={() => setIs64Bit(false)}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
              !is64Bit
                ? "bg-indigo-600 text-white shadow"
                : "text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-white"
            }`}
          >
            <Monitor className="w-4 h-4" /> 32-bit
          </button>
          <button
            onClick={() => setIs64Bit(true)}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
              is64Bit
                ? "bg-indigo-600 text-white shadow"
                : "text-slate-600 dark:text-slate-300 hover:text-slate-800 dark:hover:text-white"
            }`}
          >
            <Server className="w-4 h-4" /> 64-bit
          </button>
        </div>
      </div>

      <div className="flex gap-6">
        {/* SVG Layout */}
        <div className="flex-1">
          <svg viewBox="0 0 400 600" className="w-full max-w-md mx-auto" style={{ minHeight: 500 }}>
            {/* Address labels left side */}
            {(is64Bit ? addressLabels64 : addressLabels32).map((label, i) => {
              const yPos = (i / (addressLabels32.length - 1)) * 580 + 10;
              return (
                <text key={i} x="2" y={yPos + 4} className="fill-slate-500 dark:fill-slate-400" fontSize="8" fontFamily="monospace">
                  {label}
                </text>
              );
            })}

            {/* Regions */}
            {regions.map((region, idx) => {
              const y = (region.startPct / 100) * 580 + 10;
              const h = (region.heightPct / 100) * 580;
              const isHovered = hoveredIdx === idx || selectedRegion?.name === region.name;
              return (
                <g key={idx} onClick={() => setSelectedRegion(region)} style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHoveredIdx(idx)} onMouseLeave={() => setHoveredIdx(null)}>
                  <motion.rect
                    x={100} y={y} width={280} height={h - 2} rx={6}
                    className={`${region.color} dark:${region.darkColor}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{
                      opacity: 1,
                      x: 0,
                      scale: isHovered ? 1.02 : 1,
                      filter: isHovered ? "brightness(1.15)" : "brightness(1)",
                    }}
                    transition={{ delay: idx * 0.1, duration: 0.3 }}
                    style={{ fill: isHovered ? undefined : undefined }}
                  />
                  {/* Animated fill */}
                  <motion.rect
                    x={100} y={y} width={280} height={h - 2} rx={6}
                    fill="none"
                    stroke={isHovered ? "white" : "transparent"}
                    strokeWidth={2}
                    animate={{ strokeOpacity: isHovered ? 0.8 : 0 }}
                    transition={{ duration: 0.2 }}
                  />
                  <text x={240} y={y + h / 2} textAnchor="middle" dominantBaseline="middle"
                    className="fill-white font-semibold" fontSize={h > 40 ? 12 : 9}
                    style={{ pointerEvents: "none" }}>
                    {region.name}
                  </text>
                  {/* Growth arrows */}
                  {region.growth === "Upward" && (
                    <motion.g animate={{ y: [0, -4, 0] }} transition={{ repeat: Infinity, duration: 1.5 }}>
                      <path d={`M ${370} ${y + h - 10} L ${370} ${y + 10} L ${375} ${y + 18} M ${370} ${y + 10} L ${365} ${y + 18}`}
                        stroke="white" strokeWidth={2} fill="none" style={{ pointerEvents: "none" }} />
                    </motion.g>
                  )}
                  {region.growth === "Downward" && (
                    <motion.g animate={{ y: [0, 4, 0] }} transition={{ repeat: Infinity, duration: 1.5 }}>
                      <path d={`M ${370} ${y + 10} L ${370} ${y + h - 10} L ${375} ${y + h - 18} M ${370} ${y + h - 10} L ${365} ${y + h - 18}`}
                        stroke="white" strokeWidth={2} fill="none" style={{ pointerEvents: "none" }} />
                    </motion.g>
                  )}
                  {region.growth === "Fixed" && (
                    <line x1={365} y1={y + h / 2 - 8} x2={375} y2={y + h / 2 - 8}
                      stroke="white" strokeWidth={1.5} style={{ pointerEvents: "none" }} opacity={0.6} />
                  )}
                </g>
              );
            })}

            {/* Direction label */}
            <text x="30" y="300" textAnchor="middle" transform="rotate(-90, 30, 300)"
              className="fill-slate-500 dark:fill-slate-400 font-semibold" fontSize="11">
              High Addresses
            </text>
            <text x="30" y="460" textAnchor="middle" transform="rotate(-90, 30, 460)"
              className="fill-slate-500 dark:fill-slate-400 font-semibold" fontSize="11">
              Low Addresses
            </text>
          </svg>
        </div>

        {/* Detail panel */}
        <div className="w-80 flex-shrink-0">
          <AnimatePresence mode="wait">
            {selectedRegion ? (
              <motion.div
                key={selectedRegion.name}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="p-5 bg-white dark:bg-gray-800 rounded-xl shadow-md border border-slate-200 dark:border-gray-700"
              >
                <div className="flex items-center gap-2 mb-4">
                  <div className={`w-4 h-4 rounded ${selectedRegion.color}`} />
                  <h3 className="text-lg font-bold text-slate-800 dark:text-gray-100">
                    {selectedRegion.name}
                  </h3>
                </div>
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-slate-500 dark:text-slate-400 font-medium">Permissions: </span>
                    <span className="text-slate-700 dark:text-slate-200 font-mono bg-slate-100 dark:bg-gray-700 px-2 py-0.5 rounded">
                      {selectedRegion.permissions}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400 font-medium">Growth: </span>
                    <span className="text-slate-700 dark:text-slate-200">{selectedRegion.growth}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 dark:text-slate-400 font-medium">Address Range ({is64Bit ? "64-bit" : "32-bit"}): </span>
                    <div className="font-mono text-xs mt-1 bg-indigo-50 dark:bg-indigo-900/30 p-2 rounded text-indigo-700 dark:text-indigo-300 break-all">
                      {is64Bit ? selectedRegion.addrRange64 : selectedRegion.addrRange32}
                    </div>
                  </div>
                  <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                    {selectedRegion.description}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedRegion(null)}
                  className="mt-4 text-sm text-indigo-600 dark:text-indigo-400 hover:underline"
                >
                  Close
                </button>
              </motion.div>
            ) : (
              <motion.div
                key="hint"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="p-5 bg-white/60 dark:bg-gray-800/60 rounded-xl border border-dashed border-slate-300 dark:border-gray-600 text-center"
              >
                <p className="text-slate-500 dark:text-slate-400 text-sm">
                  Click on a memory region to view its details
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Legend */}
          <div className="mt-4 p-4 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-slate-200 dark:border-gray-700">
            <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">Legend</h4>
            <div className="space-y-1.5">
              {regions.map((r, i) => (
                <div key={i} className="flex items-center gap-2 text-xs">
                  <div className={`w-3 h-3 rounded-sm ${r.color}`} />
                  <span className="text-slate-600 dark:text-slate-400">{r.name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
        <p className="text-sm text-indigo-800 dark:text-indigo-200">
          <strong>Virtual Address Space:</strong> Each process has its own virtual address space, created during process initialization.
          The kernel occupies the top portion (protected by hardware privilege levels), while user-space regions are arranged below.
          Stack and heap grow toward each other, with the gap between them available for memory-mapped files and shared libraries.
          {is64Bit && " In 64-bit mode, only 48 bits of the virtual address are used (canonical addresses), leaving a large gap in the middle for future use."}
        </p>
      </div>
    </div>
  );
}
