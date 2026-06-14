"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Maximize2 } from "lucide-react";

interface PageSizeInfo {
  label: string;
  size: string;
  sizeBytes: number;
  offsetBits: number;
  color: string;
  colorDark: string;
}

const pageSizes: PageSizeInfo[] = [
  { label: "4 KB", size: "4KB", sizeBytes: 4 * 1024, offsetBits: 12, color: "bg-blue-500", colorDark: "bg-blue-600" },
  { label: "2 MB", size: "2MB", sizeBytes: 2 * 1024 * 1024, offsetBits: 21, color: "bg-green-500", colorDark: "bg-green-600" },
  { label: "1 GB", size: "1GB", sizeBytes: 1 * 1024 * 1024 * 1024, offsetBits: 30, color: "bg-purple-500", colorDark: "bg-purple-600" },
];

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(0)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

export default function PageSizeComparison() {
  const [vaBits, setVaBits] = useState(48);
  const [pteSize, setPTESize] = useState(8); // bytes
  const [tlbEntries, setTLBEntries] = useState(1024);

  const vaSpaceBytes = Math.pow(2, vaBits);

  const computeMetrics = (ps: PageSizeInfo) => {
    const numPages = vaSpaceBytes / ps.sizeBytes;
    const pageTableEntries = numPages;
    const linearPTSize = pageTableEntries * pteSize;
    const tlbReach = (tlbEntries * ps.sizeBytes);
    const avgInternalFrag = ps.sizeBytes / 2; // average half a page wasted
    return { numPages, pageTableEntries, linearPTSize, tlbReach, avgInternalFrag };
  };

  const metrics = pageSizes.map(computeMetrics);
  const maxPTSize = Math.max(...metrics.map((m) => m.linearPTSize));

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <div className="flex items-center justify-center gap-3 mb-6">
        <Maximize2 className="w-7 h-7 text-emerald-600 dark:text-emerald-400" />
        <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100">
          Page Size Comparison
        </h2>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-6 mb-6 items-end justify-center">
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            VA bits: {vaBits} ({formatSize(vaSpaceBytes)} address space)
          </label>
          <input type="range" min={32} max={52} value={vaBits} onChange={(e) => setVaBits(Number(e.target.value))}
            className="w-48" />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            PTE size: {pteSize} bytes
          </label>
          <input type="range" min={4} max={16} step={4} value={pteSize} onChange={(e) => setPTESize(Number(e.target.value))}
            className="w-40" />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            TLB entries: {tlbEntries}
          </label>
          <input type="range" min={64} max={4096} step={64} value={tlbEntries} onChange={(e) => setTLBEntries(Number(e.target.value))}
            className="w-40" />
        </div>
      </div>

      {/* Three columns */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        {pageSizes.map((ps, i) => {
          const m = metrics[i];
          return (
            <motion.div
              key={ps.size}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.15 }}
              className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md border border-slate-200 dark:border-gray-700"
            >
              <div className="flex items-center gap-2 mb-4">
                <div className={`w-4 h-4 rounded ${ps.color}`} />
                <h3 className="text-lg font-bold text-slate-800 dark:text-gray-100">{ps.label}</h3>
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 font-mono mb-3">
                Offset: {ps.offsetBits} bits
              </div>
              <div className="space-y-3">
                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">Page Table Entries</div>
                  <div className="text-lg font-bold text-slate-800 dark:text-gray-100 font-mono">
                    {m.pageTableEntries >= 1e9 ? `${(m.pageTableEntries / 1e9).toFixed(1)}B` : m.pageTableEntries >= 1e6 ? `${(m.pageTableEntries / 1e6).toFixed(1)}M` : `${(m.pageTableEntries / 1e3).toFixed(0)}K`}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">Linear PT Size</div>
                  <div className="text-lg font-bold text-slate-800 dark:text-gray-100 font-mono">
                    {formatSize(m.linearPTSize)}
                  </div>
                  {/* Bar */}
                  <div className="mt-1 h-3 bg-slate-100 dark:bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className={`h-full rounded-full ${ps.color}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${(m.linearPTSize / maxPTSize) * 100}%` }}
                      transition={{ duration: 0.8, delay: i * 0.2 }}
                    />
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">TLB Reach</div>
                  <div className="text-lg font-bold text-emerald-700 dark:text-emerald-300 font-mono">
                    {formatSize(m.tlbReach)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">Avg Internal Fragmentation</div>
                  <div className="text-lg font-bold text-amber-700 dark:text-amber-300 font-mono">
                    {formatSize(m.avgInternalFrag)}
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Bar chart comparison */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md border border-slate-200 dark:border-gray-700 mb-6">
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-4">Memory Overhead Comparison (Linear Page Table Size)</h4>
        <div className="space-y-3">
          {pageSizes.map((ps, i) => {
            const m = metrics[i];
            return (
              <div key={ps.size} className="flex items-center gap-3">
                <div className="w-16 text-sm font-medium text-slate-700 dark:text-slate-300">{ps.label}</div>
                <div className="flex-1 h-8 bg-slate-100 dark:bg-gray-700 rounded-lg overflow-hidden relative">
                  <motion.div
                    className={`h-full rounded-lg ${ps.color} flex items-center justify-end pr-2`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max((m.linearPTSize / maxPTSize) * 100, 2)}%` }}
                    transition={{ duration: 1, delay: i * 0.2 }}
                  >
                    <span className="text-white text-xs font-mono font-bold drop-shadow">
                      {formatSize(m.linearPTSize)}
                    </span>
                  </motion.div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* TLB Reach comparison */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-md border border-slate-200 dark:border-gray-700 mb-6">
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-4">TLB Coverage Comparison</h4>
        <div className="space-y-3">
          {pageSizes.map((ps, i) => {
            const m = metrics[i];
            const maxReach = Math.max(...metrics.map((mm) => mm.tlbReach));
            return (
              <div key={ps.size} className="flex items-center gap-3">
                <div className="w-16 text-sm font-medium text-slate-700 dark:text-slate-300">{ps.label}</div>
                <div className="flex-1 h-8 bg-slate-100 dark:bg-gray-700 rounded-lg overflow-hidden relative">
                  <motion.div
                    className="h-full rounded-lg bg-emerald-500 flex items-center justify-end pr-2"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max((m.tlbReach / maxReach) * 100, 2)}%` }}
                    transition={{ duration: 1, delay: i * 0.2 }}
                  >
                    <span className="text-white text-xs font-mono font-bold drop-shadow">
                      {formatSize(m.tlbReach)}
                    </span>
                  </motion.div>
                </div>
              </div>
            );
          })}
        </div>
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
          Larger pages mean each TLB entry covers more address space, dramatically improving TLB hit rates.
        </p>
      </div>

      {/* Info */}
      <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-800">
        <p className="text-sm text-emerald-800 dark:text-emerald-200">
          <strong>Trade-off:</strong> Smaller pages (4KB) reduce internal fragmentation but require many more page table entries,
          increasing memory overhead. Larger pages (2MB, 1GB) improve TLB reach and reduce page table size, but waste memory
          due to internal fragmentation. Modern systems use a mix: 4KB for general use, 2MB/1GB huge pages for databases and VMs.
        </p>
      </div>
    </div>
  );
}
