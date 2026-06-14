"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Scale, Table, BarChart3 } from "lucide-react";

interface ComparisonRow {
  aspect: string;
  segmentation: string;
  paging: string;
  segAdvantage: boolean;
  pageAdvantage: boolean;
}

const rows: ComparisonRow[] = [
  { aspect: "Memory Unit", segmentation: "Variable-size segment (logical unit)", paging: "Fixed-size page (physical unit)", segAdvantage: false, pageAdvantage: false },
  { aspect: "Address", segmentation: "2D: (segment, offset)", paging: "1D: virtual address", segAdvantage: true, pageAdvantage: false },
  { aspect: "Fragmentation", segmentation: "External fragmentation", paging: "Internal fragmentation (last page)", segAdvantage: false, pageAdvantage: true },
  { aspect: "Sharing", segmentation: "Natural (code segment shared)", paging: "Requires shared pages", segAdvantage: true, pageAdvantage: false },
  { aspect: "Protection", segmentation: "Per-segment (base+limit)", paging: "Per-page (PTE flags)", segAdvantage: false, pageAdvantage: false },
  { aspect: "Complexity", segmentation: "Simpler hardware", paging: "Complex page table management", segAdvantage: true, pageAdvantage: false },
  { aspect: "Memory Overhead", segmentation: "Small segment table", paging: "Page tables can be large", segAdvantage: true, pageAdvantage: false },
  { aspect: "Hardware Support", segmentation: "Segment registers", paging: "MMU + TLB", segAdvantage: false, pageAdvantage: true },
  { aspect: "Virtual Memory", segmentation: "Swapping entire segments", paging: "Fine-grained page replacement", segAdvantage: false, pageAdvantage: true },
  { aspect: "Modern Usage", segmentation: "Mostly abandoned", paging: "Dominant (x86-64, ARM)", segAdvantage: false, pageAdvantage: true },
];

const chartData = [
  { label: "Hardware Complexity", seg: 3, page: 7 },
  { label: "External Frag", seg: 8, page: 1 },
  { label: "Internal Frag", seg: 1, page: 5 },
  { label: "Ease of Sharing", seg: 7, page: 3 },
  { label: "VM Efficiency", seg: 4, page: 9 },
  { label: "Protection Granularity", seg: 6, page: 8 },
];

export default function SegmentationVsPaging() {
  const [view, setView] = useState<"table" | "chart">("table");

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-sky-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <div className="flex items-center justify-center gap-3 mb-6">
        <Scale className="w-7 h-7 text-sky-600 dark:text-sky-400" />
        <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100">
          Segmentation vs Paging
        </h2>
      </div>

      {/* View toggle */}
      <div className="flex justify-center mb-6">
        <div className="flex bg-slate-200 dark:bg-gray-700 rounded-lg p-1">
          <button onClick={() => setView("table")}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
              view === "table" ? "bg-sky-600 text-white shadow" : "text-slate-600 dark:text-slate-300"
            }`}>
            <Table className="w-4 h-4" /> Table View
          </button>
          <button onClick={() => setView("chart")}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
              view === "chart" ? "bg-sky-600 text-white shadow" : "text-slate-600 dark:text-slate-300"
            }`}>
            <BarChart3 className="w-4 h-4" /> Chart View
          </button>
        </div>
      </div>

      {view === "table" ? (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-md border border-slate-200 dark:border-gray-700 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-100 dark:bg-gray-700">
                <th className="px-4 py-3 text-left text-slate-700 dark:text-slate-300 font-semibold">Aspect</th>
                <th className="px-4 py-3 text-left text-slate-700 dark:text-slate-300 font-semibold">
                  <span className="inline-flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-rose-500" /> Segmentation
                  </span>
                </th>
                <th className="px-4 py-3 text-left text-slate-700 dark:text-slate-300 font-semibold">
                  <span className="inline-flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-blue-500" /> Paging
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <motion.tr key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="hover:bg-slate-50 dark:hover:bg-gray-700 border-t border-slate-100 dark:border-gray-700">
                  <td className="px-4 py-3 font-medium text-slate-800 dark:text-slate-200">{row.aspect}</td>
                  <td className={`px-4 py-3 text-slate-600 dark:text-slate-400 ${row.segAdvantage ? "bg-rose-50 dark:bg-rose-900/10" : ""}`}>
                    {row.segmentation}
                    {row.segAdvantage && <span className="ml-2 text-rose-500 text-xs">+1</span>}
                  </td>
                  <td className={`px-4 py-3 text-slate-600 dark:text-slate-400 ${row.pageAdvantage ? "bg-blue-50 dark:bg-blue-900/10" : ""}`}>
                    {row.paging}
                    {row.pageAdvantage && <span className="ml-2 text-blue-500 text-xs">+1</span>}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      ) : (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-md border border-slate-200 dark:border-gray-700">
          <div className="space-y-4">
            {chartData.map((d, i) => {
              const maxVal = 10;
              return (
                <motion.div key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}>
                  <div className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">{d.label}</div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-xs text-slate-500">Segmentation</span>
                      <div className="flex-1 h-6 bg-slate-100 dark:bg-gray-700 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-rose-500 rounded-full flex items-center justify-end pr-2"
                          initial={{ width: 0 }}
                          animate={{ width: `${(d.seg / maxVal) * 100}%` }}
                          transition={{ duration: 0.8, delay: i * 0.15 }}
                        >
                          <span className="text-white text-xs font-bold">{d.seg}</span>
                        </motion.div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-xs text-slate-500">Paging</span>
                      <div className="flex-1 h-6 bg-slate-100 dark:bg-gray-700 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-blue-500 rounded-full flex items-center justify-end pr-2"
                          initial={{ width: 0 }}
                          animate={{ width: `${(d.page / maxVal) * 100}%` }}
                          transition={{ duration: 0.8, delay: i * 0.15 + 0.1 }}
                        >
                          <span className="text-white text-xs font-bold">{d.page}</span>
                        </motion.div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
          <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
            Scores (1-10) indicate relative advantage. Higher is better for the system.
          </p>
        </motion.div>
      )}

      {/* Summary */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="p-4 bg-rose-50 dark:bg-rose-900/20 rounded-lg border border-rose-200 dark:border-rose-800">
          <h4 className="text-sm font-semibold text-rose-700 dark:text-rose-300 mb-2">Segmentation Wins On</h4>
          <ul className="text-xs text-rose-600 dark:text-rose-400 space-y-1 list-disc list-inside">
            <li>Natural logical grouping (code/data/stack)</li>
            <li>Built-in sharing of code segments</li>
            <li>Simpler hardware requirements</li>
            <li>Smaller metadata overhead</li>
          </ul>
        </div>
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h4 className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-2">Paging Wins On</h4>
          <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1 list-disc list-inside">
            <li>No external fragmentation</li>
            <li>Fine-grained virtual memory</li>
            <li>Efficient page replacement</li>
            <li>Universal adoption in modern systems</li>
          </ul>
        </div>
      </div>

      <div className="mt-4 p-4 bg-sky-50 dark:bg-sky-900/20 rounded-lg border border-sky-200 dark:border-sky-800">
        <p className="text-sm text-sky-800 dark:text-sky-200">
          <strong>Modern Systems:</strong> Most modern architectures (x86-64, ARMv8) use pure paging.
          x86-64 has largely abandoned segmentation (segments are set to flat base=0).
          Some systems (e.g., IBM System/38) combine both: segments for logical grouping + pages for fine-grained mapping.
        </p>
      </div>
    </div>
  );
}
