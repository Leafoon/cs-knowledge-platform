"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { HardDrive, Zap } from "lucide-react";

const categories = [
  {
    name: "协议栈",
    sata: ["AHCI 驱动", "AHCI 控制器", "SATA 物理层"],
    nvme: ["NVMe 驱动", "NVMe 控制器", "PCIe 物理层"],
  },
  {
    name: "命令队列",
    sata: { depth: 1, parallel: 1, desc: "单命令队列，深度32" },
    nvme: { depth: 65535, parallel: 64, desc: "最多64K队列，每队列64K命令" },
  },
  {
    name: "性能",
    sata: { seqRead: 560, seqWrite: 530, iops: 100000, latency: 6 },
    nvme: { seqRead: 7000, seqWrite: 5000, iops: 1000000, latency: 0.025 },
  },
  {
    name: "接口",
    sata: { bus: "SATA 3.0", bandwidth: "6 Gbps", connector: "7-pin 数据 + 15-pin 电源" },
    nvme: { bus: "PCIe 3.0/4.0/5.0", bandwidth: "32/64/128 GT/s", connector: "M.2 / U.2 / PCIe" },
  },
];

export function SATAvsNVMe() {
  const [cat, setCat] = useState(0);
  const current = categories[cat];

  const renderMetric = (label: string, sataVal: number, nvmeVal: number, unit: string) => (
    <div className="mb-3">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="flex gap-2 items-center">
        <span className="text-xs w-20 text-right text-orange-300">{sataVal.toLocaleString()} {unit}</span>
        <div className="flex-1 flex gap-1">
          <motion.div
            className="h-5 bg-orange-500/40 rounded"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, (sataVal / Math.max(sataVal, nvmeVal)) * 100)}%` }}
            transition={{ duration: 0.5 }}
          />
          <motion.div
            className="h-5 bg-blue-500/40 rounded"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, (nvmeVal / Math.max(sataVal, nvmeVal)) * 100)}%` }}
            transition={{ duration: 0.5, delay: 0.1 }}
          />
        </div>
        <span className="text-xs w-20 text-blue-300">{nvmeVal.toLocaleString()} {unit}</span>
      </div>
    </div>
  );

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <HardDrive className="w-5 h-5 text-orange-400" />
        <h3 className="text-lg font-semibold">SATA vs NVMe 对比</h3>
      </div>

      <div className="flex gap-4 mb-4">
        <span className="flex items-center gap-1 text-xs"><span className="w-3 h-3 bg-orange-500/60 rounded" />SATA</span>
        <span className="flex items-center gap-1 text-xs"><span className="w-3 h-3 bg-blue-500/60 rounded" />NVMe</span>
      </div>

      <div className="flex gap-2 mb-4">
        {categories.map((c, i) => (
          <button
            key={c.name}
            onClick={() => setCat(i)}
            className={`px-3 py-1 rounded text-xs ${cat === i ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300 hover:bg-gray-600"}`}
          >
            {c.name}
          </button>
        ))}
      </div>

      <motion.div key={cat} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
        {cat === 0 && (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm font-medium text-orange-300 mb-2">SATA 协议栈</div>
              {(current as any).sata.map((l: string, i: number) => (
                <motion.div key={l} initial={{ x: -10, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: i * 0.1 }}
                  className="p-2 mb-1 bg-gray-800/50 rounded text-xs text-gray-300">{l}</motion.div>
              ))}
            </div>
            <div>
              <div className="text-sm font-medium text-blue-300 mb-2">NVMe 协议栈</div>
              {(current as any).nvme.map((l: string, i: number) => (
                <motion.div key={l} initial={{ x: 10, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: i * 0.1 }}
                  className="p-2 mb-1 bg-gray-800/50 rounded text-xs text-gray-300">{l}</motion.div>
              ))}
            </div>
          </div>
        )}
        {cat === 1 && (
          <div>
            {renderMetric("队列数量", (current as any).sata.parallel, (current as any).nvme.parallel, "个")}
            {renderMetric("队列深度", (current as any).sata.depth, (current as any).nvme.depth, "")}
            <p className="text-xs text-gray-400 mt-2">{(current as any).nvme.desc}</p>
          </div>
        )}
        {cat === 2 && (
          <div>
            {renderMetric("顺序读", (current as any).sata.seqRead, (current as any).nvme.seqRead, "MB/s")}
            {renderMetric("顺序写", (current as any).sata.seqWrite, (current as any).nvme.seqWrite, "MB/s")}
            {renderMetric("IOPS", (current as any).sata.iops, (current as any).nvme.iops, "")}
            {renderMetric("延迟", (current as any).sata.latency, (current as any).nvme.latency, "μs")}
          </div>
        )}
        {cat === 3 && (
          <div className="grid grid-cols-2 gap-4">
            {["sata", "nvme"].map((key) => {
              const d = (current as any)[key];
              return (
                <div key={key} className="space-y-2">
                  {Object.entries(d).map(([k, v]) => (
                    <div key={k} className="p-2 bg-gray-800/50 rounded text-xs">
                      <span className="text-gray-400">{k}: </span>
                      <span className="text-gray-200">{String(v)}</span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        )}
      </motion.div>
    </div>
  );
}
