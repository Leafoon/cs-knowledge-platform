"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Database, Shield, Zap, ArrowRight } from "lucide-react";

type RAID = "RAID0" | "RAID1" | "RAID5" | "RAID6" | "RAID10";

interface RAIDInfo {
  name: string;
  desc: string;
  minDisks: number;
  capacity: string;
  fault: string;
  read: number;
  write: number;
}

const RAID_DATA: Record<RAID, RAIDInfo> = {
  RAID0: { name: "RAID 0", desc: "条带化，无冗余", minDisks: 2, capacity: "N×D", fault: "0盘", read: 5, write: 5 },
  RAID1: { name: "RAID 1", desc: "镜像，完全备份", minDisks: 2, capacity: "1×D", fault: "N-1盘", read: 4, write: 2 },
  RAID5: { name: "RAID 5", desc: "分布式奇偶校验", minDisks: 3, capacity: "(N-1)×D", fault: "1盘", read: 4, write: 3 },
  RAID6: { name: "RAID 6", desc: "双重分布式校验", minDisks: 4, capacity: "(N-2)×D", fault: "2盘", read: 4, write: 2 },
  RAID10: { name: "RAID 10", desc: "镜像+条带化", minDisks: 4, capacity: "(N/2)×D", fault: "每组1盘", read: 5, write: 4 },
};

function getBlockType(raid: RAID, disk: number, block: number, total: number): string {
  switch (raid) {
    case "RAID0": return "data";
    case "RAID1": return "mirror";
    case "RAID5": {
      const p = (total - 1 - block % total + total) % total;
      return disk === p ? "parity" : "data";
    }
    case "RAID6": {
      const p1 = (total - 1 - block % total) % total;
      const p2 = (total - 2 - block % total) % total;
      if (disk === p1 || disk === (p2 + total) % total) return "parity";
      return "data";
    }
    case "RAID10": return disk % 2 === 0 ? "data" : "mirror";
  }
}

const BLOCK_COLORS: Record<string, string> = {
  data: "bg-blue-500",
  mirror: "bg-green-500",
  parity: "bg-amber-500",
};

export function RAIDComparisonViz() {
  const [raid, setRaid] = useState<RAID>("RAID0");
  const disks = raid === "RAID10" ? 4 : 4;
  const info = RAID_DATA[raid];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Database className="w-5 h-5 text-violet-500" />
        RAID级别对比
      </h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {(Object.keys(RAID_DATA) as RAID[]).map(r => (
          <button key={r} onClick={() => setRaid(r)}
            className={`px-3 py-1.5 rounded text-sm ${raid === r ? "bg-violet-500 text-white" : "bg-bg-subtle"}`}>
            {RAID_DATA[r].name}
          </button>
        ))}
      </div>
      <p className="text-sm text-text-secondary mb-4">{info.desc}</p>
      <div className="mb-4">
        <div className="text-xs text-text-secondary mb-2">数据分布示意</div>
        <div className="flex gap-1">
          {Array.from({ length: disks }).map((_, d) => (
            <div key={d} className="flex-1">
              <div className="text-xs text-center mb-1 text-text-secondary">磁盘{d}</div>
              {Array.from({ length: 4 }).map((_, b) => (
                <motion.div key={b} whileHover={{ scale: 1.05 }}
                  className={`h-8 mb-0.5 rounded ${BLOCK_COLORS[getBlockType(raid, d, b, disks)]} flex items-center justify-center text-white text-xs font-bold`}>
                  {getBlockType(raid, d, b, disks) === "parity" ? "P" : `D${b}`}
                </motion.div>
              ))}
            </div>
          ))}
        </div>
        <div className="flex gap-3 mt-2 text-xs">
          <span className="flex items-center gap-1"><div className="w-3 h-3 bg-blue-500 rounded" /> 数据</span>
          <span className="flex items-center gap-1"><div className="w-3 h-3 bg-green-500 rounded" /> 镜像</span>
          <span className="flex items-center gap-1"><div className="w-3 h-3 bg-amber-500 rounded" /> 校验</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-bg-subtle rounded p-3 text-sm">
          <div className="text-text-secondary text-xs">最少磁盘</div>
          <div className="font-bold">{info.minDisks}</div>
        </div>
        <div className="bg-bg-subtle rounded p-3 text-sm">
          <div className="text-text-secondary text-xs">有效容量</div>
          <div className="font-bold">{info.capacity}</div>
        </div>
        <div className="bg-bg-subtle rounded p-3 text-sm">
          <div className="text-text-secondary text-xs">容错能力</div>
          <div className="font-bold text-amber-500">{info.fault}</div>
        </div>
        <div className="bg-bg-subtle rounded p-3 text-sm">
          <div className="text-text-secondary text-xs">读/写性能</div>
          <div className="flex gap-1 mt-1">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className={`w-4 h-4 rounded ${i < info.read ? "bg-green-500" : "bg-bg-elevated"}`} />
            ))}
            <span className="text-text-secondary mx-1">/</span>
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className={`w-4 h-4 rounded ${i < info.write ? "bg-blue-500" : "bg-bg-elevated"}`} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
