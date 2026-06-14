"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { HardDrive, RotateCcw, Layers } from "lucide-react";

export function DiskStructureViz() {
  const [hoveredPlatter, setHoveredPlatter] = useState<number | null>(null);
  const [hoveredTrack, setHoveredTrack] = useState<number | null>(null);
  const platters = 3;
  const tracks = 6;
  const sectors = 8;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <HardDrive className="w-5 h-5 text-amber-500" />
        磁盘结构可视化
      </h3>
      <div className="flex flex-col items-center gap-6">
        <div className="relative w-64 h-64">
          {Array.from({ length: tracks }).map((_, t) => {
            const radius = 30 + t * 20;
            return (
              <motion.div key={t} whileHover={{ scale: 1.05 }}
                onMouseEnter={() => setHoveredTrack(t)}
                onMouseLeave={() => setHoveredTrack(null)}
                className={`absolute rounded-full border-2 ${hoveredTrack === t ? "border-amber-500" : "border-border-subtle"}`}
                style={{
                  width: radius * 2, height: radius * 2,
                  left: 128 - radius, top: 128 - radius,
                }}>
                {Array.from({ length: sectors }).map((_, s) => {
                  const angle = (s / sectors) * 360;
                  const r = radius;
                  const x = 128 + r * Math.cos((angle * Math.PI) / 180) - 128;
                  const y = 128 + r * Math.sin((angle * Math.PI) / 180) - 128;
                  return (
                    <div key={s} className="absolute w-2 h-2 bg-amber-500/30 rounded-full"
                      style={{ left: radius + x - 4, top: radius + y - 4 }} />
                  );
                })}
              </motion.div>
            );
          })}
          <div className="absolute w-4 h-4 bg-amber-500 rounded-full" style={{ left: 126, top: 126 }} />
          <div className="absolute w-1 h-24 bg-amber-500 origin-bottom"
            style={{ left: 127, top: 128, transform: "rotate(45deg)" }} />
        </div>
        <div className="flex gap-4">
          {Array.from({ length: platters }).map((_, i) => (
            <motion.div key={i} whileHover={{ y: -5 }}
              onMouseEnter={() => setHoveredPlatter(i)}
              onMouseLeave={() => setHoveredPlatter(null)}
              className={`w-24 h-8 rounded-lg border-2 flex items-center justify-center text-xs ${hoveredPlatter === i ? "border-amber-500 bg-amber-500/20" : "border-border-subtle bg-bg-subtle"}`}>
              盘片 {i}
            </motion.div>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-4 gap-3 mt-6 text-center text-xs">
        <div className="bg-bg-subtle rounded p-3">
          <div className="text-amber-500 font-bold mb-1">盘片 (Platter)</div>
          <div className="text-text-secondary">双面可读写的圆形盘片</div>
        </div>
        <div className="bg-bg-subtle rounded p-3">
          <div className="text-amber-500 font-bold mb-1">磁道 (Track)</div>
          <div className="text-text-secondary">盘片上的同心圆环</div>
        </div>
        <div className="bg-bg-subtle rounded p-3">
          <div className="text-amber-500 font-bold mb-1">扇区 (Sector)</div>
          <div className="text-text-secondary">磁道上的弧形区域，最小读写单位</div>
        </div>
        <div className="bg-bg-subtle rounded p-3">
          <div className="text-amber-500 font-bold mb-1">磁头 (Head)</div>
          <div className="text-text-secondary">读写数据的电磁装置</div>
        </div>
      </div>
    </div>
  );
}
