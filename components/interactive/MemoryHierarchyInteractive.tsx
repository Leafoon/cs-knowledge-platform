"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const levels = [
  {
    id: "register",
    name: "寄存器",
    nameEn: "Register",
    speed: "< 1ns",
    capacity: "几B ~ 几十B",
    cost: "最高",
    tech: "触发器",
    color: "#ef4444",
    width: "16%",
    details: "位于CPU内部，速度最快，容量最小，由触发器构成，用于暂存指令、数据和地址",
  },
  {
    id: "cache",
    name: "Cache (高速缓存)",
    nameEn: "Cache",
    speed: "1~10ns",
    capacity: "几MB ~ 几十MB",
    cost: "高",
    tech: "SRAM",
    color: "#f59e0b",
    width: "28%",
    details: "位于CPU与主存之间，利用局部性原理，缓存主存中常用数据，由SRAM构成",
  },
  {
    id: "main",
    name: "主存储器",
    nameEn: "Main Memory",
    speed: "10~100ns",
    capacity: "几GB ~ 几十GB",
    cost: "中",
    tech: "DRAM",
    color: "#10b981",
    width: "48%",
    details: "即内存，存放运行中的程序和数据，由DRAM构成，CPU可直接访问",
  },
  {
    id: "disk",
    name: "辅助存储器",
    nameEn: "Secondary Storage",
    speed: "ms级",
    capacity: "几TB",
    cost: "最低",
    tech: "磁盘/SSD",
    color: "#667eea",
    width: "72%",
    details: "外存，容量大、断电不丢失，包括机械硬盘(HDD)和固态硬盘(SSD)，CPU不可直接访问",
  },
];

export function MemoryHierarchyInteractive() {
  const [active, setActive] = useState<string | null>(null);
  const info = levels.find((l) => l.id === active);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        存储器层次结构
      </h3>
      <p className="text-sm text-text-secondary mb-6">
        点击各层查看详细参数 — 金字塔越高，速度越快、容量越小、成本越高
      </p>

      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* pyramid */}
        <div className="flex-1 flex flex-col items-center gap-2">
          {levels.map((lvl, i) => (
            <motion.button
              key={lvl.id}
              onClick={() => setActive(active === lvl.id ? null : lvl.id)}
              className="relative rounded-lg border-2 transition-all text-center py-3 px-4 cursor-pointer"
              style={{
                width: lvl.width,
                minWidth: 160,
                borderColor: active === lvl.id ? lvl.color : lvl.color + "40",
                backgroundColor: active === lvl.id ? lvl.color + "20" : lvl.color + "08",
              }}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.98 }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <div className="font-semibold text-sm text-text-primary">{lvl.name}</div>
              <div className="text-xs text-text-secondary">{lvl.speed} | {lvl.capacity}</div>
              {/* arrows */}
              {i < levels.length - 1 && (
                <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-text-secondary text-lg">
                  ↕
                </div>
              )}
            </motion.button>
          ))}

          {/* axis labels */}
          <div className="flex justify-between w-full max-w-[72%] mt-8 text-xs text-text-secondary">
            <div className="text-center">
              <div className="font-semibold text-red-500">速度 ↑</div>
              <div>容量 ↓</div>
              <div>成本 ↑</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-blue-500">速度 ↓</div>
              <div>容量 ↑</div>
              <div>成本 ↓</div>
            </div>
          </div>
        </div>

        {/* detail panel */}
        <div className="w-full lg:w-64 shrink-0">
          <AnimatePresence mode="wait">
            {info ? (
              <motion.div
                key={info.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="rounded-lg border border-border-subtle bg-bg-secondary p-4"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: info.color }} />
                  <h4 className="font-semibold text-sm text-text-primary">{info.name}</h4>
                </div>
                <div className="space-y-2 text-xs">
                  {[
                    { label: "访问速度", value: info.speed },
                    { label: "容量", value: info.capacity },
                    { label: "单位成本", value: info.cost },
                    { label: "实现技术", value: info.tech },
                  ].map((row) => (
                    <div key={row.label} className="flex justify-between">
                      <span className="text-text-secondary">{row.label}</span>
                      <span className="font-mono font-semibold text-text-primary">{row.value}</span>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-xs text-text-secondary leading-relaxed border-t border-border-subtle pt-3">
                  {info.details}
                </p>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="rounded-lg border border-dashed border-border-subtle bg-bg-secondary p-4 text-center"
              >
                <p className="text-sm text-text-secondary">点击金字塔各层查看信息</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
