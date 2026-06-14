"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Info, ArrowRight, CheckCircle2 } from "lucide-react";

interface PageTableEntry {
  level: number;
  index: string;
  bits: string;
  pfn?: string;
  offset?: string;
}

interface TranslationStep {
  level: number;
  description: string;
  address: string;
  result: string;
}

export function AddressTranslationSimulator() {
  const [virtualAddress, setVirtualAddress] = useState("0x00007F8A12345678");
  const [pageSize, setPageSize] = useState<"4K" | "2M" | "1G">("4K");
  const [showSteps, setShowSteps] = useState(false);
  const [translationSteps, setTranslationSteps] = useState<TranslationStep[]>([]);
  const [physicalAddress, setPhysicalAddress] = useState<string>("");

  const performTranslation = () => {
    const addr = BigInt(virtualAddress);
    const steps: TranslationStep[] = [];
    let currentPFN = 0x00050000n; // 示例页表基址

    if (pageSize === "4K") {
      // 4-level paging (48-bit VA)
      const pml4Index = (addr >> 39n) & 0x1FFn;
      const pdptIndex = (addr >> 30n) & 0x1FFn;
      const pdIndex = (addr >> 21n) & 0x1FFn;
      const ptIndex = (addr >> 12n) & 0x1FFn;
      const offset = addr & 0xFFFn;

      steps.push({
        level: 4,
        description: "PML4 索引",
        address: `bits 47-39: ${pml4Index.toString(16).toUpperCase()}`,
        result: `PML4E → PDPT 基址`,
      });

      steps.push({
        level: 3,
        description: "PDPT 索引",
        address: `bits 38-30: ${pdptIndex.toString(16).toUpperCase()}`,
        result: `PDPTE → PD 基址`,
      });

      steps.push({
        level: 2,
        description: "PD 索引",
        address: `bits 29-21: ${pdIndex.toString(16).toUpperCase()}`,
        result: `PDE → PT 基址`,
      });

      steps.push({
        level: 1,
        description: "PT 索引",
        address: `bits 20-12: ${ptIndex.toString(16).toUpperCase()}`,
        result: `PTE → 物理页框号 (PFN)`,
      });

      const pfn = currentPFN + 0x1234n;
      const physAddr = (pfn << 12n) | offset;
      setPhysicalAddress(`0x${physAddr.toString(16).toUpperCase().padStart(16, "0")}`);

      steps.push({
        level: 0,
        description: "页内偏移",
        address: `bits 11-0: ${offset.toString(16).toUpperCase()}`,
        result: `最终物理地址: 0x${physAddr.toString(16).toUpperCase().padStart(16, "0")}`,
      });
    } else if (pageSize === "2M") {
      // 3-level (PD points to 2MB page)
      const pml4Index = (addr >> 39n) & 0x1FFn;
      const pdptIndex = (addr >> 30n) & 0x1FFn;
      const pdIndex = (addr >> 21n) & 0x1FFn;
      const offset = addr & 0x1FFFFFn; // 21 bits

      steps.push({
        level: 4,
        description: "PML4 索引",
        address: `bits 47-39: ${pml4Index.toString(16).toUpperCase()}`,
        result: `PML4E → PDPT 基址`,
      });

      steps.push({
        level: 3,
        description: "PDPT 索引",
        address: `bits 38-30: ${pdptIndex.toString(16).toUpperCase()}`,
        result: `PDPTE → PD 基址`,
      });

      steps.push({
        level: 2,
        description: "PD 索引 (2MB 页)",
        address: `bits 29-21: ${pdIndex.toString(16).toUpperCase()}`,
        result: `PDE (PS=1) → 物理页框号`,
      });

      const pfn = currentPFN + 0x800n;
      const physAddr = (pfn << 21n) | offset;
      setPhysicalAddress(`0x${physAddr.toString(16).toUpperCase().padStart(16, "0")}`);

      steps.push({
        level: 0,
        description: "页内偏移",
        address: `bits 20-0: ${offset.toString(16).toUpperCase()}`,
        result: `最终物理地址: 0x${physAddr.toString(16).toUpperCase().padStart(16, "0")}`,
      });
    }

    setTranslationSteps(steps);
    setShowSteps(true);
  };

  const handleReset = () => {
    setShowSteps(false);
    setTranslationSteps([]);
    setPhysicalAddress("");
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-6 text-text-primary">
        地址翻译模拟器 (x86-64 分页)
      </h3>

      {/* Input Section */}
      <div className="mb-6 space-y-4">
        <div>
          <label className="block text-sm font-semibold mb-2 text-text-primary">
            虚拟地址 (48-bit)
          </label>
          <input
            type="text"
            value={virtualAddress}
            onChange={(e) => setVirtualAddress(e.target.value)}
            placeholder="0x00007F8A12345678"
            className="w-full px-4 py-2 border border-border-subtle rounded-lg bg-bg-primary text-text-primary font-mono"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold mb-2 text-text-primary">
            页面大小
          </label>
          <div className="flex gap-3">
            {(["4K", "2M", "1G"] as const).map((size) => (
              <button
                key={size}
                onClick={() => {
                  setPageSize(size);
                  handleReset();
                }}
                className={`px-6 py-2 rounded-lg border-2 transition ${
                  pageSize === size
                    ? "border-blue-600 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300"
                    : "border-gray-300 dark:border-gray-700 hover:border-gray-400"
                }`}
              >
                {size}
                {size === "4K" && " (4-level)"}
                {size === "2M" && " (Huge Page)"}
                {size === "1G" && " (Giant Page)"}
              </button>
            ))}
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={performTranslation}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            执行地址翻译
          </button>
          <button
            onClick={handleReset}
            className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
          >
            重置
          </button>
        </div>
      </div>

      {/* Address Breakdown */}
      {showSteps && (
        <div className="mb-6">
          <h4 className="font-semibold mb-3 text-text-primary">
            虚拟地址位域分解
          </h4>
          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg font-mono text-sm">
            <div className="grid grid-cols-5 gap-2 text-center">
              {pageSize === "4K" ? (
                <>
                  <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded">
                    <div className="text-xs text-text-secondary">47-39</div>
                    <div className="font-semibold text-purple-700 dark:text-purple-300">
                      PML4
                    </div>
                  </div>
                  <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded">
                    <div className="text-xs text-text-secondary">38-30</div>
                    <div className="font-semibold text-blue-700 dark:text-blue-300">
                      PDPT
                    </div>
                  </div>
                  <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded">
                    <div className="text-xs text-text-secondary">29-21</div>
                    <div className="font-semibold text-green-700 dark:text-green-300">
                      PD
                    </div>
                  </div>
                  <div className="p-2 bg-yellow-100 dark:bg-yellow-900/30 rounded">
                    <div className="text-xs text-text-secondary">20-12</div>
                    <div className="font-semibold text-yellow-700 dark:text-yellow-300">
                      PT
                    </div>
                  </div>
                  <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded">
                    <div className="text-xs text-text-secondary">11-0</div>
                    <div className="font-semibold text-red-700 dark:text-red-300">
                      Offset
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded">
                    <div className="text-xs text-text-secondary">47-39</div>
                    <div className="font-semibold text-purple-700 dark:text-purple-300">
                      PML4
                    </div>
                  </div>
                  <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded">
                    <div className="text-xs text-text-secondary">38-30</div>
                    <div className="font-semibold text-blue-700 dark:text-blue-300">
                      PDPT
                    </div>
                  </div>
                  <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded">
                    <div className="text-xs text-text-secondary">29-21</div>
                    <div className="font-semibold text-green-700 dark:text-green-300">
                      PD
                    </div>
                  </div>
                  <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded col-span-2">
                    <div className="text-xs text-text-secondary">20-0</div>
                    <div className="font-semibold text-red-700 dark:text-red-300">
                      Offset (2MB)
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Translation Steps */}
      {showSteps && translationSteps.length > 0 && (
        <div className="mb-6">
          <h4 className="font-semibold mb-3 text-text-primary">
            翻译步骤 ({translationSteps.length} 步)
          </h4>
          <div className="space-y-3">
            {translationSteps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.2 }}
                className="flex items-center gap-4"
              >
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold">
                  {index + 1}
                </div>
                <div className="flex-1 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold text-text-primary">
                        {step.description}
                      </div>
                      <div className="text-sm text-text-secondary mt-1 font-mono">
                        {step.address}
                      </div>
                    </div>
                    <ArrowRight className="w-5 h-5 text-text-secondary mx-4" />
                    <div className="text-sm text-text-primary font-mono bg-blue-50 dark:bg-blue-900/20 px-3 py-2 rounded">
                      {step.result}
                    </div>
                  </div>
                </div>
                {index === translationSteps.length - 1 && (
                  <CheckCircle2 className="w-6 h-6 text-green-600 dark:text-green-400" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Final Result */}
      {physicalAddress && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 bg-green-50 dark:bg-green-900/20 rounded-lg border-2 border-green-500"
        >
          <div className="flex items-center gap-3 mb-3">
            <CheckCircle2 className="w-6 h-6 text-green-600 dark:text-green-400" />
            <h4 className="font-semibold text-green-700 dark:text-green-300">
              翻译完成
            </h4>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-text-secondary mb-1">虚拟地址</div>
              <div className="text-lg font-mono font-semibold text-text-primary">
                {virtualAddress}
              </div>
            </div>
            <div>
              <div className="text-sm text-text-secondary mb-1">物理地址</div>
              <div className="text-lg font-mono font-semibold text-green-700 dark:text-green-300">
                {physicalAddress}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Info Box */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-text-secondary">
            <p className="mb-2">
              <strong className="text-text-primary">4-level 分页</strong>：
              PML4 (9 bits) → PDPT (9 bits) → PD (9 bits) → PT (9 bits) → Offset (12
              bits)
            </p>
            <p>
              <strong className="text-text-primary">大页 (2MB)</strong>：
              PML4 → PDPT → PD → Offset (21 bits)，跳过 PT 级别，减少 TLB 缺失
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
