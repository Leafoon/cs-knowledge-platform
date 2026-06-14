"use client";
import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, Minus, RotateCcw, ArrowRight } from "lucide-react";

interface Page {
  id: number;
  address: string;
  inFreeList: boolean;
  allocatedTo: string | null;
}

function createInitialPages(count: number): Page[] {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    address: `0x${(0x80000000 + (i + 10) * 0x1000).toString(16)}`,
    inFreeList: i >= 3,
    allocatedTo: i < 3 ? `进程${i + 1}` : null,
  }));
}

export default function Xv6PhysicalAllocator() {
  const [pages, setPages] = useState<Page[]>(() => createInitialPages(10));
  const [log, setLog] = useState<string[]>([
    "初始化完成：3 个页面已分配，7 个页面在空闲链表中",
  ]);
  const [nextId, setNextId] = useState(10);
  const [animatingId, setAnimatingId] = useState<number | null>(null);

  const freeListPages = pages.filter((p) => p.inFreeList);
  const allocatedPages = pages.filter((p) => !p.inFreeList);

  const kalloc = useCallback(() => {
    if (freeListPages.length === 0) {
      setLog((prev) => [...prev, "❌ kalloc 失败：空闲链表为空"]);
      return;
    }
    const page = freeListPages[0];
    setAnimatingId(page.id);
    setTimeout(() => {
      setPages((prev) =>
        prev.map((p) =>
          p.id === page.id
            ? { ...p, inFreeList: false, allocatedTo: `进程${nextId}` }
            : p
        )
      );
      setNextId((n) => n + 1);
      setLog((prev) => [
        ...prev,
        `kalloc() → 返回 ${page.address}，分配给进程${nextId}`,
      ]);
      setAnimatingId(null);
    }, 600);
  }, [freeListPages, nextId]);

  const kfree = useCallback(() => {
    if (allocatedPages.length === 0) {
      setLog((prev) => [...prev, "❌ kfree 失败：没有已分配的页面"]);
      return;
    }
    const page = allocatedPages[allocatedPages.length - 1];
    setAnimatingId(page.id);
    setTimeout(() => {
      setPages((prev) =>
        prev.map((p) =>
          p.id === page.id ? { ...p, inFreeList: true, allocatedTo: null } : p
        )
      );
      setLog((prev) => [
        ...prev,
        `kfree(${page.address}) → 页面归还到空闲链表头部`,
      ]);
      setAnimatingId(null);
    }, 600);
  }, [allocatedPages]);

  const reset = useCallback(() => {
    setPages(createInitialPages(10));
    setNextId(10);
    setLog(["重置完成：3 个页面已分配，7 个页面在空闲链表中"]);
  }, []);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">
        xv6 物理内存分配器
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
        空闲链表（Free List）分配器：kalloc 从头部取出，kfree 归还到头部
      </p>

      <div className="flex gap-3 mb-6">
        <button
          onClick={kalloc}
          disabled={freeListPages.length === 0 || animatingId !== null}
          className="flex items-center gap-1.5 px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <Plus size={16} /> kalloc()
        </button>
        <button
          onClick={kfree}
          disabled={allocatedPages.length === 0 || animatingId !== null}
          className="flex items-center gap-1.5 px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <Minus size={16} /> kfree()
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-1.5 px-4 py-2 bg-slate-500 text-white rounded-lg text-sm font-medium hover:bg-slate-600 transition-colors"
        >
          <RotateCcw size={16} /> 重置
        </button>
      </div>

      <div className="mb-6">
        <h4 className="text-sm font-semibold text-slate-600 dark:text-slate-300 mb-2">
          空闲链表 (kmem.freelist)
        </h4>
        <div className="flex items-center gap-1 flex-wrap min-h-[56px] p-3 bg-slate-50 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
          <span className="text-xs font-mono text-slate-400 mr-1">HEAD →</span>
          <AnimatePresence mode="popLayout">
            {freeListPages.map((page, i) => (
              <motion.div
                key={page.id}
                layout
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8, x: 40 }}
                transition={{ type: "spring", stiffness: 300, damping: 25 }}
                className={`flex items-center gap-1 ${
                  animatingId === page.id ? "ring-2 ring-yellow-400" : ""
                }`}
              >
                <div className="px-2.5 py-1.5 bg-blue-100 dark:bg-blue-900/40 border border-blue-300 dark:border-blue-700 rounded text-xs font-mono text-blue-700 dark:text-blue-300">
                  {page.address}
                </div>
                {i < freeListPages.length - 1 && (
                  <ArrowRight
                    size={14}
                    className="text-slate-400 dark:text-slate-500"
                  />
                )}
              </motion.div>
            ))}
          </AnimatePresence>
          {freeListPages.length === 0 && (
            <span className="text-xs text-red-500 font-mono">NULL</span>
          )}
          <span className="text-xs font-mono text-slate-400 ml-1">→ NULL</span>
        </div>
      </div>

      <div className="mb-6">
        <h4 className="text-sm font-semibold text-slate-600 dark:text-slate-300 mb-2">
          已分配页面
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
          <AnimatePresence mode="popLayout">
            {allocatedPages.map((page) => (
              <motion.div
                key={page.id}
                layout
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className={`px-2 py-2 rounded-lg text-center border ${
                  animatingId === page.id
                    ? "bg-yellow-100 dark:bg-yellow-900/40 border-yellow-400"
                    : "bg-green-50 dark:bg-green-900/30 border-green-300 dark:border-green-700"
                }`}
              >
                <div className="text-xs font-mono text-slate-600 dark:text-slate-300">
                  {page.address}
                </div>
                <div className="text-[10px] text-green-600 dark:text-green-400 mt-0.5">
                  {page.allocatedTo}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {allocatedPages.length === 0 && (
            <div className="col-span-full text-xs text-slate-400 text-center py-2">
              无已分配页面
            </div>
          )}
        </div>
      </div>

      <div>
        <h4 className="text-sm font-semibold text-slate-600 dark:text-slate-300 mb-2">
          操作日志
        </h4>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-lg p-3 h-32 overflow-y-auto font-mono text-xs">
          {log.map((entry, i) => (
            <div
              key={i}
              className={`py-0.5 ${
                entry.includes("❌")
                  ? "text-red-400"
                  : entry.includes("kalloc")
                  ? "text-green-400"
                  : entry.includes("kfree")
                  ? "text-yellow-400"
                  : "text-slate-400"
              }`}
            >
              <span className="text-slate-600">$</span> {entry}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
