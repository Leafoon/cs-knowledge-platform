"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Download, FileIcon, Clock, Gauge, CheckCircle, Loader2 } from "lucide-react";

interface DownloadTask {
  id: number;
  name: string;
  size: number;
  downloaded: number;
  speed: number;
  status: "等待" | "下载中" | "完成";
  color: string;
}

const FILE_NAMES = ["video.mp4", "archive.zip", "image.png", "document.pdf", "setup.exe", "data.csv"];
const COLORS = ["bg-blue-500", "bg-green-500", "bg-purple-500", "bg-orange-500", "bg-pink-500", "bg-cyan-500"];

export function DownloadProgressDemo() {
  const [maxConcurrent, setMaxConcurrent] = useState(3);
  const [files, setFiles] = useState<DownloadTask[]>([]);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const generateFiles = () => {
    return FILE_NAMES.map((name, i) => ({
      id: i,
      name,
      size: Math.floor(Math.random() * 90) + 10,
      downloaded: 0,
      speed: Math.floor(Math.random() * 8) + 2,
      status: "等待" as const,
      color: COLORS[i],
    }));
  };

  const reset = () => {
    setRunning(false);
    setFiles(generateFiles());
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  useEffect(() => { reset(); }, []);

  const start = () => {
    if (running) return;
    setRunning(true);
    setFiles((prev) => prev.map((f) => ({ ...f, downloaded: 0, status: "等待" as const })));
  };

  useEffect(() => {
    if (!running) return;
    intervalRef.current = setInterval(() => {
      setFiles((prev) => {
        const next = prev.map((f) => ({ ...f }));
        let activeCount = next.filter((f) => f.status === "下载中").length;

        // Start waiting tasks
        for (const f of next) {
          if (f.status === "等待" && activeCount < maxConcurrent) {
            f.status = "下载中";
            activeCount++;
          }
        }

        // Progress active tasks
        for (const f of next) {
          if (f.status === "下载中") {
            f.downloaded = Math.min(f.downloaded + f.speed, f.size);
            if (f.downloaded >= f.size) {
              f.status = "完成";
              f.downloaded = f.size;
            }
          }
        }

        // Check if all done
        if (next.every((f) => f.status === "完成")) {
          setRunning(false);
        }

        return next;
      });
    }, 200);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, maxConcurrent]);

  const activeCount = files.filter((f) => f.status === "下载中").length;
  const completedCount = files.filter((f) => f.status === "完成").length;
  const totalSize = files.reduce((a, f) => a + f.size, 0);
  const totalDownloaded = files.reduce((a, f) => a + f.downloaded, 0);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Download className="w-5 h-5 text-emerald-500" />
        并发下载进度演示
      </h3>

      <div className="flex flex-wrap items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600 dark:text-slate-400">最大并发:</label>
          <div className="flex gap-1">
            {[1, 2, 3, 6].map((n) => (
              <button
                key={n}
                onClick={() => setMaxConcurrent(n)}
                className={`w-8 h-8 rounded-lg text-sm font-bold ${n === maxConcurrent ? "bg-emerald-600 text-white" : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"}`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
        <button onClick={start} disabled={running} className="px-4 py-2 rounded-lg bg-emerald-600 text-white font-medium text-sm flex items-center gap-2 disabled:opacity-50">
          <Play className="w-4 h-4" /> 开始下载
        </button>
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 font-medium text-sm">
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Summary bar */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-600 dark:text-slate-400">总进度</span>
          <span className="text-sm font-mono text-slate-500">{totalDownloaded}/{totalSize} MB</span>
        </div>
        <div className="h-3 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-emerald-500 rounded-full"
            animate={{ width: `${(totalDownloaded / totalSize) * 100}%` }}
          />
        </div>
        <div className="flex items-center justify-between mt-2 text-xs text-slate-500">
          <span className="flex items-center gap-1"><Loader2 className="w-3 h-3" /> {activeCount} 下载中</span>
          <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3" /> {completedCount}/{files.length} 完成</span>
        </div>
      </div>

      {/* File list */}
      <div className="space-y-3">
        {files.map((f) => (
          <motion.div
            key={f.id}
            layout
            className={`rounded-xl border p-4 ${
              f.status === "完成"
                ? "border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/10"
                : f.status === "下载中"
                ? "border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/10"
                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FileIcon className="w-4 h-4 text-slate-400" />
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{f.name}</span>
                <span className="text-xs text-slate-400">{f.size} MB</span>
              </div>
              <div className="flex items-center gap-3">
                {f.status === "下载中" && (
                  <span className="text-xs text-blue-600 dark:text-blue-400 flex items-center gap-1">
                    <Gauge className="w-3 h-3" /> {f.speed} MB/s
                  </span>
                )}
                {f.status === "下载中" && (
                  <span className="text-xs text-slate-400 flex items-center gap-1">
                    <Clock className="w-3 h-3" /> {Math.ceil((f.size - f.downloaded) / f.speed)}s
                  </span>
                )}
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  f.status === "完成" ? "bg-green-200 text-green-800 dark:bg-green-800 dark:text-green-200" :
                  f.status === "下载中" ? "bg-blue-200 text-blue-800 dark:bg-blue-800 dark:text-blue-200" :
                  "bg-slate-200 text-slate-600 dark:bg-slate-700 dark:text-slate-300"
                }`}>{f.status}</span>
              </div>
            </div>
            <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${f.status === "完成" ? "bg-green-500" : "bg-blue-500"}`}
                animate={{ width: `${(f.downloaded / f.size) * 100}%` }}
              />
            </div>
            <div className="text-xs text-slate-400 mt-1 text-right">{f.downloaded}/{f.size} MB ({Math.round((f.downloaded / f.size) * 100)}%)</div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
