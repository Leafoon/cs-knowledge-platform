"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Folder,
  File,
  Search,
  ArrowRight,
  Play,
  RotateCcw,
  CheckCircle,
  XCircle,
  ChevronRight,
} from "lucide-react";

interface DirEntry {
  name: string;
  inode: number;
  type: "dir" | "file";
}

interface Directory {
  inode: number;
  entries: DirEntry[];
}

// Simulated file system
const directories: Record<number, Directory> = {
  2: {
    inode: 2,
    entries: [
      { name: "home", inode: 131073, type: "dir" },
      { name: "etc", inode: 131074, type: "dir" },
      { name: "var", inode: 131075, type: "dir" },
      { name: "tmp", inode: 131076, type: "dir" },
      { name: "usr", inode: 131077, type: "dir" },
      { name: "bin", inode: 131078, type: "dir" },
    ],
  },
  131073: {
    inode: 131073,
    entries: [
      { name: "user", inode: 262145, type: "dir" },
      { name: "admin", inode: 262146, type: "dir" },
      { name: "guest", inode: 262147, type: "dir" },
    ],
  },
  262145: {
    inode: 262145,
    entries: [
      { name: "Documents", inode: 393217, type: "dir" },
      { name: "Pictures", inode: 393218, type: "dir" },
      { name: ".bashrc", inode: 393219, type: "file" },
      { name: "README.md", inode: 393220, type: "file" },
    ],
  },
  393217: {
    inode: 393217,
    entries: [
      { name: "report.pdf", inode: 524289, type: "file" },
      { name: "notes.txt", inode: 524290, type: "file" },
      { name: "project", inode: 524291, type: "dir" },
    ],
  },
  524291: {
    inode: 524291,
    entries: [
      { name: "main.py", inode: 655361, type: "file" },
      { name: "utils.py", inode: 655362, type: "file" },
      { name: "data.csv", inode: 655363, type: "file" },
    ],
  },
  131074: {
    inode: 131074,
    entries: [
      { name: "passwd", inode: 262148, type: "file" },
      { name: "hosts", inode: 262149, type: "file" },
      { name: "fstab", inode: 262150, type: "file" },
    ],
  },
  262146: {
    inode: 262146,
    entries: [
      { name: "config.json", inode: 393221, type: "file" },
    ],
  },
  262147: {
    inode: 262147,
    entries: [],
  },
  393218: {
    inode: 393218,
    entries: [
      { name: "photo.jpg", inode: 524292, type: "file" },
      { name: "wallpaper.png", inode: 524293, type: "file" },
    ],
  },
};

interface ResolutionStep {
  pathSegment: string;
  fullPath: string;
  directoryInode: number;
  entries: DirEntry[];
  foundEntry: DirEntry | null;
  isLast: boolean;
  error?: string;
}

export default function DirectoryLookup() {
  const [inputPath, setInputPath] = useState("/home/user/Documents/report.pdf");
  const [isResolving, setIsResolving] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [steps, setSteps] = useState<ResolutionStep[]>([]);
  const [resolved, setResolved] = useState(false);
  const [foundInode, setFoundInode] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const resolvePath = useCallback(() => {
    const segments = inputPath.split("/").filter(Boolean);
    if (segments.length === 0) {
      setError("Please enter a valid path.");
      return;
    }

    const result: ResolutionStep[] = [];
    let currentInode = 2; // root inode

    // Root directory step
    const rootDir = directories[currentInode];
    if (!rootDir) {
      setError("Root directory not found.");
      return;
    }

    result.push({
      pathSegment: "/",
      fullPath: "/",
      directoryInode: 2,
      entries: rootDir.entries,
      foundEntry: null,
      isLast: segments.length === 0,
    });

    let failed = false;

    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      const dir = directories[currentInode];

      if (!dir) {
        result.push({
          pathSegment: segment,
          fullPath: "/" + segments.slice(0, i + 1).join("/"),
          directoryInode: currentInode,
          entries: [],
          foundEntry: null,
          isLast: i === segments.length - 1,
          error: `Directory with inode ${currentInode} not found on disk.`,
        });
        failed = true;
        break;
      }

      const entry = dir.entries.find((e) => e.name === segment);

      if (!entry) {
        result.push({
          pathSegment: segment,
          fullPath: "/" + segments.slice(0, i + 1).join("/"),
          directoryInode: currentInode,
          entries: dir.entries,
          foundEntry: null,
          isLast: i === segments.length - 1,
          error: `"${segment}" not found in directory (inode ${currentInode})`,
        });
        failed = true;
        break;
      }

      result.push({
        pathSegment: segment,
        fullPath: "/" + segments.slice(0, i + 1).join("/"),
        directoryInode: currentInode,
        entries: dir.entries,
        foundEntry: entry,
        isLast: i === segments.length - 1,
      });

      if (entry.type === "dir") {
        currentInode = entry.inode;
      } else if (i < segments.length - 1) {
        // Trying to descend into a file
        result.push({
          pathSegment: segments[i + 1],
          fullPath: "/" + segments.slice(0, i + 2).join("/"),
          directoryInode: entry.inode,
          entries: [],
          foundEntry: null,
          isLast: i + 1 === segments.length - 1,
          error: `"${entry.name}" is a file, not a directory. Cannot access "${segments[i + 1]}".`,
        });
        failed = true;
        break;
      }
    }

    setSteps(result);
    setCurrentStep(0);
    setIsResolving(true);
    setResolved(false);
    setFoundInode(null);
    setError(null);

    if (failed) {
      // Will be set during animation
    }
  }, [inputPath]);

  useEffect(() => {
    if (!isResolving || currentStep < 0) return;
    if (currentStep >= steps.length) {
      setIsResolving(false);
      const lastStep = steps[steps.length - 1];
      if (lastStep?.error) {
        setError(lastStep.error);
      } else if (lastStep?.foundEntry) {
        setResolved(true);
        setFoundInode(lastStep.foundEntry.inode);
      }
      return;
    }
    const timer = setTimeout(() => {
      setCurrentStep((prev) => prev + 1);
    }, 1200);
    return () => clearTimeout(timer);
  }, [isResolving, currentStep, steps]);

  const reset = () => {
    setIsResolving(false);
    setCurrentStep(-1);
    setSteps([]);
    setResolved(false);
    setFoundInode(null);
    setError(null);
  };

  const examplePaths = [
    "/home/user/Documents/report.pdf",
    "/home/user/Documents/project/main.py",
    "/etc/passwd",
    "/home/admin/config.json",
    "/home/user/nonexistent.txt",
    "/home/user/Documents/report.pdf/invalid",
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-teal-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Directory Lookup / Path Resolution
      </h2>

      {/* Path input */}
      <div className="flex flex-col sm:flex-row gap-3 mb-4 items-center justify-center">
        <div className="relative flex-1 max-w-lg">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={inputPath}
            onChange={(e) => {
              setInputPath(e.target.value);
              reset();
            }}
            onKeyDown={(e) => e.key === "Enter" && resolvePath()}
            placeholder="/path/to/file"
            className="w-full pl-10 pr-4 py-2.5 text-sm font-mono bg-white dark:bg-gray-800 border border-slate-300 dark:border-gray-600 rounded-lg text-slate-800 dark:text-gray-100 focus:ring-2 focus:ring-teal-400 focus:border-transparent"
          />
        </div>
        <div className="flex gap-2">
          <button
            onClick={resolvePath}
            disabled={isResolving}
            className="flex items-center gap-2 px-4 py-2.5 bg-teal-500 text-white rounded-lg hover:bg-teal-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Play className="w-4 h-4" />
            Resolve
          </button>
          <button
            onClick={reset}
            className="flex items-center gap-2 px-4 py-2.5 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Example paths */}
      <div className="flex gap-2 mb-6 justify-center flex-wrap">
        {examplePaths.map((p) => (
          <button
            key={p}
            onClick={() => {
              setInputPath(p);
              reset();
            }}
            className="px-3 py-1 text-xs font-mono bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-gray-300 rounded-full hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors truncate max-w-[200px]"
          >
            {p}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Resolution steps */}
        <div className="lg:col-span-2">
          <AnimatePresence>
            {steps.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-3"
              >
                {steps.map((step, i) => {
                  const isActive = currentStep === i;
                  const isDone = currentStep > i;
                  const hasError = !!step.error && isDone;

                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: i <= currentStep ? 1 : 0.2, x: 0 }}
                      className={`bg-white dark:bg-gray-800 rounded-lg p-4 border-l-4 transition-all ${
                        hasError
                          ? "border-red-400"
                          : isActive
                          ? "border-yellow-400 shadow-md"
                          : isDone
                          ? "border-emerald-400"
                          : "border-slate-200 dark:border-gray-700"
                      }`}
                    >
                      {/* Step header */}
                      <div className="flex items-center gap-2 mb-2">
                        {isActive && (
                          <motion.div
                            animate={{ scale: [1, 1.3, 1] }}
                            transition={{ repeat: Infinity, duration: 1 }}
                            className="w-2.5 h-2.5 rounded-full bg-yellow-400"
                          />
                        )}
                        {isDone && !hasError && (
                          <CheckCircle className="w-4 h-4 text-emerald-500" />
                        )}
                        {hasError && <XCircle className="w-4 h-4 text-red-500" />}

                        <span className="text-sm font-bold text-slate-700 dark:text-gray-200">
                          Step {i + 1}:{" "}
                          {i === 0
                            ? "Read root directory"
                            : `Look up "${step.pathSegment}" in ${step.fullPath
                                .split("/")
                                .slice(0, -1)
                                .join("/") || "/"}`}
                        </span>

                        <span className="ml-auto text-xs font-mono text-slate-500 dark:text-gray-400">
                          inode {step.directoryInode}
                        </span>
                      </div>

                      {/* Directory contents */}
                      {isDone && step.entries.length > 0 && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          className="mb-2"
                        >
                          <p className="text-xs text-slate-500 dark:text-gray-400 mb-1">
                            Directory entries:
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {step.entries.map((entry) => {
                              const isFound =
                                step.foundEntry && step.foundEntry.name === entry.name;
                              return (
                                <span
                                  key={entry.name}
                                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono ${
                                    isFound
                                      ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 ring-1 ring-emerald-400 font-bold"
                                      : "bg-slate-100 dark:bg-gray-700 text-slate-600 dark:text-gray-300"
                                  }`}
                                >
                                  {entry.type === "dir" ? (
                                    <Folder className="w-3 h-3 text-blue-500" />
                                  ) : (
                                    <File className="w-3 h-3 text-slate-500" />
                                  )}
                                  {entry.name}
                                  <span className="text-[10px] text-slate-400 dark:text-gray-500">
                                    ({entry.inode})
                                  </span>
                                </span>
                              );
                            })}
                          </div>
                        </motion.div>
                      )}

                      {/* Result */}
                      {isDone && step.foundEntry && (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="flex items-center gap-2 text-xs"
                        >
                          <ArrowRight className="w-3 h-3 text-emerald-500" />
                          <span className="text-emerald-600 dark:text-emerald-400 font-medium">
                            Found &quot;{step.foundEntry.name}&quot; (inode {step.foundEntry.inode},{" "}
                            {step.foundEntry.type})
                            {step.isLast && " -- target reached!"}
                          </span>
                        </motion.div>
                      )}

                      {/* Error */}
                      {hasError && (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="flex items-center gap-2 text-xs"
                        >
                          <XCircle className="w-3 h-3 text-red-500" />
                          <span className="text-red-600 dark:text-red-400 font-medium">
                            {step.error}
                          </span>
                        </motion.div>
                      )}
                    </motion.div>
                  );
                })}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Result panel */}
        <div className="lg:col-span-1">
          <AnimatePresence mode="wait">
            {resolved && foundInode !== null ? (
              <motion.div
                key="resolved"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-5 border border-emerald-300 dark:border-emerald-700"
              >
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle className="w-6 h-6 text-emerald-500" />
                  <h3 className="text-lg font-bold text-emerald-700 dark:text-emerald-300">
                    Path Resolved
                  </h3>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-gray-300">Path</span>
                    <span className="font-mono text-slate-800 dark:text-gray-100">{inputPath}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-gray-300">Inode</span>
                    <span className="font-mono font-bold text-emerald-700 dark:text-emerald-300">
                      {foundInode}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-gray-300">Steps</span>
                    <span className="font-mono text-slate-800 dark:text-gray-100">
                      {steps.length}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-gray-300">Disk reads</span>
                    <span className="font-mono text-slate-800 dark:text-gray-100">
                      {steps.length} (one per directory)
                    </span>
                  </div>
                </div>
              </motion.div>
            ) : error ? (
              <motion.div
                key="error"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-red-50 dark:bg-red-900/20 rounded-lg p-5 border border-red-300 dark:border-red-700"
              >
                <div className="flex items-center gap-2 mb-3">
                  <XCircle className="w-6 h-6 text-red-500" />
                  <h3 className="text-lg font-bold text-red-700 dark:text-red-300">
                    Resolution Failed
                  </h3>
                </div>
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700 text-center"
              >
                <Search className="w-8 h-8 text-slate-300 dark:text-gray-600 mx-auto mb-3" />
                <p className="text-sm text-slate-500 dark:text-gray-400">
                  Enter a path and click Resolve to see step-by-step path resolution.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* How it works */}
          <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4 shadow-md border border-slate-200 dark:border-gray-700">
            <h4 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-2">
              Path Resolution Algorithm
            </h4>
            <div className="text-xs text-slate-600 dark:text-gray-300 space-y-1.5">
              <div className="flex items-start gap-2">
                <span className="font-bold text-indigo-500 flex-shrink-0">1.</span>
                <span>Start at root inode (always inode 2)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="font-bold text-indigo-500 flex-shrink-0">2.</span>
                <span>Read directory data blocks into memory</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="font-bold text-indigo-500 flex-shrink-0">3.</span>
                <span>Search directory entries for next path component</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="font-bold text-indigo-500 flex-shrink-0">4.</span>
                <span>Get inode number from matching entry</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="font-bold text-indigo-500 flex-shrink-0">5.</span>
                <span>Read inode from disk, repeat for next component</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="font-bold text-indigo-500 flex-shrink-0">6.</span>
                <span>Each component requires at least one disk read</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
