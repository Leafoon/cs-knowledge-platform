"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Link2, Unlink, ArrowRight, Play, RotateCcw, Users } from "lucide-react";

interface FDEntry {
  fd: number;
  fileName: string;
  openFileIndex: number | null;
}

interface OpenFileEntry {
  id: number;
  fileName: string;
  offset: number;
  mode: string;
  refCount: number;
  inodeNumber: number;
}

interface InodeEntry {
  number: number;
  type: string;
  size: number;
  permissions: string;
  nlink: number;
}

const initialInodes: InodeEntry[] = [
  { number: 100, type: "regular", size: 4096, permissions: "-rw-r--r--", nlink: 1 },
  { number: 101, type: "regular", size: 8192, permissions: "-rw-rw-r--", nlink: 2 },
  { number: 102, type: "regular", size: 2048, permissions: "-rwxr-xr-x", nlink: 1 },
  { number: 103, type: "regular", size: 16384, permissions: "-rw-------", nlink: 1 },
  { number: 104, type: "directory", size: 4096, permissions: "drwxr-xr-x", nlink: 3 },
];

const predefinedFiles = [
  { name: "stdin", inode: 100, mode: "read" },
  { name: "stdout", inode: 101, mode: "write" },
  { name: "stderr", inode: 101, mode: "write" },
];

export default function FileDescriptorTable() {
  const [fdTable, setFdTable] = useState<(FDEntry | null)[]>([
    { fd: 0, fileName: "stdin", openFileIndex: 0 },
    { fd: 1, fileName: "stdout", openFileIndex: 1 },
    { fd: 2, fileName: "stderr", openFileIndex: 2 },
    null, null, null, null, null,
  ]);
  const [openFileTable, setOpenFileTable] = useState<OpenFileEntry[]>([
    { id: 0, fileName: "stdin", offset: 0, mode: "read", refCount: 1, inodeNumber: 100 },
    { id: 1, fileName: "stdout", offset: 0, mode: "write", refCount: 2, inodeNumber: 101 },
    { id: 2, fileName: "stderr", offset: 0, mode: "write", refCount: 1, inodeNumber: 101 },
  ]);
  const [animatingFd, setAnimatingFd] = useState<number | null>(null);
  const [animatingAction, setAnimatingAction] = useState<"open" | "close" | null>(null);
  const [highlightInode, setHighlightInode] = useState<number | null>(null);
  const [operationLog, setOperationLog] = useState<string[]>([]);
  const [nextOpenFileId, setNextOpenFileId] = useState(3);

  const addLog = (msg: string) => {
    setOperationLog((prev) => [...prev.slice(-5), msg]);
  };

  const openFile = useCallback(() => {
    const freeFd = fdTable.findIndex((entry) => entry === null);
    if (freeFd === -1) {
      addLog("Error: No free file descriptors (max 8)");
      return;
    }

    const fileOptions = [
      { name: "data.txt", inode: 102 },
      { name: "config.json", inode: 103 },
      { name: "/tmp", inode: 104 },
    ];
    const file = fileOptions[Math.floor(Math.random() * fileOptions.length)];

    // Check if file is already open
    const existingEntry = openFileTable.find((e) => e.inodeNumber === file.inode);

    setAnimatingFd(freeFd);
    setAnimatingAction("open");

    setTimeout(() => {
      if (existingEntry) {
        // Share open file entry
        setOpenFileTable((prev) =>
          prev.map((e) => (e.id === existingEntry.id ? { ...e, refCount: e.refCount + 1 } : e))
        );
        setFdTable((prev) => {
          const next = [...prev];
          next[freeFd] = { fd: freeFd, fileName: file.name, openFileIndex: existingEntry.id };
          return next;
        });
        addLog(`open("${file.name}") -> fd=${freeFd} (shared, refcount=${existingEntry.refCount + 1})`);
      } else {
        const newId = nextOpenFileId;
        setNextOpenFileId((prev) => prev + 1);
        const newEntry: OpenFileEntry = {
          id: newId,
          fileName: file.name,
          offset: 0,
          mode: "read-write",
          refCount: 1,
          inodeNumber: file.inode,
        };
        setOpenFileTable((prev) => [...prev, newEntry]);
        setFdTable((prev) => {
          const next = [...prev];
          next[freeFd] = { fd: freeFd, fileName: file.name, openFileIndex: newId };
          return next;
        });
        addLog(`open("${file.name}") -> fd=${freeFd} (new open file entry)`);
      }

      setAnimatingFd(null);
      setAnimatingAction(null);
    }, 800);
  }, [fdTable, openFileTable, nextOpenFileId]);

  const closeFile = useCallback(() => {
    const openFds = fdTable
      .map((entry, i) => (entry && i > 2 ? { ...entry, index: i } : null))
      .filter(Boolean) as (FDEntry & { index: number })[];

    if (openFds.length === 0) {
      addLog("Error: No user file descriptors to close");
      return;
    }

    const target = openFds[openFds.length - 1];
    setAnimatingFd(target.fd);
    setAnimatingAction("close");

    setTimeout(() => {
      const openEntry = openFileTable.find((e) => e.id === target.openFileIndex);

      if (openEntry && openEntry.refCount > 1) {
        setOpenFileTable((prev) =>
          prev.map((e) => (e.id === openEntry.id ? { ...e, refCount: e.refCount - 1 } : e))
        );
        addLog(`close(fd=${target.fd}) -> refcount decreased to ${openEntry.refCount - 1}`);
      } else if (openEntry) {
        setOpenFileTable((prev) => prev.filter((e) => e.id !== openEntry.id));
        addLog(`close(fd=${target.fd}) -> open file entry removed`);
      }

      setFdTable((prev) => {
        const next = [...prev];
        next[target.fd] = null;
        return next;
      });

      setAnimatingFd(null);
      setAnimatingAction(null);
    }, 800);
  }, [fdTable, openFileTable]);

  const reset = () => {
    setFdTable([
      { fd: 0, fileName: "stdin", openFileIndex: 0 },
      { fd: 1, fileName: "stdout", openFileIndex: 1 },
      { fd: 2, fileName: "stderr", openFileIndex: 2 },
      null, null, null, null, null,
    ]);
    setOpenFileTable([
      { id: 0, fileName: "stdin", offset: 0, mode: "read", refCount: 1, inodeNumber: 100 },
      { id: 1, fileName: "stdout", offset: 0, mode: "write", refCount: 2, inodeNumber: 101 },
      { id: 2, fileName: "stderr", offset: 0, mode: "write", refCount: 1, inodeNumber: 101 },
    ]);
    setAnimatingFd(null);
    setAnimatingAction(null);
    setHighlightInode(null);
    setOperationLog([]);
    setNextOpenFileId(3);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        File Descriptor Table
      </h2>

      <div className="flex gap-3 mb-6 justify-center flex-wrap">
        <button
          onClick={openFile}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors"
        >
          <Link2 className="w-4 h-4" />
          Open File
        </button>
        <button
          onClick={closeFile}
          className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
        >
          <Unlink className="w-4 h-4" />
          Close File
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>

      {/* Three tables side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        {/* Process FD Table */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 text-center">
            Process FD Table
          </h3>
          <div className="space-y-1.5">
            {fdTable.map((entry, i) => (
              <motion.div
                key={i}
                className={`flex items-center gap-2 px-3 py-2 rounded text-sm font-mono ${
                  entry
                    ? "bg-slate-50 dark:bg-gray-700"
                    : "bg-slate-100/50 dark:bg-gray-800/50 opacity-50"
                } ${animatingFd === i ? "ring-2 ring-yellow-400" : ""} ${
                  i < 3 ? "border-l-4 border-blue-400" : ""
                }`}
                animate={
                  animatingFd === i
                    ? animatingAction === "open"
                      ? { scale: [1, 1.05, 1], backgroundColor: ["#f0fdf4", "#dcfce7", "#f0fdf4"] }
                      : { scale: [1, 0.95, 1], opacity: [1, 0.5, 1] }
                    : {}
                }
                transition={{ duration: 0.6 }}
              >
                <span className="text-indigo-600 dark:text-indigo-400 font-bold w-8">fd {i}</span>
                {entry ? (
                  <>
                    <span className="text-slate-700 dark:text-gray-200 truncate flex-1">
                      {entry.fileName}
                    </span>
                    <ArrowRight className="w-3 h-3 text-slate-400 flex-shrink-0" />
                    <span className="text-xs text-emerald-600 dark:text-emerald-400 flex-shrink-0">
                      OFT[{entry.openFileIndex}]
                    </span>
                  </>
                ) : (
                  <span className="text-slate-400 dark:text-gray-500 italic">free</span>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Open File Table */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 text-center">
            System Open File Table
          </h3>
          <div className="space-y-1.5">
            <AnimatePresence>
              {openFileTable.map((entry) => (
                <motion.div
                  key={entry.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className={`px-3 py-2 rounded text-xs bg-slate-50 dark:bg-gray-700 border-l-4 border-emerald-400 ${
                    animatingAction === "close" &&
                    fdTable[animatingFd!]?.openFileIndex === entry.id
                      ? "ring-2 ring-red-400"
                      : ""
                  }`}
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-mono font-bold text-emerald-700 dark:text-emerald-300">
                      [{entry.id}] {entry.fileName}
                    </span>
                    <span className="flex items-center gap-1 text-slate-500 dark:text-gray-400">
                      <Users className="w-3 h-3" />
                      ref={entry.refCount}
                    </span>
                  </div>
                  <div className="flex gap-3 text-slate-500 dark:text-gray-400">
                    <span>offset: {entry.offset}</span>
                    <span>mode: {entry.mode}</span>
                  </div>
                  <div className="flex items-center gap-1 mt-1">
                    <ArrowRight className="w-3 h-3 text-slate-400" />
                    <span
                      className="text-xs text-amber-600 dark:text-amber-400 cursor-pointer hover:underline"
                      onMouseEnter={() => setHighlightInode(entry.inodeNumber)}
                      onMouseLeave={() => setHighlightInode(null)}
                    >
                      inode {entry.inodeNumber}
                    </span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            {openFileTable.length === 0 && (
              <p className="text-xs text-slate-400 dark:text-gray-500 text-center py-4 italic">
                No open files
              </p>
            )}
          </div>
        </div>

        {/* Inode Table */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 text-center">
            Inode Table
          </h3>
          <div className="space-y-1.5">
            {initialInodes.map((inode) => {
              const isUsed = openFileTable.some((e) => e.inodeNumber === inode.number);
              return (
                <motion.div
                  key={inode.number}
                  className={`px-3 py-2 rounded text-xs transition-all ${
                    highlightInode === inode.number
                      ? "bg-amber-100 dark:bg-amber-900/40 ring-2 ring-amber-400"
                      : isUsed
                      ? "bg-slate-50 dark:bg-gray-700 border-l-4 border-amber-400"
                      : "bg-slate-100/50 dark:bg-gray-800/50 opacity-60"
                  }`}
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-mono font-bold text-amber-700 dark:text-amber-300">
                      inode {inode.number}
                    </span>
                    <span className="text-slate-500 dark:text-gray-400">{inode.type}</span>
                  </div>
                  <div className="flex gap-3 text-slate-500 dark:text-gray-400">
                    <span>size: {inode.size}</span>
                    <span>{inode.permissions}</span>
                    <span>nlink: {inode.nlink}</span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Operation log */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
        <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-2">Operation Log</h3>
        <div className="space-y-1 max-h-32 overflow-y-auto">
          <AnimatePresence>
            {operationLog.map((msg, i) => (
              <motion.div
                key={`${i}-${msg}`}
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-xs font-mono text-slate-600 dark:text-gray-300"
              >
                <span className="text-slate-400 dark:text-gray-500">$</span> {msg}
              </motion.div>
            ))}
          </AnimatePresence>
          {operationLog.length === 0 && (
            <p className="text-xs text-slate-400 dark:text-gray-500 italic">
              Click &quot;Open File&quot; or &quot;Close File&quot; to see operations.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
