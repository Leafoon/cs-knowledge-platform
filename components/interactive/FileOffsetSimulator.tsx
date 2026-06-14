"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  PenLine,
  BookOpen,
  RotateCcw,
  SkipForward,
  SkipBack,
} from "lucide-react";

type Whence = "SET" | "CUR" | "END";

export default function FileOffsetSimulator() {
  const fileSize = 40;
  const [fileContent, setFileContent] = useState<string>(
    "Hello, World! This is a sample file content."
  );
  const [offset, setOffset] = useState(0);
  const [readOutput, setReadOutput] = useState("");
  const [writeInput, setWriteInput] = useState("");
  const [readCount, setReadCount] = useState(5);
  const [seekOffset, setSeekOffset] = useState(0);
  const [seekWhence, setSeekWhence] = useState<Whence>("SET");
  const [operationLog, setOperationLog] = useState<string[]>([]);
  const [highlightRange, setHighlightRange] = useState<[number, number] | null>(null);
  const [animationKey, setAnimationKey] = useState(0);

  const addLog = (msg: string) => {
    setOperationLog((prev) => [...prev.slice(-8), msg]);
  };

  const doRead = useCallback(() => {
    const end = Math.min(offset + readCount, fileContent.length);
    const data = fileContent.slice(offset, end);
    const bytesRead = data.length;
    setOffset(end);
    setReadOutput(data);
    setHighlightRange([offset, end]);
    setAnimationKey((k) => k + 1);
    addLog(`read(fd, buf, ${readCount}) = ${bytesRead}  ->  "${data}"`);
  }, [offset, readCount, fileContent]);

  const doWrite = useCallback(() => {
    if (writeInput.length === 0) return;
    const before = fileContent.slice(0, offset);
    const after = fileContent.slice(offset + writeInput.length);
    const newContent = before + writeInput + after;
    setFileContent(newContent);
    setHighlightRange([offset, offset + writeInput.length]);
    setAnimationKey((k) => k + 1);
    addLog(`write(fd, "${writeInput}", ${writeInput.length}) = ${writeInput.length}`);
    setOffset((prev) => prev + writeInput.length);
  }, [offset, writeInput, fileContent]);

  const doLseek = useCallback(() => {
    let newOffset: number;
    switch (seekWhence) {
      case "SET":
        newOffset = seekOffset;
        break;
      case "CUR":
        newOffset = offset + seekOffset;
        break;
      case "END":
        newOffset = fileContent.length + seekOffset;
        break;
    }
    newOffset = Math.max(0, Math.min(newOffset, fileContent.length));
    setOffset(newOffset);
    setHighlightRange(null);
    setAnimationKey((k) => k + 1);
    addLog(`lseek(fd, ${seekOffset}, SEEK_${seekWhence}) -> offset = ${newOffset}`);
  }, [seekOffset, seekWhence, offset, fileContent.length]);

  const reset = () => {
    setFileContent("Hello, World! This is a sample file content.");
    setOffset(0);
    setReadOutput("");
    setWriteInput("");
    setReadCount(5);
    setSeekOffset(0);
    setSeekWhence("SET");
    setOperationLog([]);
    setHighlightRange(null);
  };

  const contentArr = fileContent.split("");

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        File Offset Simulator
      </h2>

      {/* File visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-5 mb-6 border border-slate-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200">
            File Content ({fileContent.length} bytes)
          </h3>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-slate-500 dark:text-gray-400">Offset:</span>
            <motion.span
              key={animationKey}
              initial={{ scale: 1.3 }}
              animate={{ scale: 1 }}
              className="font-mono font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/30 px-2 py-0.5 rounded"
            >
              {offset}
            </motion.span>
          </div>
        </div>

        {/* Byte positions */}
        <div className="flex flex-wrap gap-0 mb-1">
          {contentArr.map((_, i) => (
            <div
              key={i}
              className="w-[1.4rem] text-center text-[9px] text-slate-400 dark:text-gray-500 font-mono"
            >
              {i % 10 === 0 ? i : ""}
            </div>
          ))}
        </div>

        {/* Characters */}
        <div className="flex flex-wrap gap-0 relative mb-2">
          {contentArr.map((ch, i) => {
            const isAtOffset = i === offset;
            const isHighlighted = highlightRange && i >= highlightRange[0] && i < highlightRange[1];

            return (
              <motion.div
                key={i}
                className={`w-[1.4rem] h-[1.4rem] flex items-center justify-center text-xs font-mono border border-slate-200 dark:border-gray-600 rounded-sm transition-colors ${
                  isAtOffset
                    ? "bg-indigo-500 text-white border-indigo-500"
                    : isHighlighted
                    ? "bg-yellow-200 dark:bg-yellow-800 text-slate-800 dark:text-gray-100 border-yellow-400"
                    : "bg-slate-50 dark:bg-gray-700 text-slate-700 dark:text-gray-200"
                }`}
              >
                {ch === " " ? " " : ch}
              </motion.div>
            );
          })}

          {/* Offset cursor */}
          <motion.div
            className="absolute bottom-0 w-[1.4rem] flex justify-center pointer-events-none"
            style={{ left: `${offset * 1.4}rem` }}
            animate={{ bottom: "-20px" }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            key={`cursor-${animationKey}`}
          >
            <div className="flex flex-col items-center">
              <div className="w-0 h-0 border-l-[5px] border-r-[5px] border-b-[8px] border-l-transparent border-r-transparent border-b-indigo-500" />
              <span className="text-[9px] font-mono text-indigo-500 font-bold mt-0.5">
                {offset}
              </span>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Operations */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Read */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-blue-500" />
            read()
          </h3>
          <div className="flex items-center gap-2 mb-3">
            <label className="text-xs text-slate-500 dark:text-gray-400">count:</label>
            <input
              type="number"
              min={1}
              max={40}
              value={readCount}
              onChange={(e) => setReadCount(Math.max(1, Math.min(40, Number(e.target.value))))}
              className="w-16 px-2 py-1 text-sm font-mono bg-slate-100 dark:bg-gray-700 border border-slate-300 dark:border-gray-600 rounded text-slate-800 dark:text-gray-100"
            />
          </div>
          <button
            onClick={doRead}
            className="w-full px-3 py-2 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 transition-colors"
          >
            Read {readCount} bytes
          </button>
          {readOutput && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-xs font-mono text-blue-700 dark:text-blue-300 break-all"
            >
              &quot;{readOutput}&quot;
            </motion.div>
          )}
        </div>

        {/* Write */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
            <PenLine className="w-4 h-4 text-emerald-500" />
            write()
          </h3>
          <div className="mb-3">
            <label className="text-xs text-slate-500 dark:text-gray-400 mb-1 block">data:</label>
            <input
              type="text"
              value={writeInput}
              onChange={(e) => setWriteInput(e.target.value)}
              placeholder="Enter text to write..."
              className="w-full px-2 py-1 text-sm font-mono bg-slate-100 dark:bg-gray-700 border border-slate-300 dark:border-gray-600 rounded text-slate-800 dark:text-gray-100"
            />
          </div>
          <button
            onClick={doWrite}
            disabled={!writeInput}
            className="w-full px-3 py-2 bg-emerald-500 text-white text-sm rounded hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Write Data
          </button>
        </div>

        {/* Lseek */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
            <SkipForward className="w-4 h-4 text-violet-500" />
            lseek()
          </h3>
          <div className="space-y-2 mb-3">
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-500 dark:text-gray-400">offset:</label>
              <input
                type="number"
                value={seekOffset}
                onChange={(e) => setSeekOffset(Number(e.target.value))}
                className="w-20 px-2 py-1 text-sm font-mono bg-slate-100 dark:bg-gray-700 border border-slate-300 dark:border-gray-600 rounded text-slate-800 dark:text-gray-100"
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-500 dark:text-gray-400">whence:</label>
              <select
                value={seekWhence}
                onChange={(e) => setSeekWhence(e.target.value as Whence)}
                className="px-2 py-1 text-sm font-mono bg-slate-100 dark:bg-gray-700 border border-slate-300 dark:border-gray-600 rounded text-slate-800 dark:text-gray-100"
              >
                <option value="SET">SEEK_SET (from start)</option>
                <option value="CUR">SEEK_CUR (from current)</option>
                <option value="END">SEEK_END (from end)</option>
              </select>
            </div>
          </div>
          <button
            onClick={doLseek}
            className="w-full px-3 py-2 bg-violet-500 text-white text-sm rounded hover:bg-violet-600 transition-colors"
          >
            Seek
          </button>
        </div>
      </div>

      {/* Quick seek buttons */}
      <div className="flex gap-2 mb-6 justify-center flex-wrap">
        <button
          onClick={() => {
            setOffset(0);
            setAnimationKey((k) => k + 1);
            addLog("lseek(fd, 0, SEEK_SET) -> offset = 0");
          }}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-200 dark:bg-gray-700 text-slate-700 dark:text-gray-200 text-xs rounded hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors"
        >
          <SkipBack className="w-3 h-3" /> Start
        </button>
        <button
          onClick={() => {
            const newOff = Math.min(offset + 5, fileContent.length);
            setOffset(newOff);
            setAnimationKey((k) => k + 1);
            addLog(`lseek(fd, +5, SEEK_CUR) -> offset = ${newOff}`);
          }}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-200 dark:bg-gray-700 text-slate-700 dark:text-gray-200 text-xs rounded hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors"
        >
          <ArrowRight className="w-3 h-3" /> +5
        </button>
        <button
          onClick={() => {
            const newOff = Math.max(offset - 5, 0);
            setOffset(newOff);
            setAnimationKey((k) => k + 1);
            addLog(`lseek(fd, -5, SEEK_CUR) -> offset = ${newOff}`);
          }}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-200 dark:bg-gray-700 text-slate-700 dark:text-gray-200 text-xs rounded hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors"
        >
          <ArrowLeft className="w-3 h-3" /> -5
        </button>
        <button
          onClick={() => {
            setOffset(fileContent.length);
            setAnimationKey((k) => k + 1);
            addLog(`lseek(fd, 0, SEEK_END) -> offset = ${fileContent.length}`);
          }}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-200 dark:bg-gray-700 text-slate-700 dark:text-gray-200 text-xs rounded hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors"
        >
          <SkipForward className="w-3 h-3" /> End
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-200 dark:bg-gray-700 text-slate-700 dark:text-gray-200 text-xs rounded hover:bg-slate-300 dark:hover:bg-gray-600 transition-colors"
        >
          <RotateCcw className="w-3 h-3" /> Reset
        </button>
      </div>

      {/* Operation log */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
        <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-2">
          Operation Log
        </h3>
        <div className="space-y-1 max-h-40 overflow-y-auto font-mono">
          <AnimatePresence>
            {operationLog.map((msg, i) => (
              <motion.div
                key={`${i}-${msg}`}
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-xs text-slate-600 dark:text-gray-300"
              >
                <span className="text-emerald-500">$</span> {msg}
              </motion.div>
            ))}
          </AnimatePresence>
          {operationLog.length === 0 && (
            <p className="text-xs text-slate-400 dark:text-gray-500 italic">
              Perform read, write, or lseek operations to see the log.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
