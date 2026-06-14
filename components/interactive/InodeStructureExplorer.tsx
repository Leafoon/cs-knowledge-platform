"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  HardDrive,
  Database,
  Clock,
  User,
  Shield,
  Layers,
  ArrowRight,
  Info,
  Play,
  RotateCcw,
} from "lucide-react";

interface InodeField {
  name: string;
  offset: string;
  size: string;
  value: string;
  description: string;
  color: string;
  category: "metadata" | "ownership" | "time" | "size" | "pointers";
}

const inodeFields: InodeField[] = [
  {
    name: "i_mode",
    offset: "0",
    size: "16 bits",
    value: "0x81A4",
    description: "File type and permissions. Upper 4 bits: type (regular=1000). Lower 12 bits: permissions (rwxr--r-- = 644).",
    color: "bg-purple-500",
    category: "metadata",
  },
  {
    name: "i_uid",
    offset: "2",
    size: "16 bits",
    value: "1000",
    description: "User ID of the file owner. 1000 = first regular user.",
    color: "bg-blue-500",
    category: "ownership",
  },
  {
    name: "i_size",
    offset: "4",
    size: "32 bits",
    value: "40960",
    description: "File size in bytes. For ext4, this is 32 bits in the classic inode; the i_size_hi field extends it to 64 bits.",
    color: "bg-emerald-500",
    category: "size",
  },
  {
    name: "i_atime",
    offset: "8",
    size: "32 bits",
    value: "1717986000",
    description: "Last access time (seconds since Unix epoch). Updated on read.",
    color: "bg-amber-500",
    category: "time",
  },
  {
    name: "i_ctime",
    offset: "12",
    size: "32 bits",
    value: "1717899600",
    description: "Inode change time. Updated when inode metadata changes (permissions, ownership, etc.).",
    color: "bg-amber-500",
    category: "time",
  },
  {
    name: "i_mtime",
    offset: "16",
    size: "32 bits",
    value: "1717900800",
    description: "Last modification time. Updated when file content changes.",
    color: "bg-amber-500",
    category: "time",
  },
  {
    name: "i_dtime",
    offset: "20",
    size: "32 bits",
    value: "0",
    description: "Deletion time. 0 means the file is not deleted. Set when the inode is freed.",
    color: "bg-amber-500",
    category: "time",
  },
  {
    name: "i_gid",
    offset: "24",
    size: "16 bits",
    value: "1000",
    description: "Group ID of the file owner.",
    color: "bg-blue-500",
    category: "ownership",
  },
  {
    name: "i_nlink",
    offset: "26",
    size: "16 bits",
    value: "1",
    description: "Hard link count. When it reaches 0, the inode and its data blocks can be freed.",
    color: "bg-purple-500",
    category: "metadata",
  },
  {
    name: "i_blocks",
    offset: "28",
    size: "32 bits",
    value: "80",
    description: "Number of 512-byte blocks allocated. 80 blocks = 40960 bytes.",
    color: "bg-emerald-500",
    category: "size",
  },
  {
    name: "i_block[0-11]",
    offset: "32",
    size: "12 x 32 bits",
    value: "[1024, 1025, ...]",
    description: "12 direct block pointers. Each points to a data block. Max: 12 * 4KB = 48KB directly.",
    color: "bg-red-500",
    category: "pointers",
  },
  {
    name: "i_block[12]",
    offset: "80",
    size: "32 bits",
    value: "2048",
    description: "Single indirect pointer. Points to a block containing 1024 more block pointers (4MB).",
    color: "bg-red-500",
    category: "pointers",
  },
  {
    name: "i_block[13]",
    offset: "84",
    size: "32 bits",
    value: "3072",
    description: "Double indirect pointer. Points to a block of pointers, each pointing to another block of pointers (4GB).",
    color: "bg-red-500",
    category: "pointers",
  },
  {
    name: "i_block[14]",
    offset: "88",
    size: "32 bits",
    value: "0",
    description: "Triple indirect pointer. Three levels of indirection (4TB).",
    color: "bg-red-500",
    category: "pointers",
  },
];

export default function InodeStructureExplorer() {
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [isReading, setIsReading] = useState(false);
  const [readProgress, setReadProgress] = useState(-1);
  const [filterCategory, setFilterCategory] = useState<string | null>(null);

  const startReadAnimation = () => {
    if (isReading) return;
    setIsReading(true);
    setReadProgress(0);
  };

  useEffect(() => {
    if (!isReading) return;
    const timer = setInterval(() => {
      setReadProgress((prev) => {
        if (prev >= inodeFields.length - 1) {
          setIsReading(false);
          return -1;
        }
        return prev + 1;
      });
    }, 300);
    return () => clearInterval(timer);
  }, [isReading]);

  const resetAnimation = () => {
    setIsReading(false);
    setReadProgress(-1);
    setSelectedField(null);
  };

  const selectedFieldData = inodeFields.find((f) => f.name === selectedField);

  const filteredFields = filterCategory
    ? inodeFields.filter((f) => f.category === filterCategory)
    : inodeFields;

  const categories = [
    { id: "metadata", label: "Metadata", icon: <Info className="w-3 h-3" /> },
    { id: "ownership", label: "Ownership", icon: <User className="w-3 h-3" /> },
    { id: "time", label: "Timestamps", icon: <Clock className="w-3 h-3" /> },
    { id: "size", label: "Size/Blocks", icon: <Database className="w-3 h-3" /> },
    { id: "pointers", label: "Pointers", icon: <Layers className="w-3 h-3" /> },
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Inode Structure Explorer
      </h2>

      <div className="flex gap-3 mb-6 justify-center flex-wrap">
        <button
          onClick={startReadAnimation}
          disabled={isReading}
          className="flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Play className="w-4 h-4" />
          Animate Reading Inode
        </button>
        <button
          onClick={resetAnimation}
          className="flex items-center gap-2 px-4 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>

      {/* Category filters */}
      <div className="flex gap-2 mb-4 justify-center flex-wrap">
        <button
          onClick={() => setFilterCategory(null)}
          className={`px-3 py-1.5 text-xs rounded-full transition-colors ${
            filterCategory === null
              ? "bg-slate-700 text-white dark:bg-slate-300 dark:text-gray-900"
              : "bg-slate-200 text-slate-600 dark:bg-gray-700 dark:text-gray-300 hover:bg-slate-300"
          }`}
        >
          All
        </button>
        {categories.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setFilterCategory(filterCategory === cat.id ? null : cat.id)}
            className={`flex items-center gap-1 px-3 py-1.5 text-xs rounded-full transition-colors ${
              filterCategory === cat.id
                ? "bg-slate-700 text-white dark:bg-slate-300 dark:text-gray-900"
                : "bg-slate-200 text-slate-600 dark:bg-gray-700 dark:text-gray-300 hover:bg-slate-300"
            }`}
          >
            {cat.icon}
            {cat.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Inode table */}
        <div className="lg:col-span-2">
          {/* Disk layout header */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 mb-4">
            <h3 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3 flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-amber-500" />
              Inode on Disk (128 bytes, ext4)
            </h3>
            <div className="flex items-center gap-1 text-xs text-slate-500 dark:text-gray-400 mb-2">
              <span>Byte 0</span>
              <div className="flex-1 h-px bg-slate-300 dark:bg-gray-600" />
              <span>Byte 127</span>
            </div>
          </div>

          {/* Inode fields as cells */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <AnimatePresence>
                {filteredFields.map((field, index) => {
                  const isReadingNow = readProgress === index;
                  const hasBeenRead = readProgress >= index && isReading;

                  return (
                    <motion.button
                      key={field.name}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      onClick={() =>
                        setSelectedField(selectedField === field.name ? null : field.name)
                      }
                      className={`
                        p-3 rounded-lg text-left transition-all cursor-pointer border-2
                        ${
                          selectedField === field.name
                            ? "border-indigo-400 shadow-md"
                            : "border-transparent hover:border-slate-300 dark:hover:border-gray-600"
                        }
                        ${isReadingNow ? "ring-2 ring-yellow-400" : ""}
                        ${hasBeenRead ? "bg-slate-50 dark:bg-gray-750" : "bg-white dark:bg-gray-800"}
                      `}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      {/* Scan effect */}
                      {isReadingNow && (
                        <motion.div
                          className="absolute inset-0 bg-amber-400/20 rounded-lg"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: [0, 0.5, 0] }}
                          transition={{ duration: 0.5 }}
                        />
                      )}

                      <div className="flex items-center gap-2 mb-1">
                        <div className={`w-2 h-2 rounded-full ${field.color}`} />
                        <span className="text-xs font-mono font-bold text-slate-700 dark:text-gray-200">
                          {field.name}
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-500 dark:text-gray-400 font-mono">
                        offset: {field.offset} | {field.size}
                      </div>
                      <div className="text-[10px] text-slate-600 dark:text-gray-300 font-mono mt-1 truncate">
                        {field.value}
                      </div>
                    </motion.button>
                  );
                })}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Details panel */}
        <div className="lg:col-span-1">
          <AnimatePresence mode="wait">
            {selectedFieldData ? (
              <motion.div
                key={selectedFieldData.name}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className={`w-3 h-3 rounded-full ${selectedFieldData.color}`} />
                  <h3 className="text-lg font-bold text-slate-800 dark:text-gray-100 font-mono">
                    {selectedFieldData.name}
                  </h3>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500 dark:text-gray-400">Offset</span>
                    <span className="text-sm font-mono text-slate-700 dark:text-gray-200">
                      {selectedFieldData.offset}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500 dark:text-gray-400">Size</span>
                    <span className="text-sm font-mono text-slate-700 dark:text-gray-200">
                      {selectedFieldData.size}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500 dark:text-gray-400">Value</span>
                    <span className="text-sm font-mono text-slate-700 dark:text-gray-200">
                      {selectedFieldData.value}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-xs text-slate-500 dark:text-gray-400">Category</span>
                    <span className="text-sm text-slate-700 dark:text-gray-200 capitalize">
                      {selectedFieldData.category}
                    </span>
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-slate-200 dark:border-gray-700">
                  <p className="text-sm text-slate-600 dark:text-gray-300 leading-relaxed">
                    {selectedFieldData.description}
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700 text-center"
              >
                <Info className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                <p className="text-sm text-slate-500 dark:text-gray-400">
                  Click on any inode field to see its description and bit layout.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Data block pointers visualization */}
          <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4 shadow-md border border-slate-200 dark:border-gray-700">
            <h4 className="text-sm font-bold text-slate-700 dark:text-gray-200 mb-3">
              Data Block Pointers
            </h4>
            <div className="space-y-2">
              {[
                { label: "Direct [0-11]", count: 12, size: "48 KB", color: "bg-red-400" },
                { label: "Single Indirect", count: 1, size: "4 MB", color: "bg-orange-400" },
                { label: "Double Indirect", count: 1, size: "4 GB", color: "bg-yellow-400" },
                { label: "Triple Indirect", count: 1, size: "4 TB", color: "bg-green-400" },
              ].map((ptr) => (
                <div key={ptr.label} className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded ${ptr.color}`} />
                  <span className="text-xs text-slate-600 dark:text-gray-300 flex-1">
                    {ptr.label}
                  </span>
                  <span className="text-xs font-mono text-slate-500 dark:text-gray-400">
                    {ptr.count} ptr{ptr.count > 1 ? "s" : ""}
                  </span>
                  <ArrowRight className="w-3 h-3 text-slate-400" />
                  <span className="text-xs font-mono text-emerald-600 dark:text-emerald-400">
                    {ptr.size}
                  </span>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-3 border-t border-slate-200 dark:border-gray-700 text-xs text-slate-500 dark:text-gray-400">
              Total max file size: ~4 TB (with 4KB blocks)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
