"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { File, Folder, FolderOpen, Edit, Trash2, Copy, Play, CheckCircle } from "lucide-react";

interface FileNode {
  name: string;
  type: "file" | "dir";
  size?: number;
  children?: FileNode[];
}

interface Operation {
  id: string;
  name: string;
  syscall: string;
  description: string;
  icon: React.ReactNode;
}

const operations: Operation[] = [
  {
    id: "create",
    name: "创建文件",
    syscall: "open(path, O_CREAT|O_WRONLY, mode)",
    description: "创建新文件并返回文件描述符",
    icon: <File className="w-5 h-5" />
  },
  {
    id: "read",
    name: "读取文件",
    syscall: "read(fd, buffer, count)",
    description: "从文件读取数据到缓冲区",
    icon: <File className="w-5 h-5" />
  },
  {
    id: "write",
    name: "写入文件",
    syscall: "write(fd, buffer, count)",
    description: "将缓冲区数据写入文件",
    icon: <Edit className="w-5 h-5" />
  },
  {
    id: "delete",
    name: "删除文件",
    syscall: "unlink(path)",
    description: "删除文件链接，引用计数为0时删除",
    icon: <Trash2 className="w-5 h-5" />
  },
  {
    id: "mkdir",
    name: "创建目录",
    syscall: "mkdir(path, mode)",
    description: "创建新目录",
    icon: <Folder className="w-5 h-5" />
  },
  {
    id: "rmdir",
    name: "删除目录",
    syscall: "rmdir(path)",
    description: "删除空目录",
    icon: <Trash2 className="w-5 h-5" />
  }
];

export default function FileSystemOperations() {
  const [selectedOp, setSelectedOp] = useState<string>("create");
  const [executing, setExecuting] = useState(false);
  const [executed, setExecuted] = useState(false);

  const [fileSystem, setFileSystem] = useState<FileNode[]>([
    {
      name: "home",
      type: "dir",
      children: [
        { name: "document.txt", type: "file", size: 1024 },
        { name: "photo.jpg", type: "file", size: 2048 }
      ]
    }
  ]);

  const executeOperation = () => {
    setExecuting(true);
    setExecuted(false);

    setTimeout(() => {
      if (selectedOp === "create") {
        setFileSystem(prev => {
          const newFs = [...prev];
          newFs[0].children?.push({
            name: "new_file.txt",
            type: "file",
            size: 0
          });
          return newFs;
        });
      } else if (selectedOp === "mkdir") {
        setFileSystem(prev => {
          const newFs = [...prev];
          newFs[0].children?.push({
            name: "new_folder",
            type: "dir",
            children: []
          });
          return newFs;
        });
      }

      setExecuting(false);
      setExecuted(true);

      setTimeout(() => setExecuted(false), 2000);
    }, 1000);
  };

  const selected = operations.find(op => op.id === selectedOp);

  const renderFileTree = (nodes: FileNode[], depth = 0) => {
    return nodes.map((node, i) => (
      <div key={i} style={{ marginLeft: `${depth * 20}px` }} className="my-1">
        <div className="flex items-center gap-2 text-sm">
          {node.type === "dir" ? (
            <FolderOpen className="w-4 h-4 text-yellow-500" />
          ) : (
            <File className="w-4 h-4 text-blue-500" />
          )}
          <span className="text-slate-700 dark:text-slate-300">{node.name}</span>
          {node.size !== undefined && (
            <span className="text-xs text-slate-500">({node.size} bytes)</span>
          )}
        </div>
        {node.children && renderFileTree(node.children, depth + 1)}
      </div>
    ));
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-emerald-50 to-emerald-100 dark:from-emerald-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        文件系统基本操作
      </h3>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* 操作列表 */}
        <div>
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">选择操作</h4>
          <div className="space-y-2">
            {operations.map((op) => (
              <motion.button
                key={op.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setSelectedOp(op.id)}
                className={`
                  w-full text-left p-3 rounded-lg flex items-center gap-3 transition-all
                  ${selectedOp === op.id
                    ? "bg-emerald-600 text-white shadow-lg"
                    : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-emerald-50 dark:hover:bg-slate-700"
                  }
                `}
              >
                {op.icon}
                <div className="flex-1">
                  <div className="font-medium">{op.name}</div>
                  <div className={`text-xs font-mono mt-1 ${selectedOp === op.id ? "text-emerald-100" : "text-slate-500"}`}>
                    {op.syscall}
                  </div>
                </div>
              </motion.button>
            ))}
          </div>

          {/* 执行按钮 */}
          {selected && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={executeOperation}
              disabled={executing}
              className="mt-4 w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-semibold flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {executing ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Play className="w-5 h-5" />
                  </motion.div>
                  执行中...
                </>
              ) : executed ? (
                <>
                  <CheckCircle className="w-5 h-5" />
                  执行成功
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  执行操作
                </>
              )}
            </motion.button>
          )}
        </div>

        {/* 详情 & 文件树 */}
        <div className="space-y-4">
          {/* 操作说明 */}
          {selected && (
            <div className="p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
              <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">
                系统调用详情
              </h5>
              <div className="p-3 bg-slate-100 dark:bg-slate-900 rounded font-mono text-sm text-slate-800 dark:text-slate-200 mb-3">
                {selected.syscall}
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                {selected.description}
              </p>
            </div>
          )}

          {/* 文件系统状态 */}
          <div className="p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
            <h5 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">
              当前文件系统状态
            </h5>
            <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded">
              {renderFileTree(fileSystem)}
            </div>
          </div>
        </div>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-800">
        <p className="text-sm text-emerald-900 dark:text-emerald-100">
          <strong>提示：</strong> 所有文件操作都通过系统调用完成。应用程序不能直接访问磁盘，必须通过内核提供的接口。
          这保证了文件系统的安全性和一致性。
        </p>
      </div>
    </div>
  );
}
