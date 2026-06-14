"use client";

import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Folder,
  FolderOpen,
  File,
  FileText,
  FileCode,
  FileImage,
  ChevronRight,
  ChevronDown,
  Home,
  HardDrive,
} from "lucide-react";

interface FSNode {
  name: string;
  type: "dir" | "file";
  inode: number;
  size?: number;
  permissions?: string;
  owner?: string;
  modified?: string;
  hidden?: boolean;
  children?: FSNode[];
  fileType?: string;
}

const fileSystem: FSNode = {
  name: "/",
  type: "dir",
  inode: 2,
  permissions: "drwxr-xr-x",
  owner: "root",
  modified: "2026-01-15",
  children: [
    {
      name: "home",
      type: "dir",
      inode: 131073,
      permissions: "drwxr-xr-x",
      owner: "root",
      modified: "2026-01-10",
      children: [
        {
          name: "user",
          type: "dir",
          inode: 262145,
          permissions: "drwxr-xr-x",
          owner: "user",
          modified: "2026-06-08",
          children: [
            {
              name: "Documents",
              type: "dir",
              inode: 393217,
              permissions: "drwxr-xr-x",
              owner: "user",
              modified: "2026-06-05",
              children: [
                {
                  name: "report.pdf",
                  type: "file",
                  inode: 524289,
                  size: 245760,
                  permissions: "-rw-r--r--",
                  owner: "user",
                  modified: "2026-06-01",
                  fileType: "document",
                },
                {
                  name: "notes.txt",
                  type: "file",
                  inode: 524290,
                  size: 1024,
                  permissions: "-rw-r--r--",
                  owner: "user",
                  modified: "2026-06-05",
                  fileType: "text",
                },
                {
                  name: "budget.xlsx",
                  type: "file",
                  inode: 524291,
                  size: 51200,
                  permissions: "-rw-r--r--",
                  owner: "user",
                  modified: "2026-05-20",
                  fileType: "document",
                },
              ],
            },
            {
              name: "Pictures",
              type: "dir",
              inode: 393218,
              permissions: "drwxr-xr-x",
              owner: "user",
              modified: "2026-06-02",
              children: [
                {
                  name: "photo.jpg",
                  type: "file",
                  inode: 524292,
                  size: 3145728,
                  permissions: "-rw-r--r--",
                  owner: "user",
                  modified: "2026-06-02",
                  fileType: "image",
                },
                {
                  name: "wallpaper.png",
                  type: "file",
                  inode: 524293,
                  size: 5242880,
                  permissions: "-rw-r--r--",
                  owner: "user",
                  modified: "2026-05-15",
                  fileType: "image",
                },
              ],
            },
            {
              name: ".bashrc",
              type: "file",
              inode: 393219,
              size: 3500,
              permissions: "-rw-r--r--",
              owner: "user",
              modified: "2026-01-15",
              hidden: true,
              fileType: "text",
            },
            {
              name: ".profile",
              type: "file",
              inode: 393220,
              size: 807,
              permissions: "-rw-r--r--",
              owner: "user",
              modified: "2026-01-15",
              hidden: true,
              fileType: "text",
            },
          ],
        },
        {
          name: "admin",
          type: "dir",
          inode: 262146,
          permissions: "drwxr-xr-x",
          owner: "admin",
          modified: "2026-04-20",
          children: [
            {
              name: "config.json",
              type: "file",
              inode: 524294,
              size: 2048,
              permissions: "-rw-------",
              owner: "admin",
              modified: "2026-04-20",
              fileType: "code",
            },
          ],
        },
      ],
    },
    {
      name: "etc",
      type: "dir",
      inode: 131074,
      permissions: "drwxr-xr-x",
      owner: "root",
      modified: "2026-03-01",
      children: [
        {
          name: "passwd",
          type: "file",
          inode: 262147,
          size: 2048,
          permissions: "-rw-r--r--",
          owner: "root",
          modified: "2026-03-01",
          fileType: "text",
        },
        {
          name: "fstab",
          type: "file",
          inode: 262148,
          size: 512,
          permissions: "-rw-r--r--",
          owner: "root",
          modified: "2026-01-10",
          fileType: "text",
        },
      ],
    },
    {
      name: "var",
      type: "dir",
      inode: 131075,
      permissions: "drwxr-xr-x",
      owner: "root",
      modified: "2026-06-08",
      children: [
        {
          name: "log",
          type: "dir",
          inode: 262149,
          permissions: "drwxr-xr-x",
          owner: "root",
          modified: "2026-06-08",
          children: [
            {
              name: "syslog",
              type: "file",
              inode: 393221,
              size: 1048576,
              permissions: "-rw-r-----",
              owner: "root",
              modified: "2026-06-08",
              fileType: "text",
            },
          ],
        },
      ],
    },
    {
      name: "tmp",
      type: "dir",
      inode: 131076,
      permissions: "drwxrwxrwt",
      owner: "root",
      modified: "2026-06-08",
      children: [],
    },
  ],
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${(bytes / 1073741824).toFixed(1)} GB`;
}

function getFileIcon(node: FSNode) {
  if (node.type === "dir") return null;
  switch (node.fileType) {
    case "image":
      return <FileImage className="w-4 h-4 text-pink-500" />;
    case "code":
      return <FileCode className="w-4 h-4 text-emerald-500" />;
    case "document":
      return <FileText className="w-4 h-4 text-blue-500" />;
    default:
      return <File className="w-4 h-4 text-slate-500" />;
  }
}

interface TreeNodeProps {
  node: FSNode;
  depth: number;
  path: string[];
  expanded: Set<string>;
  toggleExpand: (path: string) => void;
  selectedFile: string | null;
  setSelectedFile: (path: string | null) => void;
  setPathSegments: (segments: string[]) => void;
}

function TreeNode({
  node,
  depth,
  path,
  expanded,
  toggleExpand,
  selectedFile,
  setSelectedFile,
  setPathSegments,
}: TreeNodeProps) {
  const currentPath = [...path, node.name].join("/");
  const isExpanded = expanded.has(currentPath);
  const isSelected = selectedFile === currentPath;
  const isDir = node.type === "dir";

  const handleClick = () => {
    if (isDir) {
      toggleExpand(currentPath);
      setPathSegments([...path, node.name]);
    } else {
      setSelectedFile(isSelected ? null : currentPath);
      setPathSegments([...path, node.name]);
    }
  };

  // Sort: directories first, then files; hidden files last
  const sortedChildren = node.children
    ? [...node.children].sort((a, b) => {
        if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
        if (a.hidden !== b.hidden) return a.hidden ? 1 : -1;
        return a.name.localeCompare(b.name);
      })
    : [];

  return (
    <div>
      <motion.div
        className={`flex items-center gap-1 py-1 px-2 rounded cursor-pointer transition-colors group ${
          isSelected
            ? "bg-indigo-100 dark:bg-indigo-900/40"
            : "hover:bg-slate-100 dark:hover:bg-gray-700/50"
        }`}
        style={{ paddingLeft: `${depth * 20 + 8}px` }}
        onClick={handleClick}
        whileTap={{ scale: 0.99 }}
      >
        {/* Expand/collapse icon for directories */}
        {isDir ? (
          <motion.div
            animate={{ rotate: isExpanded ? 90 : 0 }}
            transition={{ duration: 0.15 }}
            className="w-4 h-4 flex-shrink-0"
          >
            <ChevronRight className="w-4 h-4 text-slate-400" />
          </motion.div>
        ) : (
          <div className="w-4 h-4 flex-shrink-0" />
        )}

        {/* File/folder icon */}
        {isDir ? (
          isExpanded ? (
            <FolderOpen className="w-4 h-4 text-blue-500 flex-shrink-0" />
          ) : (
            <Folder className="w-4 h-4 text-blue-500 flex-shrink-0" />
          )
        ) : (
          <div className="flex-shrink-0">{getFileIcon(node)}</div>
        )}

        {/* Name */}
        <span
          className={`text-sm font-mono ${
            node.hidden
              ? "text-slate-400 dark:text-gray-500"
              : isDir
              ? "text-blue-700 dark:text-blue-300 font-medium"
              : "text-slate-700 dark:text-gray-200"
          }`}
        >
          {node.name}
          {isDir && node.name !== "/" && node.name !== "." && node.name !== ".." ? "/" : ""}
        </span>

        {/* . and .. highlight */}
        {(node.name === "." || node.name === "..") && (
          <span className="text-xs px-1.5 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded ml-1">
            special
          </span>
        )}

        {/* Inode number */}
        <span className="text-xs text-slate-400 dark:text-gray-500 ml-auto font-mono opacity-0 group-hover:opacity-100 transition-opacity">
          inode:{node.inode}
        </span>
      </motion.div>

      {/* Children */}
      <AnimatePresence>
        {isDir && isExpanded && sortedChildren.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            {/* . entry */}
            <div
              className="flex items-center gap-1 py-1 px-2"
              style={{ paddingLeft: `${(depth + 1) * 20 + 8}px` }}
            >
              <div className="w-4 h-4" />
              <Folder className="w-4 h-4 text-slate-400" />
              <span className="text-sm font-mono text-amber-600 dark:text-amber-400">.</span>
              <span className="text-xs px-1.5 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded">
                self
              </span>
              <span className="text-xs text-slate-400 dark:text-gray-500 ml-auto font-mono">
                inode:{node.inode}
              </span>
            </div>

            {/* .. entry */}
            {depth > 0 && (
              <div
                className="flex items-center gap-1 py-1 px-2"
                style={{ paddingLeft: `${(depth + 1) * 20 + 8}px` }}
              >
                <div className="w-4 h-4" />
                <Folder className="w-4 h-4 text-slate-400" />
                <span className="text-sm font-mono text-amber-600 dark:text-amber-400">..</span>
                <span className="text-xs px-1.5 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded">
                  parent
                </span>
              </div>
            )}

            {sortedChildren.map((child) => (
              <TreeNode
                key={child.name}
                node={child}
                depth={depth + 1}
                path={[...path, node.name]}
                expanded={expanded}
                toggleExpand={toggleExpand}
                selectedFile={selectedFile}
                setSelectedFile={setSelectedFile}
                setPathSegments={setPathSegments}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function DirectoryTreeVisualizer() {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(["/"]));
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [pathSegments, setPathSegments] = useState<string[]>(["/"]);

  const toggleExpand = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const selectedNode = useMemo(() => {
    if (!selectedFile) return null;
    const segments = selectedFile.split("/").filter(Boolean);

    function findNode(node: FSNode, segs: string[]): FSNode | null {
      if (segs.length === 0) return node;
      if (!node.children) return null;
      const child = node.children.find((c) => c.name === segs[0]);
      if (!child) return null;
      return findNode(child, segs.slice(1));
    }

    return findNode(fileSystem, segments);
  }, [selectedFile]);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-6 text-center">
        Directory Tree Visualizer
      </h2>

      {/* Path bar */}
      <div className="flex items-center gap-1 mb-4 p-3 bg-white dark:bg-gray-800 rounded-lg border border-slate-200 dark:border-gray-700 overflow-x-auto">
        <HardDrive className="w-4 h-4 text-slate-500 flex-shrink-0" />
        {pathSegments.map((seg, i) => (
          <React.Fragment key={i}>
            {i > 0 && <ChevronRight className="w-3 h-3 text-slate-400 flex-shrink-0" />}
            <button
              className="text-sm font-mono text-slate-600 dark:text-gray-300 hover:text-indigo-500 dark:hover:text-indigo-400 transition-colors whitespace-nowrap"
              onClick={() => {
                const newPath = pathSegments.slice(0, i + 1);
                setPathSegments(newPath);
              }}
            >
              {seg === "/" ? (
                <Home className="w-3.5 h-3.5 inline" />
              ) : (
                seg
              )}
            </button>
          </React.Fragment>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Tree view */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 overflow-auto max-h-[500px]">
          <TreeNode
            node={fileSystem}
            depth={0}
            path={[]}
            expanded={expanded}
            toggleExpand={toggleExpand}
            selectedFile={selectedFile}
            setSelectedFile={setSelectedFile}
            setPathSegments={setPathSegments}
          />
        </div>

        {/* Details panel */}
        <div className="lg:col-span-1">
          <AnimatePresence mode="wait">
            {selectedNode ? (
              <motion.div
                key={selectedNode.inode}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700"
              >
                <div className="flex items-center gap-2 mb-4">
                  {selectedNode.type === "dir" ? (
                    <Folder className="w-6 h-6 text-blue-500" />
                  ) : (
                    getFileIcon(selectedNode) || <File className="w-6 h-6 text-slate-500" />
                  )}
                  <h3 className="text-lg font-bold text-slate-800 dark:text-gray-100">
                    {selectedNode.name}
                  </h3>
                </div>

                <div className="space-y-3">
                  <DetailRow label="Type" value={selectedNode.type === "dir" ? "Directory" : "Regular File"} />
                  <DetailRow label="Inode" value={String(selectedNode.inode)} mono />
                  {selectedNode.size !== undefined && (
                    <DetailRow label="Size" value={formatSize(selectedNode.size)} />
                  )}
                  {selectedNode.permissions && (
                    <DetailRow label="Permissions" value={selectedNode.permissions} mono />
                  )}
                  {selectedNode.owner && (
                    <DetailRow label="Owner" value={selectedNode.owner} />
                  )}
                  {selectedNode.modified && (
                    <DetailRow label="Modified" value={selectedNode.modified} />
                  )}
                  {selectedNode.hidden !== undefined && (
                    <DetailRow label="Hidden" value={selectedNode.hidden ? "Yes" : "No"} />
                  )}
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow-md border border-slate-200 dark:border-gray-700 text-center"
              >
                <Folder className="w-10 h-10 text-slate-300 dark:text-gray-600 mx-auto mb-3" />
                <p className="text-sm text-slate-500 dark:text-gray-400">
                  Click on a file or folder to view its inode details.
                </p>
                <p className="text-xs text-slate-400 dark:text-gray-500 mt-2">
                  Directories can be expanded/collapsed by clicking.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Color legend */}
          <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4 shadow-md border border-slate-200 dark:border-gray-700">
            <h4 className="text-sm font-semibold text-slate-700 dark:text-gray-200 mb-2">Legend</h4>
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center gap-2">
                <Folder className="w-3.5 h-3.5 text-blue-500" />
                <span className="text-slate-600 dark:text-gray-300">Directory (blue)</span>
              </div>
              <div className="flex items-center gap-2">
                <File className="w-3.5 h-3.5 text-slate-500" />
                <span className="text-slate-600 dark:text-gray-300">Regular file (green/gray)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3.5 h-3.5 rounded bg-amber-400/50" />
                <span className="text-slate-600 dark:text-gray-300">Special entries (., ..)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-400 dark:text-gray-500 text-xs font-mono">.hidden</span>
                <span className="text-slate-600 dark:text-gray-300">Hidden files (gray)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DetailRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-slate-500 dark:text-gray-400">{label}</span>
      <span className={`text-sm text-slate-800 dark:text-gray-100 ${mono ? "font-mono" : ""}`}>
        {value}
      </span>
    </div>
  );
}
