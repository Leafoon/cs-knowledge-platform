"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, ChevronDown } from "lucide-react";

interface Node {
  id: string;
  label: string;
  children?: Node[];
}

const tree: Node = {
  id: "root",
  label: "计算机组成原理",
  children: [
    {
      id: "ch1",
      label: "系统概述",
      children: [
        { id: "ch1-1", label: "发展历程" },
        { id: "ch1-2", label: "层次结构" },
        { id: "ch1-3", label: "性能指标" },
      ],
    },
    {
      id: "ch2",
      label: "数据表示与运算",
      children: [
        { id: "ch2-1", label: "进制转换" },
        { id: "ch2-2", label: "定点运算" },
        { id: "ch2-3", label: "浮点运算" },
        { id: "ch2-4", label: "ALU设计" },
      ],
    },
    {
      id: "ch3",
      label: "存储系统",
      children: [
        { id: "ch3-1", label: "主存" },
        { id: "ch3-2", label: "Cache" },
        { id: "ch3-3", label: "虚拟存储" },
      ],
    },
    {
      id: "ch4",
      label: "指令系统",
      children: [
        { id: "ch4-1", label: "指令格式" },
        { id: "ch4-2", label: "寻址方式" },
        { id: "ch4-3", label: "CISC/RISC" },
      ],
    },
    {
      id: "ch5",
      label: "CPU",
      children: [
        { id: "ch5-1", label: "数据通路" },
        { id: "ch5-2", label: "控制器" },
        { id: "ch5-3", label: "流水线" },
      ],
    },
    {
      id: "ch6",
      label: "总线与I/O",
      children: [
        { id: "ch6-1", label: "总线" },
        { id: "ch6-2", label: "I/O方式" },
        { id: "ch6-3", label: "中断/DMA" },
      ],
    },
  ],
};

const colorMap: Record<string, string> = {
  root: "#667eea",
  ch1: "#f59e0b",
  ch2: "#10b981",
  ch3: "#ef4444",
  ch4: "#8b5cf6",
  ch5: "#ec4899",
  ch6: "#06b6d4",
};

function getParentColor(id: string): string {
  const prefix = id.split("-")[0];
  return colorMap[prefix] || colorMap.root;
}

function MapNode({ node, depth }: { node: Node; depth: number }) {
  const [open, setOpen] = useState(depth < 2);
  const hasChildren = node.children && node.children.length > 0;
  const color = depth === 0 ? colorMap.root : getParentColor(node.id);

  return (
    <div>
      <motion.button
        onClick={() => hasChildren && setOpen(!open)}
        className="flex items-center gap-1.5 py-1 text-sm text-text-primary hover:text-accent-primary transition-colors"
        style={{ paddingLeft: depth * 20 }}
        whileTap={{ scale: 0.98 }}
      >
        {hasChildren ? (
          <motion.span animate={{ rotate: open ? 90 : 0 }} transition={{ duration: 0.15 }}>
            <ChevronRight size={14} style={{ color }} />
          </motion.span>
        ) : (
          <span className="w-3.5 h-3.5 flex items-center justify-center">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
          </span>
        )}
        <span className={depth === 0 ? "font-bold text-base" : depth === 1 ? "font-semibold" : ""}>
          {node.label}
        </span>
      </motion.button>
      <AnimatePresence>
        {hasChildren && open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            {node.children!.map((child) => (
              <MapNode key={child.id} node={child} depth={depth + 1} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function ComputerOrganizationMindMap() {
  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        课程知识脉络思维导图
      </h3>
      <p className="text-sm text-text-secondary mb-4">
        点击节点展开/折叠，浏览课程知识结构
      </p>
      <div className="rounded-lg border border-border-subtle bg-bg-secondary p-4">
        <MapNode node={tree} depth={0} />
      </div>
    </div>
  );
}
