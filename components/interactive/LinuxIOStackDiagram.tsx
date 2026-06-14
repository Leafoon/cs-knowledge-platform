"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, ArrowDown, Info, ChevronDown, ChevronUp, Cpu, HardDrive, FileText, Monitor } from "lucide-react";

interface IOLayer {
  id: string;
  name: string;
  subtitle: string;
  icon: React.ReactNode;
  color: string;
  description: string;
  keyPoints: string[];
  example: string;
}

const LAYERS: IOLayer[] = [
  {
    id: "app",
    name: "用户应用 (User Application)",
    subtitle: "read() / write() / open() 系统调用",
    icon: <Monitor className="w-5 h-5" />,
    color: "blue",
    description: "应用程序通过系统调用接口发起 I/O 请求。POSIX 定义了标准文件操作 API，应用无需关心底层存储细节。",
    keyPoints: [
      "调用 read()/write() 等 POSIX API",
      "使用文件描述符 (fd) 标识文件",
      "可以是缓冲 I/O 或直接 I/O",
    ],
    example: "int fd = open(\"/data/file.txt\", O_RDONLY);\nread(fd, buf, 4096);",
  },
  {
    id: "syscall",
    name: "系统调用接口 (System Call Interface)",
    subtitle: "用户态 → 内核态切换",
    icon: <Cpu className="w-5 h-5" />,
    color: "indigo",
    description: "系统调用是用户态进入内核态的门户。内核验证参数合法性后，将请求转发给 VFS 层。涉及上下文切换和权限检查。",
    keyPoints: [
      "触发 trap/中断进入内核态",
      "验证文件描述符和缓冲区指针",
      "copy_from_user() / copy_to_user() 数据拷贝",
    ],
    example: "SYSCALL_DEFINE3(read, unsigned int, fd, char __user *, buf, size_t, count)",
  },
  {
    id: "vfs",
    name: "虚拟文件系统 (VFS)",
    subtitle: "统一抽象层：inode, dentry, superblock, file",
    icon: <FileText className="w-5 h-5" />,
    color: "purple",
    description: "VFS 为不同文件系统（ext4, xfs, btrfs 等）提供统一接口。通过 inode/dentry/superblock/file 四大对象抽象文件操作。",
    keyPoints: [
      "定义统一的文件操作接口 (file_operations)",
      "管理 inode 缓存、dentry 缓存",
      "路径解析：目录项逐级查找",
      "支持多种文件系统共存",
    ],
    example: "file->f_op->read_iter() → 调用具体文件系统的读实现",
  },
  {
    id: "fs",
    name: "文件系统 (Filesystem)",
    subtitle: "ext4 / XFS / Btrfs 等",
    icon: <FileText className="w-5 h-5" />,
    color: "green",
    description: "具体文件系统将逻辑文件操作转换为块设备上的物理位置。负责空间分配、元数据管理、日志一致性等。",
    keyPoints: [
      "逻辑块地址 → 物理块地址映射",
      "管理位图、inode 表、日志 (journal)",
      "处理文件扩展、截断、碎片整理",
      "日志保证崩溃一致性 (crash consistency)",
    ],
    example: "ext4_readpages() → 确定磁盘块位置，构建 bio 请求",
  },
  {
    id: "pagecache",
    name: "页缓存 (Page Cache)",
    subtitle: "内存中的文件数据缓存",
    icon: <Cpu className="w-5 h-5" />,
    color: "teal",
    description: "页缓存缓存最近访问的文件数据页，避免重复磁盘 I/O。readahead 预读机制提前加载后续数据。dirty 页由后台线程回写。",
    keyPoints: [
      "缓存热点数据，减少磁盘访问",
      "Readahead 预读提升顺序读性能",
      "Dirty page 回写：定期或内存压力时",
      "Page Lock 防止并发读写冲突",
    ],
    example: "find_get_page() → 命中则直接返回，未命中则从磁盘读取",
  },
  {
    id: "block",
    name: "块层 (Block Layer)",
    subtitle: "bio 请求合并、排队、调度",
    icon: <Layers className="w-5 h-5" />,
    color: "orange",
    description: "块层接收 bio（块 I/O）请求，进行请求合并（merge）和排序，通过 I/O 调度器优化请求顺序，减少磁头移动。",
    keyPoints: [
      "bio 合并：相邻请求合并减少 I/O 次数",
      "I/O 调度器：MQ-Deadline / BFQ / Kyber / None",
      "请求队列管理 (request_queue)",
      "多队列架构 (blk-mq) 提升并行性",
    ],
    example: "submit_bio() → blk_mq_submit_bio() → 调度器派发",
  },
  {
    id: "scheduler",
    name: "I/O 调度器 (I/O Scheduler)",
    subtitle: "MQ-Deadline / BFQ / Kyber",
    icon: <Cpu className="w-5 h-5" />,
    color: "yellow",
    description: "I/O 调度器决定请求的派发顺序，在吞吐量和延迟之间权衡。对 HDD 减少寻道，对 SSD 减少队列深度竞争。",
    keyPoints: [
      "MQ-Deadline：防止请求饥饿，适合 HDD",
      "BFQ：公平带宽分配，适合桌面",
      "Kyber：令牌桶控制，适合快速 SSD",
      "None：直接派发，适合 NVMe SSD",
    ],
    example: "调度器根据 LBA 排序 → 派发到设备驱动",
  },
  {
    id: "driver",
    name: "设备驱动 (Device Driver)",
    subtitle: "SCSI / NVMe / ATA 驱动",
    icon: <HardDrive className="w-5 h-5" />,
    color: "red",
    description: "设备驱动将标准块请求转换为设备特定的命令协议（如 SCSI CDB、NVMe 命令）。管理 DMA 传输和中断处理。",
    keyPoints: [
      "构建设备命令（SCBI CDB / NVMe CMD）",
      "DMA 映射：建立内存 ↔ 设备数据通道",
      "中断处理：完成回调和错误处理",
      "电源管理和错误恢复",
    ],
    example: "nvme_queue_rq() → 写入 NVMe 提交队列 SQ",
  },
  {
    id: "hw",
    name: "硬件 (Hardware)",
    subtitle: "HDD / SSD / NVMe / RAID 控制器",
    icon: <HardDrive className="w-5 h-5" />,
    color: "gray",
    description: "物理存储设备执行实际的数据读写。HDD 涉及寻道和旋转，SSD 涉及闪存转换层 (FTL)，NVMe 通过 PCIe 直连高效传输。",
    keyPoints: [
      "HDD：磁头寻道 → 旋转等待 → 数据传输",
      "SSD：FTL 地址映射 → NAND 闪存读写",
      "NVMe：PCIe 直连，多队列高并行",
      "完成中断 → 通知上层 I/O 完成",
    ],
    example: "设备完成 DMA 传输 → 触发完成中断 → 唤醒等待进程",
  },
];

const COLOR_MAP: Record<string, { bg: string; bgHover: string; border: string; text: string; light: string; dark: string }> = {
  blue: { bg: "bg-blue-500", bgHover: "hover:bg-blue-50 dark:hover:bg-blue-900/20", border: "border-blue-300 dark:border-blue-700", text: "text-blue-700 dark:text-blue-300", light: "bg-blue-50 dark:bg-blue-900/20", dark: "border-blue-200 dark:border-blue-800" },
  indigo: { bg: "bg-indigo-500", bgHover: "hover:bg-indigo-50 dark:hover:bg-indigo-900/20", border: "border-indigo-300 dark:border-indigo-700", text: "text-indigo-700 dark:text-indigo-300", light: "bg-indigo-50 dark:bg-indigo-900/20", dark: "border-indigo-200 dark:border-indigo-800" },
  purple: { bg: "bg-purple-500", bgHover: "hover:bg-purple-50 dark:hover:bg-purple-900/20", border: "border-purple-300 dark:border-purple-700", text: "text-purple-700 dark:text-purple-300", light: "bg-purple-50 dark:bg-purple-900/20", dark: "border-purple-200 dark:border-purple-800" },
  green: { bg: "bg-green-500", bgHover: "hover:bg-green-50 dark:hover:bg-green-900/20", border: "border-green-300 dark:border-green-700", text: "text-green-700 dark:text-green-300", light: "bg-green-50 dark:bg-green-900/20", dark: "border-green-200 dark:border-green-800" },
  teal: { bg: "bg-teal-500", bgHover: "hover:bg-teal-50 dark:hover:bg-teal-900/20", border: "border-teal-300 dark:border-teal-700", text: "text-teal-700 dark:text-teal-300", light: "bg-teal-50 dark:bg-teal-900/20", dark: "border-teal-200 dark:border-teal-800" },
  orange: { bg: "bg-orange-500", bgHover: "hover:bg-orange-50 dark:hover:bg-orange-900/20", border: "border-orange-300 dark:border-orange-700", text: "text-orange-700 dark:text-orange-300", light: "bg-orange-50 dark:bg-orange-900/20", dark: "border-orange-200 dark:border-orange-800" },
  yellow: { bg: "bg-yellow-500", bgHover: "hover:bg-yellow-50 dark:hover:bg-yellow-900/20", border: "border-yellow-300 dark:border-yellow-700", text: "text-yellow-700 dark:text-yellow-300", light: "bg-yellow-50 dark:bg-yellow-900/20", dark: "border-yellow-200 dark:border-yellow-800" },
  red: { bg: "bg-red-500", bgHover: "hover:bg-red-50 dark:hover:bg-red-900/20", border: "border-red-300 dark:border-red-700", text: "text-red-700 dark:text-red-300", light: "bg-red-50 dark:bg-red-900/20", dark: "border-red-200 dark:border-red-800" },
  gray: { bg: "bg-gray-500", bgHover: "hover:bg-gray-50 dark:hover:bg-gray-900/20", border: "border-gray-300 dark:border-gray-700", text: "text-gray-700 dark:text-gray-300", light: "bg-gray-50 dark:bg-gray-900/20", dark: "border-gray-200 dark:border-gray-800" },
};

export function LinuxIOStackDiagram() {
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [expandAll, setExpandAll] = useState(false);

  const toggleLayer = (id: string) => {
    setSelectedLayer(selectedLayer === id ? null : id);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Layers className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
          <h3 className="text-lg font-bold text-text-primary">Linux I/O 栈</h3>
        </div>
        <button
          onClick={() => { setExpandAll(!expandAll); setSelectedLayer(null); }}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm bg-gray-100 dark:bg-gray-800 text-text-secondary hover:bg-gray-200 dark:hover:bg-gray-700"
        >
          {expandAll ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          {expandAll ? "折叠全部" : "展开全部"}
        </button>
      </div>

      <div className="text-sm text-text-secondary mb-2">
        点击各层查看详细说明
      </div>

      <div className="space-y-1">
        {LAYERS.map((layer, index) => {
          const c = COLOR_MAP[layer.color];
          const isExpanded = expandAll || selectedLayer === layer.id;

          return (
            <div key={layer.id}>
              <motion.button
                onClick={() => toggleLayer(layer.id)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all ${c.bgHover} ${
                  isExpanded ? `${c.light} ${c.border}` : "border-gray-200 dark:border-gray-700"
                }`}
                whileHover={{ x: 4 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <div className={`w-8 h-8 rounded-lg ${c.bg} text-white flex items-center justify-center flex-shrink-0`}>
                  {layer.icon}
                </div>
                <div className="flex-1 text-left">
                  <div className="font-semibold text-sm text-text-primary">{layer.name}</div>
                  <div className="text-xs text-text-secondary">{layer.subtitle}</div>
                </div>
                {isExpanded ? (
                  <ChevronUp className="w-4 h-4 text-text-secondary flex-shrink-0" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-text-secondary flex-shrink-0" />
                )}
              </motion.button>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className={`ml-11 p-4 mb-1 rounded-lg ${c.light} border ${c.dark}`}>
                      <p className="text-sm text-text-secondary mb-3">{layer.description}</p>

                      <div className="mb-3">
                        <div className="text-xs font-semibold text-text-primary mb-1.5">关键要点</div>
                        <ul className="space-y-1">
                          {layer.keyPoints.map((point, i) => (
                            <motion.li
                              key={i}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: i * 0.05 }}
                              className="flex items-start gap-2 text-xs text-text-secondary"
                            >
                              <span className={`w-1.5 h-1.5 rounded-full ${c.bg} mt-1 flex-shrink-0`} />
                              {point}
                            </motion.li>
                          ))}
                        </ul>
                      </div>

                      <div className="p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700">
                        <div className="text-xs text-text-secondary mb-1">示例</div>
                        <pre className="text-xs font-mono text-text-primary whitespace-pre-wrap">{layer.example}</pre>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {index < LAYERS.length - 1 && (
                <div className="flex justify-center py-0.5">
                  <ArrowDown className="w-4 h-4 text-gray-300 dark:text-gray-600" />
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-text-secondary">
            <p><strong className="text-text-primary">Linux I/O 路径</strong>从用户空间系统调用开始，经过 VFS 抽象、文件系统映射、页缓存优化、块层调度，最终到达设备驱动和硬件。每一层都有缓存和优化策略。</p>
          </div>
        </div>
      </div>
    </div>
  );
}
