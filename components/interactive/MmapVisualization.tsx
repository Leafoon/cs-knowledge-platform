"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Map } from "lucide-react";

export default function MmapVisualization() {
  const [mmapType, setMmapType] = useState<"file" | "anonymous">("file");

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Map className="w-7 h-7 text-orange-600" />
        mmap() 内存映射可视化
      </h3>

      <div className="flex justify-center gap-4 mb-6">
        <button onClick={() => setMmapType("file")} className={`px-6 py-3 rounded-lg font-semibold ${mmapType === "file" ? "bg-orange-600 text-white" : "bg-slate-200"}`}>文件映射</button>
        <button onClick={() => setMmapType("anonymous")} className={`px-6 py-3 rounded-lg font-semibold ${mmapType === "anonymous" ? "bg-purple-600 text-white" : "bg-slate-200"}`}>匿名映射</button>
      </div>

      <motion.div key={mmapType} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h4 className="font-bold text-xl mb-4">{mmapType === "file" ? "文件映射 (MAP_SHARED)" : "匿名映射 (MAP_ANONYMOUS)"}</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <div className="bg-blue-100 border-2 border-blue-400 p-4 rounded">
              <div className="font-bold text-blue-800 mb-2">虚拟内存（进程地址空间）</div>
              <div className="space-y-2">
                <div className="bg-white p-2 rounded text-sm">0x7F000000 - 0x7F001000</div>
                <div className="text-xs text-slate-600">4 KB 映射区域</div>
              </div>
            </div>
            {mmapType === "file" && (
              <div className="bg-green-100 border-2 border-green-400 p-4 rounded">
                <div className="font-bold text-green-800 mb-2">磁盘文件</div>
                <div className="bg-white p-2 rounded text-sm font-mono">file.txt (4 KB)</div>
                <div className="text-xs text-slate-600 mt-2">文件内容直接映射到内存</div>
              </div>
            )}
            {mmapType === "anonymous" && (
              <div className="bg-purple-100 border-2 border-purple-400 p-4 rounded">
                <div className="font-bold text-purple-800 mb-2">物理内存（按需分配）</div>
                <div className="text-sm text-slate-700">首次访问时分配物理页，用于堆、共享内存等</div>
              </div>
            )}
          </div>

          <div className="bg-slate-50 p-4 rounded border border-slate-200">
            <div className="font-semibold text-slate-800 mb-3">特性</div>
            <ul className="text-sm text-slate-700 space-y-2 list-disc list-inside">
              {mmapType === "file" ? (
                <>
                  <li>文件内容直接映射到虚拟内存</li>
                  <li>修改内存 = 修改文件（MAP_SHARED）</li>
                  <li>减少 read()/write() 系统调用</li>
                  <li>页缓存自动管理</li>
                </>
              ) : (
                <>
                  <li>不关联任何文件</li>
                  <li>按需分配物理页（懒加载）</li>
                  <li>用于堆扩展、共享内存</li>
                  <li>进程间共享需 MAP_SHARED</li>
                </>
              )}
            </ul>
          </div>
        </div>

        <div className="mt-6 bg-white p-4 rounded border border-slate-200">
          <div className="font-semibold text-slate-800 mb-2">代码示例</div>
          <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto">
{mmapType === "file" ? 
`int fd = open("file.txt", O_RDWR);
void *addr = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                  MAP_SHARED, fd, 0);
// 直接修改内存即修改文件
((char*)addr)[0] = 'H';
munmap(addr, 4096);
close(fd);` :
`void *addr = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
// 匿名内存，可作为堆使用
memset(addr, 0, 4096);
munmap(addr, 4096);`}
          </pre>
        </div>
      </motion.div>

      <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h4 className="font-bold text-amber-800 mb-2">优势</h4>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>零拷贝</strong>：无需 read()/write() 在用户空间和内核空间复制数据</li>
          <li><strong>按需加载</strong>：仅访问的页才加载到内存（懒加载）</li>
          <li><strong>共享内存</strong>：多进程可映射同一文件（MAP_SHARED）</li>
          <li><strong>性能</strong>：大文件 I/O 比传统 read()/write() 快数倍</li>
        </ul>
      </div>
    </div>
  );
}
