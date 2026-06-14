"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { FileText, ArrowRight } from "lucide-react";

export default function FileIOWorkflow() {
  const [operation, setOperation] = useState<"read" | "write">("read");

  const workflows = {
    read: [
      { step: "open()", desc: "打开文件，返回 fd", kernel: "查找 inode，分配文件表项" },
      { step: "read()", desc: "读取数据到缓冲区", kernel: "从页缓存/磁盘读取，复制到用户空间" },
      { step: "close()", desc: "关闭文件描述符", kernel: "释放文件表项，引用计数 -1" }
    ],
    write: [
      { step: "open()", desc: "打开文件（O_WRONLY | O_CREAT）", kernel: "创建 inode（如不存在），分配文件表项" },
      { step: "write()", desc: "写入数据", kernel: "复制到页缓存，标记脏页，稍后回写" },
      { step: "close()", desc: "关闭文件描述符", kernel: "触发脏页回写，释放文件表项" }
    ]
  };

  const current = workflows[operation];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <FileText className="w-7 h-7 text-green-600" />
        文件 I/O 工作流程
      </h3>

      <div className="flex justify-center gap-4 mb-6">
        <button onClick={() => setOperation("read")} className={`px-6 py-3 rounded-lg font-semibold ${operation === "read" ? "bg-blue-600 text-white" : "bg-slate-200"}`}>读取文件</button>
        <button onClick={() => setOperation("write")} className={`px-6 py-3 rounded-lg font-semibold ${operation === "write" ? "bg-green-600 text-white" : "bg-slate-200"}`}>写入文件</button>
      </div>

      <div className="space-y-4">
        {current.map((item, idx) => (
          <motion.div key={idx} initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: idx * 0.2 }} className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center gap-4">
              <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white text-lg ${operation === "read" ? "bg-blue-600" : "bg-green-600"}`}>{idx + 1}</div>
              <div className="flex-1">
                <div className="font-bold text-slate-800 text-lg mb-1">{item.step}</div>
                <div className="text-sm text-slate-600 mb-2">{item.desc}</div>
                <div className="bg-slate-100 p-2 rounded border border-slate-200 text-sm text-slate-700">
                  <span className="font-semibold">内核操作：</span>{item.kernel}
                </div>
              </div>
              {idx < current.length - 1 && <ArrowRight className="w-6 h-6 text-slate-400" />}
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-6 bg-white rounded-lg shadow-md p-4">
        <h4 className="font-bold mb-2">代码示例</h4>
        <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto">
{operation === "read" ? `int fd = open("file.txt", O_RDONLY);
char buf[1024];
ssize_t n = read(fd, buf, sizeof(buf));
close(fd);` : `int fd = open("file.txt", O_WRONLY | O_CREAT, 0644);
const char *data = "Hello, World!";
write(fd, data, strlen(data));
close(fd);`}
        </pre>
      </div>
    </div>
  );
}
