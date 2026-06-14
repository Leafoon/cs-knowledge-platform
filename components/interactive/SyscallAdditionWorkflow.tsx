"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Plus, FileCode } from "lucide-react";

export default function SyscallAdditionWorkflow() {
  const [step, setStep] = useState(0);

  const steps = [
    { 
      title: "1. 定义系统调用编号", 
      file: "include/linux/syscalls.h", 
      code: `// 在文件末尾添加
#define __NR_mycall 335  // x86-64 下一个可用编号

asmlinkage long sys_mycall(int arg1, void __user *arg2);`,
      desc: "在系统调用表中分配唯一编号"
    },
    { 
      title: "2. 实现系统调用函数", 
      file: "kernel/sys.c", 
      code: `#include <linux/syscalls.h>
#include <linux/uaccess.h>

SYSCALL_DEFINE2(mycall, int, arg1, void __user *, arg2) {
  char buf[256];
  
  // 从用户空间安全拷贝数据
  if (copy_from_user(buf, arg2, sizeof(buf)))
    return -EFAULT;
  
  printk(KERN_INFO "mycall: arg1=%d, arg2=%s\\n", arg1, buf);
  
  return 0;  // 成功返回 0
}`,
      desc: "实现系统调用逻辑，注意安全性（验证用户指针）"
    },
    { 
      title: "3. 注册到系统调用表", 
      file: "arch/x86/entry/syscalls/syscall_64.tbl", 
      code: `# 在文件末尾添加
335    common    mycall          sys_mycall`,
      desc: "将系统调用编号与实现函数关联"
    },
    { 
      title: "4. 重新编译内核", 
      file: "终端", 
      code: `make -j$(nproc)
make modules_install
make install
reboot`,
      desc: "编译内核并重启系统"
    },
    { 
      title: "5. 用户空间调用", 
      file: "test_mycall.c", 
      code: `#include <unistd.h>
#include <sys/syscall.h>
#include <stdio.h>

#define __NR_mycall 335

int main() {
  char msg[] = "Hello from user space!";
  long ret = syscall(__NR_mycall, 42, msg);
  
  printf("syscall returned: %ld\\n", ret);
  return 0;
}

// 编译运行
// gcc test_mycall.c -o test_mycall
// ./test_mycall
// dmesg | tail  // 查看内核日志`,
      desc: "使用 syscall() 函数直接调用新系统调用"
    }
  ];

  const current = steps[step];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center flex items-center justify-center gap-2">
        <Plus className="w-7 h-7 text-amber-600" />
        添加自定义系统调用
      </h3>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h4 className="font-bold text-xl text-slate-800">{current.title}</h4>
            <p className="text-sm text-slate-600 mt-1">{current.desc}</p>
          </div>
          <div className="text-2xl font-bold text-amber-600">步骤 {step + 1}/5</div>
        </div>

        <motion.div key={step} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-gradient-to-r from-amber-100 to-orange-100 rounded-lg p-6 mb-6 border-2 border-amber-300">
          <div className="flex items-center gap-2 mb-4">
            <FileCode className="w-5 h-5 text-amber-700" />
            <div className="font-semibold text-amber-900">{current.file}</div>
          </div>
          <pre className="bg-slate-900 text-green-400 p-4 rounded text-sm overflow-x-auto whitespace-pre-wrap">{current.code}</pre>
        </motion.div>

        <div className="flex justify-center gap-4">
          <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} className="px-6 py-2 bg-slate-300 rounded-lg font-semibold disabled:opacity-50">上一步</button>
          <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step === steps.length - 1} className="px-6 py-2 bg-amber-600 text-white rounded-lg font-semibold disabled:opacity-50">下一步</button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">关键宏</h5>
          <div className="space-y-3 text-sm">
            <div className="bg-blue-50 p-3 rounded border border-blue-200">
              <div className="font-mono text-blue-800 mb-1">SYSCALL_DEFINE2(name, type1, arg1, type2, arg2)</div>
              <div className="text-xs text-slate-600">定义接受 2 个参数的系统调用</div>
            </div>
            <div className="bg-green-50 p-3 rounded border border-green-200">
              <div className="font-mono text-green-800 mb-1">copy_from_user(to, from, n)</div>
              <div className="text-xs text-slate-600">从用户空间安全拷贝数据（自动验证指针）</div>
            </div>
            <div className="bg-purple-50 p-3 rounded border border-purple-200">
              <div className="font-mono text-purple-800 mb-1">copy_to_user(to, from, n)</div>
              <div className="text-xs text-slate-600">向用户空间安全写入数据</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-4">
          <h5 className="font-bold text-slate-800 mb-3">错误码</h5>
          <table className="w-full text-sm">
            <thead><tr className="border-b"><th className="text-left py-2">错误码</th><th className="text-left py-2">含义</th></tr></thead>
            <tbody>
              <tr className="border-b"><td className="py-2 font-mono">-EFAULT</td><td>用户指针无效</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">-EINVAL</td><td>参数无效</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">-ENOMEM</td><td>内存不足</td></tr>
              <tr className="border-b"><td className="py-2 font-mono">-EPERM</td><td>权限不足</td></tr>
              <tr><td className="py-2 font-mono">-ENOSYS</td><td>系统调用未实现</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h5 className="font-bold text-slate-800 mb-4">完整调用流程</h5>
        <div className="flex items-center justify-center gap-2 text-sm flex-wrap">
          <div className="bg-blue-100 px-4 py-2 rounded font-semibold">用户调用 syscall(335, ...)</div>
          <div>→</div>
          <div className="bg-green-100 px-4 py-2 rounded font-semibold">glibc 设置寄存器</div>
          <div>→</div>
          <div className="bg-purple-100 px-4 py-2 rounded font-semibold">syscall 指令</div>
          <div>→</div>
          <div className="bg-orange-100 px-4 py-2 rounded font-semibold">内核查表 [335]</div>
          <div>→</div>
          <div className="bg-red-100 px-4 py-2 rounded font-semibold">sys_mycall()</div>
          <div>→</div>
          <div className="bg-pink-100 px-4 py-2 rounded font-semibold">返回用户态</div>
        </div>
      </div>

      <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded">
        <h5 className="font-bold text-amber-800 mb-2">注意事项</h5>
        <ul className="text-sm text-slate-700 space-y-1 list-disc list-inside">
          <li><strong>编号唯一性</strong>：不同架构可能需要在多个表中添加（x86、ARM、RISC-V）</li>
          <li><strong>指针验证</strong>：必须使用 copy_from_user/copy_to_user，不能直接解引用用户指针</li>
          <li><strong>ABI 稳定性</strong>：系统调用接口一旦发布不应修改（向后兼容）</li>
          <li><strong>权限检查</strong>：必要时使用 capable() 检查权限</li>
          <li><strong>返回值</strong>：成功返回 ≥0，失败返回 -errno</li>
          <li>生产环境不建议修改内核，应使用内核模块或 eBPF</li>
        </ul>
      </div>
    </div>
  );
}
