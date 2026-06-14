"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Lock, Key, AlertTriangle, CheckCircle, X } from "lucide-react";

interface SecurityMechanism {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  examples: string[];
  threats: string[];
}

const mechanisms: SecurityMechanism[] = [
  {
    id: "authentication",
    name: "身份认证",
    description: "验证用户身份",
    icon: <Key className="w-6 h-6" />,
    examples: ["密码验证", "指纹识别", "双因素认证", "Kerberos"],
    threats: ["暴力破解", "字典攻击", "中间人攻击"]
  },
  {
    id: "authorization",
    name: "访问控制",
    description: "确定用户权限",
    icon: <Lock className="w-6 h-6" />,
    examples: ["ACL (访问控制列表)", "RBAC (基于角色)", "Capabilities", "SELinux"],
    threats: ["权限提升", "越权访问", "TOCTOU 漏洞"]
  },
  {
    id: "isolation",
    name: "隔离机制",
    description: "防止进程间干扰",
    icon: <Shield className="w-6 h-6" />,
    examples: ["虚拟内存隔离", "容器隔离", "虚拟机隔离", "沙箱"],
    threats: ["缓冲区溢出", "侧信道攻击", "Spectre/Meltdown"]
  },
  {
    id: "encryption",
    name: "加密保护",
    description: "保护数据机密性",
    icon: <Key className="w-6 h-6" />,
    examples: ["磁盘加密", "网络加密 (TLS)", "文件加密", "内存加密"],
    threats: ["中间人攻击", "密钥泄露", "降级攻击"]
  }
];

export default function SecurityMechanismsDemo() {
  const [selectedMech, setSelectedMech] = useState<string>("authentication");
  const [attackSimulation, setAttackSimulation] = useState(false);
  const [attackBlocked, setAttackBlocked] = useState(false);

  const selected = mechanisms.find(m => m.id === selectedMech)!;

  const simulateAttack = () => {
    setAttackSimulation(true);
    setTimeout(() => {
      setAttackBlocked(true);
      setTimeout(() => {
        setAttackSimulation(false);
        setAttackBlocked(false);
      }, 2000);
    }, 1500);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-950 dark:to-slate-900 rounded-xl shadow-2xl">
      <div className="flex items-center gap-3 mb-6">
        <Shield className="w-8 h-8 text-red-600 dark:text-red-400" />
        <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
          操作系统安全机制
        </h3>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 机制列表 */}
        <div className="lg:col-span-1 space-y-2">
          {mechanisms.map((mech) => (
            <motion.button
              key={mech.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setSelectedMech(mech.id)}
              className={`
                w-full text-left p-4 rounded-lg transition-all flex items-center gap-3
                ${selectedMech === mech.id
                  ? "bg-red-600 text-white shadow-lg"
                  : "bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-red-50 dark:hover:bg-slate-700"
                }
              `}
            >
              {mech.icon}
              <div>
                <div className="font-semibold">{mech.name}</div>
                <div className={`text-xs mt-1 ${selectedMech === mech.id ? "text-red-100" : "text-slate-500"}`}>
                  {mech.description}
                </div>
              </div>
            </motion.button>
          ))}
        </div>

        {/* 详细信息 */}
        <div className="lg:col-span-2 space-y-4">
          {/* 实现示例 */}
          <div className="p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              典型实现
            </h4>
            <div className="grid grid-cols-2 gap-2">
              {selected.examples.map((example, i) => (
                <div
                  key={i}
                  className="p-2 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-800 text-sm text-green-700 dark:text-green-300"
                >
                  {example}
                </div>
              ))}
            </div>
          </div>

          {/* 威胁模型 */}
          <div className="p-4 bg-white dark:bg-slate-800 rounded-lg shadow">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              潜在威胁
            </h4>
            <ul className="space-y-2">
              {selected.threats.map((threat, i) => (
                <li
                  key={i}
                  className="p-2 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-800 text-sm text-red-700 dark:text-red-300 flex items-center gap-2"
                >
                  <X className="w-4 h-4" />
                  {threat}
                </li>
              ))}
            </ul>
          </div>

          {/* 攻击模拟 */}
          <div className="p-4 bg-slate-100 dark:bg-slate-900 rounded-lg">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">
              安全演示
            </h4>
            <div className="flex items-center gap-4 mb-4">
              <button
                onClick={simulateAttack}
                disabled={attackSimulation}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-400 text-white rounded-lg font-semibold"
              >
                模拟攻击
              </button>
              <AnimatePresence>
                {attackSimulation && (
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center gap-2"
                  >
                    {attackBlocked ? (
                      <>
                        <Shield className="w-5 h-5 text-green-600" />
                        <span className="text-green-600 font-semibold">攻击已阻止！</span>
                      </>
                    ) : (
                      <>
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        >
                          <AlertTriangle className="w-5 h-5 text-red-600" />
                        </motion.div>
                        <span className="text-red-600 font-semibold">检测到攻击...</span>
                      </>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* 防御流程可视化 */}
            <div className="space-y-2">
              {["检测异常", "验证权限", "触发防护", "记录日志"].map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0.3 }}
                  animate={{
                    opacity: attackSimulation && !attackBlocked ? (i <= 1 ? 1 : 0.3) : attackBlocked ? 1 : 0.3,
                    backgroundColor: attackBlocked ? "#dcfce7" : attackSimulation && i <= 1 ? "#fef3c7" : "#f1f5f9"
                  }}
                  className="p-2 rounded text-sm text-slate-700 dark:text-slate-300"
                >
                  {i + 1}. {step}
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 总结 */}
      <div className="mt-6 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
        <p className="text-sm text-red-900 dark:text-red-100">
          <strong>深度防御 (Defense in Depth)：</strong> 现代操作系统采用多层安全机制，
          即使某一层被突破，其他层仍能提供保护。没有单一的"银弹"，安全需要多种机制协同工作。
        </p>
      </div>
    </div>
  );
}
