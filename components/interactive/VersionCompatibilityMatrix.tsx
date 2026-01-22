"use client";

import { motion } from "framer-motion";
import { useState } from "react";

interface Version {
  transformers: string;
  pytorch: string;
  python: string;
  cuda: string;
  features: string[];
  status: "current" | "stable" | "legacy";
}

const versions: Version[] = [
  {
    transformers: "v4.40+",
    pytorch: "2.0+",
    python: "3.9+",
    cuda: "11.8+",
    features: ["Gemma 2", "Qwen 2.5", "Llama 3.1"],
    status: "current"
  },
  {
    transformers: "v4.35-4.39",
    pytorch: "2.0+",
    python: "3.8+",
    cuda: "11.8+",
    features: ["Mixtral", "Phi-3", "Gemma"],
    status: "stable"
  },
  {
    transformers: "v4.30-4.34",
    pytorch: "1.13+",
    python: "3.8+",
    cuda: "11.7+",
    features: ["LLaMA 2", "Mistral", "Falcon"],
    status: "stable"
  },
  {
    transformers: "v4.25-4.29",
    pytorch: "1.11+",
    python: "3.7+",
    cuda: "11.6+",
    features: ["BLOOM", "OPT", "GPT-NeoX"],
    status: "legacy"
  },
  {
    transformers: "< v4.25",
    pytorch: "1.9+",
    python: "3.7+",
    cuda: "11.3+",
    features: ["BERT", "GPT-2", "T5"],
    status: "legacy"
  }
];

export default function VersionCompatibilityMatrix() {
  const [selectedVersion, setSelectedVersion] = useState<number>(0);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "current":
        return "from-green-500 to-emerald-600";
      case "stable":
        return "from-blue-500 to-cyan-600";
      case "legacy":
        return "from-gray-500 to-slate-600";
      default:
        return "from-gray-500 to-slate-600";
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "current":
        return "最新版本";
      case "stable":
        return "稳定版本";
      case "legacy":
        return "历史版本";
      default:
        return "";
    }
  };

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl border border-slate-700 shadow-2xl overflow-hidden">
      <h3 className="text-2xl font-bold mb-6 text-center bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
        📦 版本兼容性矩阵
      </h3>

      {/* Desktop View - Table */}
      <div className="hidden md:block overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-600">
              <th className="text-left p-3 text-blue-400 font-semibold">Transformers</th>
              <th className="text-left p-3 text-orange-400 font-semibold">PyTorch</th>
              <th className="text-left p-3 text-green-400 font-semibold">Python</th>
              <th className="text-left p-3 text-purple-400 font-semibold">CUDA</th>
              <th className="text-left p-3 text-yellow-400 font-semibold">重要特性</th>
              <th className="text-left p-3 text-pink-400 font-semibold">状态</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((version, index) => (
              <motion.tr
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ backgroundColor: "rgba(71, 85, 105, 0.3)" }}
                className="border-b border-slate-700 cursor-pointer"
                onClick={() => setSelectedVersion(index)}
              >
                <td className="p-3 font-mono text-blue-300">{version.transformers}</td>
                <td className="p-3 font-mono text-orange-300">{version.pytorch}</td>
                <td className="p-3 font-mono text-green-300">{version.python}</td>
                <td className="p-3 font-mono text-purple-300">{version.cuda}</td>
                <td className="p-3">
                  <div className="flex flex-wrap gap-1">
                    {version.features.map((feature, i) => (
                      <span
                        key={i}
                        className="px-2 py-0.5 bg-slate-700 text-yellow-300 text-xs rounded"
                      >
                        {feature}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="p-3">
                  <span className={`px-2 py-1 text-xs font-bold rounded bg-gradient-to-r ${getStatusColor(version.status)} text-white`}>
                    {getStatusBadge(version.status)}
                  </span>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile View - Cards */}
      <div className="md:hidden space-y-4">
        {versions.map((version, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileTap={{ scale: 0.98 }}
            className="p-4 bg-slate-800 rounded-lg border border-slate-700"
            onClick={() => setSelectedVersion(index)}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="font-bold text-blue-300">{version.transformers}</span>
              <span className={`px-2 py-1 text-xs font-bold rounded bg-gradient-to-r ${getStatusColor(version.status)} text-white`}>
                {getStatusBadge(version.status)}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm mb-3">
              <div>
                <span className="text-gray-400">PyTorch:</span>
                <span className="ml-2 text-orange-300 font-mono">{version.pytorch}</span>
              </div>
              <div>
                <span className="text-gray-400">Python:</span>
                <span className="ml-2 text-green-300 font-mono">{version.python}</span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-400">CUDA:</span>
                <span className="ml-2 text-purple-300 font-mono">{version.cuda}</span>
              </div>
            </div>
            <div>
              <span className="text-gray-400 text-xs">重要特性：</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {version.features.map((feature, i) => (
                  <span
                    key={i}
                    className="px-2 py-0.5 bg-slate-700 text-yellow-300 text-xs rounded"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Compatibility Tips */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="mt-6 p-4 bg-yellow-900/20 border border-yellow-700/50 rounded-lg"
      >
        <h4 className="text-sm font-bold text-yellow-400 mb-2">⚠️ 兼容性提示</h4>
        <ul className="text-xs text-gray-300 space-y-1">
          <li>• CUDA 版本必须与 PyTorch 匹配，否则 GPU 不可用</li>
          <li>• Python 3.7 已不再支持，推荐使用 3.9+</li>
          <li>• M1/M2 Mac 使用 <code className="px-1 bg-slate-700 rounded">torch</code> 而非 <code className="px-1 bg-slate-700 rounded">torch-cpu</code></li>
          <li>• 使用最新稳定版本获得最佳性能和功能</li>
        </ul>
      </motion.div>
    </div>
  );
}
