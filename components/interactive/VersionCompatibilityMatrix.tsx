"use client";

import { motion } from "framer-motion";
import { Badge, AlertCircle } from "lucide-react";

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
  const getStatusBadge = (status: string) => {
    switch (status) {
      case "current":
        return { label: "最新版本", color: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400" };
      case "stable":
        return { label: "稳定版本", color: "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400" };
      case "legacy":
        return { label: "历史版本", color: "bg-gray-100 dark:bg-gray-700/50 text-gray-700 dark:text-gray-400" };
      default:
        return { label: "", color: "" };
    }
  };

  return (
    <div className="my-8">
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-2xl font-semibold text-text-primary mb-2">
          版本兼容性矩阵
        </h3>
        <p className="text-sm text-text-secondary">
          不同 Transformers 版本对应的依赖要求
        </p>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full align-middle">
          <div className="overflow-hidden border border-gray-200 dark:border-gray-700 rounded-lg">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-text-primary uppercase tracking-wider">
                    Transformers
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-text-primary uppercase tracking-wider">
                    PyTorch
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-text-primary uppercase tracking-wider">
                    Python
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-text-primary uppercase tracking-wider">
                    CUDA
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-text-primary uppercase tracking-wider">
                    重要特性
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-text-primary uppercase tracking-wider">
                    状态
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {versions.map((version, index) => {
                  const badge = getStatusBadge(version.status);
                  return (
                    <motion.tr
                      key={index}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                    >
                      <td className="px-4 py-3 whitespace-nowrap">
                        <code className="text-sm font-mono text-text-primary bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                          {version.transformers}
                        </code>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <code className="text-sm font-mono text-text-secondary">
                          {version.pytorch}
                        </code>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <code className="text-sm font-mono text-text-secondary">
                          {version.python}
                        </code>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <code className="text-sm font-mono text-text-secondary">
                          {version.cuda}
                        </code>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex flex-wrap gap-1">
                          {version.features.map((feature, i) => (
                            <span
                              key={i}
                              className="inline-flex items-center px-2 py-0.5 text-xs font-medium bg-gray-100 dark:bg-gray-700 text-text-primary rounded"
                            >
                              {feature}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2 py-1 text-xs font-medium rounded ${badge.color}`}>
                          {badge.label}
                        </span>
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Compatibility Tips */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-200 dark:border-yellow-800/30 rounded-lg"
      >
        <div className="flex items-start gap-2 mb-3">
          <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-500 flex-shrink-0 mt-0.5" />
          <h4 className="text-sm font-semibold text-yellow-800 dark:text-yellow-400">
            兼容性提示
          </h4>
        </div>
        <ul className="text-sm text-yellow-700 dark:text-yellow-300/90 space-y-2 ml-7">
          <li>• CUDA 版本必须与 PyTorch 匹配，否则 GPU 不可用</li>
          <li>• Python 3.7 已不再支持，推荐使用 3.9+</li>
          <li>• M1/M2 Mac 使用 <code className="px-1.5 py-0.5 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-900 dark:text-yellow-200 rounded text-xs font-mono">torch</code> 而非 <code className="px-1.5 py-0.5 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-900 dark:text-yellow-200 rounded text-xs font-mono">torch-cpu</code></li>
          <li>• 使用最新稳定版本获得最佳性能和功能</li>
        </ul>
      </motion.div>
    </div>
  );
}
