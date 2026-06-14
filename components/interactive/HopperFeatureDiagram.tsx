'use client';
import { useState } from 'react';

const features = [
  {
    name: 'TMA (张量内存加速器)',
    icon: '📦',
    color: 'border-blue-500 bg-blue-900/20',
    textColor: 'text-blue-300',
    description: '硬件级异步数据搬运引擎',
    details: ['异步拷贝全局内存到共享内存', '支持多维张量布局', '无需占用计算单元', '一次拷贝整个Tile'],
    metric: '带宽利用率提升40%',
  },
  {
    name: 'Warpgroup (线程组)',
    icon: '👥',
    color: 'border-purple-500 bg-purple-900/20',
    textColor: 'text-purple-300',
    description: '4个Warp组成Warpgroup协同工作',
    details: ['128线程协作执行', 'Warpgroup级同步原语', '支持异步执行和重叠', '优化Tensor Core利用率'],
    metric: '计算效率提升30%',
  },
  {
    name: 'FP8 数据格式',
    icon: '🔢',
    color: 'border-green-500 bg-green-900/20',
    textColor: 'text-green-300',
    description: '8位浮点推理和训练',
    details: ['E4M3/E5M2两种格式', '推理算力翻倍', '支持在线缩放', '训练收敛性保持'],
    metric: '吞吐量翻倍',
  },
  {
    name: '异步拷贝',
    icon: '⚡',
    color: 'border-orange-500 bg-orange-900/20',
    textColor: 'text-orange-300',
    description: '计算与数据搬运重叠执行',
    details: ['cp.async指令', 'Pipeline执行模式', '减少内存延迟影响', '配合TMA使用'],
    metric: '延迟隐藏90%+',
  },
];

export function HopperFeatureDiagram() {
  const [selectedFeature, setSelectedFeature] = useState<number | null>(null);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-6">Hopper GPU特性架构图</h2>

      {/* GPU Core visualization */}
      <div className="border border-gray-700 rounded-xl p-4 mb-6">
        <div className="text-sm text-gray-400 mb-3">H100 SM 架构示意</div>
        <div className="grid grid-cols-4 gap-2">
          {features.map((f, i) => (
            <div key={i}
              onClick={() => setSelectedFeature(selectedFeature === i ? null : i)}
              className={`border-2 rounded-lg p-3 cursor-pointer transition-all text-center ${
                selectedFeature === i ? f.color + ' ring-2 ring-white/20' : 'border-gray-700 hover:border-gray-500'
              }`}>
              <div className="text-2xl mb-1">{f.icon}</div>
              <div className={`text-xs font-medium ${f.textColor}`}>{f.name.split('(')[0]}</div>
              <div className="text-[10px] text-gray-500 mt-1">{f.metric}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature details */}
      {selectedFeature !== null && (
        <div className={`border-2 rounded-xl p-5 mb-4 ${features[selectedFeature].color}`}>
          <div className="flex items-center gap-3 mb-3">
            <span className="text-3xl">{features[selectedFeature].icon}</span>
            <div>
              <div className={`font-bold text-lg ${features[selectedFeature].textColor}`}>
                {features[selectedFeature].name}
              </div>
              <div className="text-sm text-gray-400">{features[selectedFeature].description}</div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {features[selectedFeature].details.map((d, j) => (
              <div key={j} className="bg-gray-800/50 rounded-lg px-3 py-2 text-xs text-gray-300 flex items-center gap-2">
                <span className="text-cyan-500">▸</span>{d}
              </div>
            ))}
          </div>
          <div className="mt-3 p-2 bg-gray-800/50 rounded-lg text-sm text-center">
            <span className="text-gray-400">性能提升: </span>
            <span className="text-cyan-300 font-bold">{features[selectedFeature].metric}</span>
          </div>
        </div>
      )}

      {/* Hopper vs Previous */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-gray-400 font-medium mb-2">Hopper (H100)</div>
          <ul className="space-y-1 text-xs text-gray-300">
            <li className="text-green-400">✓ TMA硬件搬运</li>
            <li className="text-green-400">✓ Warpgroup协同</li>
            <li className="text-green-400">✓ FP8支持</li>
            <li className="text-green-400">✓ 异步执行增强</li>
          </ul>
        </div>
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="text-gray-400 font-medium mb-2">Ampere (A100)</div>
          <ul className="space-y-1 text-xs text-gray-300">
            <li className="text-red-400">✗ 无TMA</li>
            <li className="text-yellow-400">△ Warp级协作</li>
            <li className="text-red-400">✗ 无FP8</li>
            <li className="text-yellow-400">△ 基础异步</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
