'use client';
import { useState } from 'react';

interface Pass {
  name: string;
  category: string;
  latency: number;
  description: string;
  inputIR: string;
  outputIR: string;
  details: string[];
}

const passes: Pass[] = [
  { name: '算子融合', category: '优化', latency: 12, description: '将多个小算子融合为一个大算子，减少内存访问和kernel launch开销', inputIR: 'conv2d → relu → pool', outputIR: 'fused_conv_relu_pool', details: ['检测相邻的elementwise+reduce模式', '验证内存布局兼容性', '生成融合后的kernel代码'] },
  { name: '内存布局转换', category: '布局', latency: 8, description: '将数据从NCHW转换为NHWC以适配硬件', inputIR: 'tensor<NCHW, float>', outputIR: 'tensor<NHWC, float>', details: ['分析目标硬件的内存对齐要求', '插入转置或重排指令', '优化数据搬运路径'] },
  { name: '循环展开', category: '优化', latency: 5, description: '展开内层循环减少分支预测失败', inputIR: 'for(i=0;i<4;i++) { ... }', outputIR: 'unrolled_4_iterations', details: ['评估展开因子对寄存器压力的影响', '选择最优展开策略'] },
  { name: '常量折叠', category: '分析', latency: 2, description: '在编译期计算常量表达式', inputIR: 'x = 3 * 4 + 2', outputIR: 'x = 14', details: ['识别常量子表达式', '递归简化嵌套表达式'] },
  { name: 'Tiling分块', category: '优化', latency: 15, description: '将计算分块以提高缓存利用率', inputIR: 'global tile<64,64>', outputIR: 'local tile<16,16> x 4x4', details: ['分析数据局部性', '选择最优tile尺寸', '插入边界处理逻辑'] },
  { name: '流水线插入', category: '调度', latency: 18, description: '在计算和数据搬运之间插入流水线', inputIR: 'compute(); copy();', outputIR: 'pipeline(compute, copy, depth=2)', details: ['分析依赖关系', '划分pipeline阶段', '插入同步原语'] },
  { name: '向量化', category: '优化', latency: 7, description: '将标量操作转换为SIMD向量操作', inputIR: 'for(i) { a[i]=b[i]+c[i]; }', outputIR: 'vec4 a[i] = vec4(b[i]) + vec4(c[i]);', details: ['检测可向量化循环', '处理对齐和尾部'] },
  { name: '冗余消除', category: '分析', latency: 3, description: '消除重复计算和死代码', inputIR: 'x = f(a); y = f(a);', outputIR: 'x = f(a); y = x;', details: ['构建use-def链', '识别公共子表达式', '移除未使用的定义'] },
];

export function PassPipelineExplorer() {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [filter, setFilter] = useState<string>('全部');

  const categories = ['全部', ...new Set(passes.map(p => p.category))];
  const filtered = filter === '全部' ? passes : passes.filter(p => p.category === filter);

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold mb-4 text-cyan-400">编译Pass流水线探索器</h2>

      <div className="flex gap-2 mb-4 flex-wrap">
        {categories.map(cat => (
          <button key={cat} onClick={() => setFilter(cat)}
            className={`px-3 py-1 rounded-full text-sm transition-all ${
              filter === cat ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}>{cat}</button>
        ))}
      </div>

      <div className="space-y-2">
        {filtered.map((pass, i) => (
          <div key={i} className="border border-gray-700 rounded-lg overflow-hidden">
            <button onClick={() => setExpanded(expanded === i ? null : i)}
              className="w-full flex items-center justify-between p-3 hover:bg-gray-800 transition-colors text-left">
              <div className="flex items-center gap-3">
                <span className="text-lg font-mono font-bold text-yellow-400 w-6">{i + 1}</span>
                <div>
                  <div className="font-semibold">{pass.name}</div>
                  <div className="text-xs text-gray-400">{pass.category} · {pass.latency}μs</div>
                </div>
              </div>
              <svg className={`w-5 h-5 transition-transform ${expanded === i ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {expanded === i && (
              <div className="p-4 bg-gray-850 border-t border-gray-700 space-y-3" style={{ backgroundColor: '#1a2332' }}>
                <p className="text-sm text-gray-300">{pass.description}</p>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-gray-800 p-3 rounded">
                    <div className="text-gray-400 text-xs mb-1">输入IR</div>
                    <code className="text-green-400 font-mono text-xs">{pass.inputIR}</code>
                  </div>
                  <div className="bg-gray-800 p-3 rounded">
                    <div className="text-gray-400 text-xs mb-1">输出IR</div>
                    <code className="text-blue-400 font-mono text-xs">{pass.outputIR}</code>
                  </div>
                </div>
                <ul className="space-y-1">
                  {pass.details.map((d, j) => (
                    <li key={j} className="text-xs text-gray-400 flex items-start gap-2">
                      <span className="text-cyan-500 mt-0.5">▸</span>{d}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-4 text-xs text-gray-500">共 {passes.length} 个Pass · 累计延迟 {passes.reduce((s, p) => s + p.latency, 0)}μs</div>
    </div>
  );
}
