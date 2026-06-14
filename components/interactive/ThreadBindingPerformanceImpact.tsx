'use client';
import { useState } from 'react';

const strategies = [
  { name: '无绑定 (默认)', occupancy: '100%', cacheHit: '32%', bandwidth: '1.2 TB/s', perf: 1.0, color: 'bg-gray-500' },
  { name: 'Block级绑定', occupancy: '85%', cacheHit: '68%', bandwidth: '2.1 TB/s', perf: 1.75, color: 'bg-blue-500' },
  { name: 'Warp级绑定', occupancy: '92%', cacheHit: '78%', bandwidth: '2.5 TB/s', perf: 2.1, color: 'bg-green-500' },
  { name: '线程级绑定', occupancy: '78%', cacheHit: '89%', bandwidth: '2.8 TB/s', perf: 2.3, color: 'bg-purple-500' },
  { name: 'NUMA感知绑定', occupancy: '88%', cacheHit: '91%', bandwidth: '3.1 TB/s', perf: 2.6, color: 'bg-cyan-500' },
];

export function ThreadBindingPerformanceImpact() {
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [sortKey, setSortKey] = useState<'name' | 'occupancy' | 'cacheHit' | 'bandwidth' | 'perf'>('perf');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  const sorted = [...strategies].sort((a, b) => {
    const mul = sortDir === 'desc' ? -1 : 1;
    if (sortKey === 'perf') return (a.perf - b.perf) * mul;
    if (sortKey === 'name') return a.name.localeCompare(b.name) * mul;
    return (parseFloat(a[sortKey] as string) - parseFloat(b[sortKey] as string)) * mul;
  });

  const toggleSort = (key: typeof sortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  const SortIcon = ({ col }: { col: string }) => (
    <span className="ml-1 text-gray-500">
      {sortKey === col ? (sortDir === 'desc' ? '▼' : '▲') : '⇅'}
    </span>
  );

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-xl font-bold text-cyan-400 mb-4">线程绑定策略性能对比</h2>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left py-3 px-2 cursor-pointer text-gray-400" onClick={() => toggleSort('name')}>
                绑定策略<SortIcon col="name" />
              </th>
              <th className="text-center py-3 px-2 cursor-pointer text-gray-400" onClick={() => toggleSort('occupancy')}>
                占用率<SortIcon col="occupancy" />
              </th>
              <th className="text-center py-3 px-2 cursor-pointer text-gray-400" onClick={() => toggleSort('cacheHit')}>
                缓存命中<SortIcon col="cacheHit" />
              </th>
              <th className="text-center py-3 px-2 cursor-pointer text-gray-400" onClick={() => toggleSort('bandwidth')}>
                带宽<SortIcon col="bandwidth" />
              </th>
              <th className="text-left py-3 px-2 cursor-pointer text-gray-400" onClick={() => toggleSort('perf')}>
                相对性能<SortIcon col="perf" />
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s, i) => (
              <tr key={i}
                className={`border-b border-gray-800 cursor-pointer transition-colors ${
                  selectedRow === strategies.indexOf(s) ? 'bg-gray-800' : 'hover:bg-gray-800/50'
                }`}
                onClick={() => setSelectedRow(selectedRow === strategies.indexOf(s) ? null : strategies.indexOf(s))}>
                <td className="py-3 px-2">
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${s.color}`} />
                    <span>{s.name}</span>
                  </div>
                </td>
                <td className="text-center py-3 px-2 font-mono text-gray-300">{s.occupancy}</td>
                <td className="text-center py-3 px-2 font-mono text-gray-300">{s.cacheHit}</td>
                <td className="text-center py-3 px-2 font-mono text-gray-300">{s.bandwidth}</td>
                <td className="py-3 px-2">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-800 rounded-full h-4 overflow-hidden">
                      <div className={`h-full rounded-full ${s.color} transition-all`}
                        style={{ width: `${(s.perf / 3) * 100}%` }} />
                    </div>
                    <span className="font-mono text-gray-300 w-10 text-right">{s.perf}x</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedRow !== null && (
        <div className="mt-4 p-4 bg-gray-800 rounded-lg text-sm">
          <div className="font-semibold text-cyan-300">{strategies[selectedRow].name}</div>
          <div className="text-gray-300 mt-1">
            {selectedRow === 0 && '不做任何绑定，由操作系统调度线程到硬件资源。简单但无法利用数据局部性。'}
            {selectedRow === 1 && '将Block绑定到特定SM，利用SM内共享内存和L1缓存，提高数据复用率。'}
            {selectedRow === 2 && '进一步将Warp绑定到特定执行单元，减少线程迁移开销。'}
            {selectedRow === 3 && '精确控制每个线程到具体的硬件资源，最大化缓存命中率。'}
            {selectedRow === 4 && '考虑NUMA架构，将线程绑定到距离数据最近的计算单元，减少跨节点访问。'}
          </div>
        </div>
      )}
    </div>
  );
}
