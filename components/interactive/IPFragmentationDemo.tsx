"use client";
import { useState } from "react";

interface Fragment {
  id: number;
  offset: number;
  length: number;
  mf: boolean;
  data: string;
}

export function IPFragmentationDemo() {
  const [totalSize, setTotalSize] = useState(4000);
  const [mtu, setMtu] = useState(1500);
  const [df, setDf] = useState(false);

  const ipHeaderSize = 20;
  const maxPayload = mtu - ipHeaderSize;
  const alignedPayload = Math.floor(maxPayload / 8) * 8;

  const fragments: Fragment[] = [];
  if (df && totalSize > mtu) {
    // DF set, can't fragment
  } else {
    let offset = 0;
    let id = 1;
    while (offset < totalSize) {
      const len = Math.min(alignedPayload, totalSize - offset);
      const mf = offset + len < totalSize;
      fragments.push({
        id,
        offset: offset / 8,
        length: len,
        mf,
        data: `数据[${offset}-${offset + len - 1}]`,
      });
      offset += len;
      id++;
    }
  }

  const canFragment = !df || totalSize <= mtu;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">IP分片 (MF/DF/Offset)</h3>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-sm text-text-secondary">数据报大小: {totalSize} B</label>
          <input type="range" min={500} max={8000} step={100} value={totalSize} onChange={(e) => setTotalSize(Number(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-text-secondary">MTU: {mtu} B</label>
          <input type="range" min={576} max={4000} step={1} value={mtu} onChange={(e) => setMtu(Number(e.target.value))} className="w-full" />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-text-secondary">DF标志:</label>
          <button onClick={() => setDf(!df)}
            className={`px-3 py-1 rounded text-sm ${df ? "bg-red-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
            {df ? "DF=1 (禁止分片)" : "DF=0 (允许分片)"}
          </button>
        </div>
      </div>
      {!canFragment ? (
        <div className="bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 p-4 rounded-lg mb-4">
          ⚠ DF置位且数据报({totalSize}B) {">"} MTU({mtu}B),路由器将丢弃并返回ICMP "需要分片但DF置位"
        </div>
      ) : (
        <div className="space-y-2 mb-4">
          {fragments.map((f) => (
            <div key={f.id} className="bg-bg-muted rounded-lg p-3">
              <div className="flex items-center gap-4 text-sm">
                <span className="font-bold text-text-primary">分片 #{f.id}</span>
                <span className="text-text-secondary">偏移: {f.offset} ({f.offset * 8}B)</span>
                <span className="text-text-secondary">长度: {f.length}B</span>
                <span className={`px-2 py-0.5 rounded text-xs ${f.mf ? "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700" : "bg-green-100 dark:bg-green-900/30 text-green-700"}`}>
                  MF={f.mf ? 1 : 0}
                </span>
                <span className="text-text-secondary">DF={df ? 1 : 0}</span>
              </div>
              <div className="mt-1 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div className="h-2 bg-blue-500 rounded-full" style={{ width: `${(f.length / mtu) * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="flex gap-4 text-sm text-text-secondary">
        <span>原始大小: <strong>{totalSize}B</strong></span>
        <span>分片数: <strong>{canFragment ? fragments.length : "无法分片"}</strong></span>
        <span>每片最大载荷: <strong>{alignedPayload}B</strong></span>
      </div>
      <div className="text-xs text-text-secondary mt-3">
        IP分片: 当数据报大于链路MTU时,路由器将其分为多个片段。MF=1表示还有后续分片,Offset以8字节为单位。目的主机根据ID+Offset重组。
      </div>
    </div>
  );
}

export default IPFragmentationDemo;
