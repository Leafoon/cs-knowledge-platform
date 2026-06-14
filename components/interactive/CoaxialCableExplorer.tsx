"use client";
import { useState } from "react";

interface Layer {
  name: string;
  material: string;
  function: string;
  thickness: string;
  color: string;
  borderColor: string;
}

const layers: Layer[] = [
  { name: "外护套", material: "PVC/PE", function: "保护内部结构免受机械损伤和环境影响", thickness: "1-2mm", color: "bg-gray-300 dark:bg-gray-600", borderColor: "border-gray-400" },
  { name: "编织屏蔽层", material: "铜/铝编织", function: "电磁干扰(EMI)屏蔽，提供接地回路", thickness: "0.2-0.5mm", color: "bg-yellow-200 dark:bg-yellow-900/30", borderColor: "border-yellow-400" },
  { name: "绝缘介质", material: "PE/PTE/Foam", function: "隔离内外导体，决定电缆特性阻抗(如50Ω/75Ω)", thickness: "2-5mm", color: "bg-white dark:bg-gray-800", borderColor: "border-gray-300" },
  { name: "内导体", material: "铜/铜包钢", function: "信号传输的主要载体", thickness: "0.5-2mm", color: "bg-orange-300 dark:bg-orange-800", borderColor: "border-orange-500" },
];

const cableTypes = [
  { name: "RG-6", impedance: "75Ω", use: "有线电视/卫星信号", maxFreq: "3GHz" },
  { name: "RG-58", impedance: "50Ω", use: "以太网(10Base2)", maxFreq: "1GHz" },
  { name: "RG-213", impedance: "50Ω", use: "业余无线电/HF通信", maxFreq: "3GHz" },
  { name: "LMR-400", impedance: "50Ω", use: "基站天线馈线", maxFreq: "6GHz" },
];

export function CoaxialCableExplorer() {
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [selectedType, setSelectedType] = useState(0);
  const [showCutaway, setShowCutaway] = useState(true);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">同轴电缆物理结构</h3>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setShowCutaway(!showCutaway)}
          className={`px-3 py-1.5 rounded text-sm ${showCutaway ? "bg-blue-600 text-white" : "bg-gray-100 dark:bg-gray-800 text-text-secondary"}`}>
          {showCutaway ? "截面视图" : "展开视图"}
        </button>
      </div>
      {showCutaway ? (
        <div className="flex justify-center mb-4">
          <div className="relative w-52 h-52">
            {layers.map((layer, i) => {
              const size = 200 - i * 40;
              return (
                <button key={i} onClick={() => setSelectedLayer(selectedLayer === i ? null : i)}
                  className={`absolute rounded-full border-2 transition-all ${layer.color} ${layer.borderColor} ${selectedLayer === i ? "ring-2 ring-blue-500 scale-105" : ""}`}
                  style={{ width: size, height: size, top: (200 - size) / 2, left: (200 - size) / 2, zIndex: layers.length - i }}>
                  {i === layers.length - 1 && <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-text-primary">导体</span>}
                </button>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="space-y-2 mb-4">
          {layers.map((layer, i) => (
            <button key={i} onClick={() => setSelectedLayer(selectedLayer === i ? null : i)}
              className={`w-full p-3 rounded border-2 text-left transition-all ${layer.borderColor} ${layer.color} ${selectedLayer === i ? "ring-2 ring-blue-500" : ""}`}>
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-text-primary">{layer.name}</span>
                <span className="text-xs text-text-secondary">{layer.thickness}</span>
              </div>
            </button>
          ))}
        </div>
      )}
      {selectedLayer !== null && (
        <div className="p-3 rounded bg-gray-50 dark:bg-gray-800 border border-border-subtle mb-4">
          <div className="flex items-center gap-2 mb-2">
            <div className={`w-4 h-4 rounded ${layers[selectedLayer].color} ${layers[selectedLayer].borderColor} border-2`} />
            <span className="text-sm font-bold text-text-primary">{layers[selectedLayer].name}</span>
          </div>
          <div className="space-y-1 text-xs text-text-secondary">
            <p><span className="text-text-primary font-medium">材料:</span> {layers[selectedLayer].material}</p>
            <p><span className="text-text-primary font-medium">功能:</span> {layers[selectedLayer].function}</p>
            <p><span className="text-text-primary font-medium">厚度:</span> {layers[selectedLayer].thickness}</p>
          </div>
        </div>
      )}
      <div className="mb-4">
        <p className="text-xs font-medium text-text-primary mb-2">常见型号</p>
        <div className="grid grid-cols-2 gap-2">
          {cableTypes.map((ct, i) => (
            <button key={i} onClick={() => setSelectedType(i)}
              className={`p-2 rounded border text-left ${selectedType === i ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20" : "border-border-subtle bg-gray-50 dark:bg-gray-800"}`}>
              <div className="text-xs font-bold text-text-primary">{ct.name} ({ct.impedance})</div>
              <div className="text-[10px] text-text-secondary">{ct.use}</div>
              <div className="text-[10px] text-text-secondary">最高频率: {ct.maxFreq}</div>
            </button>
          ))}
        </div>
      </div>
      <p className="text-xs text-text-secondary">同轴电缆由内到外：内导体→绝缘介质→屏蔽层→外护套。特性阻抗由内外导体半径比和介质介电常数决定。</p>
    </div>
  );
}
export default CoaxialCableExplorer;
