"use client";
import { useState } from "react";

const modulations = [
  {
    name: "ASK",
    full: "Amplitude Shift Keying",
    zh: "幅移键控",
    desc: "通过改变载波幅度表示0和1",
    params: { amplitude: true, frequency: false, phase: false },
    bitsPerSymbol: 1,
    snr: "低",
    use: "光纤通信、RFID",
  },
  {
    name: "FSK",
    full: "Frequency Shift Keying",
    zh: "频移键控",
    desc: "通过改变载波频率表示0和1",
    params: { amplitude: false, frequency: true, phase: false },
    bitsPerSymbol: 1,
    snr: "中",
    use: "调制解调器、蓝牙",
  },
  {
    name: "PSK",
    full: "Phase Shift Keying",
    zh: "相移键控",
    desc: "通过改变载波相位表示数据",
    params: { amplitude: false, frequency: false, phase: true },
    bitsPerSymbol: 1,
    snr: "中高",
    use: "Wi-Fi、卫星通信",
  },
  {
    name: "QAM",
    full: "Quadrature Amplitude Modulation",
    zh: "正交幅度调制",
    desc: "同时改变幅度和相位，提高频谱效率",
    params: { amplitude: true, frequency: false, phase: true },
    bitsPerSymbol: 4,
    snr: "高",
    use: "4G/5G、有线电视、Wi-Fi",
  },
];

export function ModulationSimulator() {
  const [selected, setSelected] = useState(3);
  const [qamOrder, setQamOrder] = useState(16);
  const [signal, setSignal] = useState<number[]>([]);

  const mod = modulations[selected];
  const qamBits = Math.log2(qamOrder);

  const generateSignal = () => {
    const points = 100;
    const data = Array.from({ length: points }, (_, i) => {
      const t = i / points;
      const dataBit = Math.random() > 0.5 ? 1 : 0;
      if (selected === 0) {
        return (dataBit ? 1 : 0.3) * Math.sin(2 * Math.PI * 5 * t);
      } else if (selected === 1) {
        return Math.sin(2 * Math.PI * (dataBit ? 7 : 4) * t);
      } else if (selected === 2) {
        return Math.sin(2 * Math.PI * 5 * t + (dataBit ? Math.PI : 0));
      } else {
        const amp = 0.5 + (dataBit ? 0.5 : 0);
        const phase = dataBit ? Math.PI / 4 : -Math.PI / 4;
        return amp * Math.sin(2 * Math.PI * 5 * t + phase);
      }
    });
    setSignal(data);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Modulation Simulator <span className="text-text-secondary text-sm">— 数字调制技术</span>
      </h3>
      <div className="flex gap-2 mb-4 flex-wrap">
        {modulations.map((m, i) => (
          <button
            key={i}
            onClick={() => { setSelected(i); setSignal([]); }}
            className={`px-3 py-1 rounded text-sm font-mono ${selected === i ? "bg-blue-600 text-white" : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200"}`}
          >
            {m.name}
          </button>
        ))}
      </div>
      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded mb-4">
        <div className="font-semibold text-text-primary mb-1">{mod.name} — {mod.full}</div>
        <div className="text-sm text-text-secondary mb-2">{mod.zh}: {mod.desc}</div>
        <div className="grid grid-cols-3 gap-2 text-xs mb-3">
          <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
            <div className="text-text-secondary">每符号比特</div>
            <div className="font-bold text-text-primary">{selected === 3 ? qamBits : mod.bitsPerSymbol}</div>
          </div>
          <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
            <div className="text-text-secondary">抗噪性</div>
            <div className="font-bold text-text-primary">{mod.snr}</div>
          </div>
          <div className="bg-white dark:bg-gray-900 p-2 rounded text-center">
            <div className="text-text-secondary">应用</div>
            <div className="font-bold text-text-primary">{mod.use}</div>
          </div>
        </div>
        {selected === 3 && (
          <div className="mb-3">
            <div className="text-xs text-text-secondary mb-1">QAM阶数: {qamOrder}</div>
            <div className="flex gap-2">
              {[4, 16, 64, 256].map((q) => (
                <button
                  key={q}
                  onClick={() => { setQamOrder(q); setSignal([]); }}
                  className={`px-2 py-1 rounded text-xs ${qamOrder === q ? "bg-purple-600 text-white" : "bg-gray-200 dark:bg-gray-700"}`}
                >
                  {q}-QAM
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="flex gap-2">
          <div className="text-xs text-text-secondary">
            改变参数: {Object.entries(mod.params).filter(([, v]) => v).map(([k]) => k).join(", ")}
          </div>
        </div>
      </div>
      <button onClick={generateSignal} className="px-4 py-2 rounded bg-green-600 text-white text-sm mb-3">
        生成信号
      </button>
      {signal.length > 0 && (
        <div className="bg-gray-900 p-4 rounded">
          <svg viewBox="0 0 400 100" className="w-full h-24">
            <polyline
              fill="none"
              stroke="#60a5fa"
              strokeWidth="1.5"
              points={signal.map((v, i) => `${(i / signal.length) * 400},${50 - v * 40}`).join(" ")}
            />
            <line x1="0" y1="50" x2="400" y2="50" stroke="#4b5563" strokeWidth="0.5" />
          </svg>
          <div className="text-xs text-gray-400 text-center mt-1">时域波形</div>
        </div>
      )}
    </div>
  );
}

export default ModulationSimulator;
