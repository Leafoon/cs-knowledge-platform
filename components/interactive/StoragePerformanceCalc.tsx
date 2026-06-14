"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Calculator, HardDrive, Cpu } from "lucide-react";

type CalcType = "hdd" | "ssd" | "raid";

export function StoragePerformanceCalc() {
  const [type, setType] = useState<CalcType>("hdd");

  const [seekTime, setSeekTime] = useState(8);
  const [rotLatency, setRotLatency] = useState(4);
  const [transferRate, setTransferRate] = useState(150);

  const [readIOPS, setReadIOPS] = useState(100000);
  const [writeIOPS, setWriteIOPS] = useState(80000);
  const [readRatio, setReadRatio] = useState(70);

  const [raidLevel, setRaidLevel] = useState("5");
  const [diskCount, setDiskCount] = useState(4);
  const [singleIOPS, setSingleIOPS] = useState(200);

  const hddAccessTime = seekTime + rotLatency + (512 / (transferRate * 1024)) * 1000;
  const hddThroughput = transferRate;

  const ssdIOPS = (readIOPS * readRatio + writeIOPS * (100 - readRatio)) / 100;
  const ssdLatency = 1000000 / ssdIOPS;

  const raidIOPS = (() => {
    const r = readRatio / 100;
    switch (raidLevel) {
      case "0": return diskCount * singleIOPS;
      case "1": return diskCount * singleIOPS * r + (diskCount / 2) * singleIOPS * (1 - r);
      case "5": return diskCount * singleIOPS * r + (diskCount / 4) * singleIOPS * (1 - r);
      case "6": return diskCount * singleIOPS * r + (diskCount / 6) * singleIOPS * (1 - r);
      case "10": return diskCount * singleIOPS * r + (diskCount / 2) * singleIOPS * (1 - r);
      default: return 0;
    }
  })();

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Calculator className="w-5 h-5 text-teal-500" />
        存储性能计算器
      </h3>
      <div className="flex gap-2 mb-4">
        {(["hdd", "ssd", "raid"] as CalcType[]).map(t => (
          <button key={t} onClick={() => setType(t)}
            className={`px-3 py-1.5 rounded text-sm ${type === t ? "bg-teal-500 text-white" : "bg-bg-subtle"}`}>
            {t === "hdd" ? "HDD寻道" : t === "ssd" ? "SSD IOPS" : "RAID性能"}
          </button>
        ))}
      </div>
      {type === "hdd" && (
        <div>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <label className="text-sm text-text-secondary">平均寻道时间: {seekTime}ms</label>
              <input type="range" min={1} max={20} value={seekTime} onChange={e => setSeekTime(+e.target.value)} className="w-full" />
            </div>
            <div>
              <label className="text-sm text-text-secondary">平均旋转延迟: {rotLatency}ms</label>
              <input type="range" min={1} max={10} value={rotLatency} onChange={e => setRotLatency(+e.target.value)} className="w-full" />
            </div>
            <div>
              <label className="text-sm text-text-secondary">传输速率: {transferRate}MB/s</label>
              <input type="range" min={50} max={500} step={10} value={transferRate} onChange={e => setTransferRate(+e.target.value)} className="w-full" />
            </div>
          </div>
          <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }}
            className="bg-teal-500/10 border border-teal-500 rounded-lg p-4 text-center">
            <div className="text-sm text-text-secondary">平均访问时间</div>
            <div className="text-3xl font-bold text-teal-500 font-mono">{hddAccessTime.toFixed(2)} ms</div>
            <div className="text-xs text-text-secondary mt-1">吞吐量: {hddThroughput} MB/s</div>
          </motion.div>
        </div>
      )}
      {type === "ssd" && (
        <div>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <label className="text-sm text-text-secondary">读IOPS: {readIOPS}</label>
              <input type="range" min={10000} max={500000} step={10000} value={readIOPS} onChange={e => setReadIOPS(+e.target.value)} className="w-full" />
            </div>
            <div>
              <label className="text-sm text-text-secondary">写IOPS: {writeIOPS}</label>
              <input type="range" min={5000} max={300000} step={5000} value={writeIOPS} onChange={e => setWriteIOPS(+e.target.value)} className="w-full" />
            </div>
            <div>
              <label className="text-sm text-text-secondary">读比例: {readRatio}%</label>
              <input type="range" min={0} max={100} value={readRatio} onChange={e => setReadRatio(+e.target.value)} className="w-full" />
            </div>
          </div>
          <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }}
            className="bg-teal-500/10 border border-teal-500 rounded-lg p-4 text-center">
            <div className="text-sm text-text-secondary">混合IOPS</div>
            <div className="text-3xl font-bold text-teal-500 font-mono">{ssdIOPS.toFixed(0)}</div>
            <div className="text-xs text-text-secondary mt-1">平均延迟: {(ssdLatency / 1000).toFixed(2)} μs</div>
          </motion.div>
        </div>
      )}
      {type === "raid" && (
        <div>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <label className="text-sm text-text-secondary">RAID级别</label>
              <select value={raidLevel} onChange={e => setRaidLevel(e.target.value)}
                className="w-full mt-1 p-1 bg-bg-subtle rounded text-sm">
                {["0", "1", "5", "6", "10"].map(l => <option key={l} value={l}>RAID {l}</option>)}
              </select>
            </div>
            <div>
              <label className="text-sm text-text-secondary">磁盘数: {diskCount}</label>
              <input type="range" min={2} max={12} value={diskCount} onChange={e => setDiskCount(+e.target.value)} className="w-full" />
            </div>
            <div>
              <label className="text-sm text-text-secondary">单盘IOPS: {singleIOPS}</label>
              <input type="range" min={50} max={500} step={10} value={singleIOPS} onChange={e => setSingleIOPS(+e.target.value)} className="w-full" />
            </div>
          </div>
          <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }}
            className="bg-teal-500/10 border border-teal-500 rounded-lg p-4 text-center">
            <div className="text-sm text-text-secondary">RAID {raidLevel} IOPS</div>
            <div className="text-3xl font-bold text-teal-500 font-mono">{raidIOPS.toFixed(0)}</div>
            <div className="text-xs text-text-secondary mt-1">读/写比: {readRatio}/{100 - readRatio}</div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
