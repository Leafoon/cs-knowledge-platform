"use client";
import { useState } from "react";

export function SocketBufferVisualizer() {
  const [sendBufSize, setSendBufSize] = useState(65536);
  const [recvBufSize, setRecvBufSize] = useState(65536);
  const [sendUsed, setSendUsed] = useState(24000);
  const [recvUsed, setRecvUsed] = useState(48000);
  const [showFlow, setShowFlow] = useState(false);

  const sendPercent = (sendUsed / sendBufSize) * 100;
  const recvPercent = (recvUsed / recvBufSize) * 100;

  const fillSend = () => setSendUsed((u) => Math.min(u + 8000, sendBufSize));
  const drainSend = () => setSendUsed((u) => Math.max(u - 8000, 0));
  const fillRecv = () => setRecvUsed((u) => Math.min(u + 8000, recvBufSize));
  const drainRecv = () => setRecvUsed((u) => Math.max(u - 8000, 0));

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">套接字缓冲区可视化</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <label className="text-xs text-text-secondary">
          发送缓冲区: <span className="text-text-primary font-mono">{(sendBufSize / 1024).toFixed(0)} KB</span>
          <input type="range" min={4096} max={131072} step={4096} value={sendBufSize} onChange={(e) => setSendBufSize(+e.target.value)} className="w-full mt-1" />
        </label>
        <label className="text-xs text-text-secondary">
          接收缓冲区: <span className="text-text-primary font-mono">{(recvBufSize / 1024).toFixed(0)} KB</span>
          <input type="range" min={4096} max={131072} step={4096} value={recvBufSize} onChange={(e) => setRecvBufSize(+e.target.value)} className="w-full mt-1" />
        </label>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-sky-600 dark:text-sky-400">发送缓冲区 (Write Buffer)</span>
            <span className="text-xs font-mono text-text-secondary">{(sendUsed / 1024).toFixed(0)}/{(sendBufSize / 1024).toFixed(0)} KB</span>
          </div>
          <div className="relative h-6 bg-bg-tertiary rounded-full overflow-hidden border border-border-subtle">
            <div className={`h-full rounded-full transition-all ${sendPercent > 90 ? "bg-red-500" : sendPercent > 70 ? "bg-amber-500" : "bg-sky-500"}`}
              style={{ width: `${sendPercent}%` }} />
          </div>
          <div className="flex gap-1 mt-1">
            <button onClick={fillSend} className="flex-1 px-2 py-0.5 rounded bg-sky-500/15 text-sky-600 dark:text-sky-400 text-[10px]">写入 +8KB</button>
            <button onClick={drainSend} className="flex-1 px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 text-[10px]">发送 -8KB</button>
          </div>
          {sendPercent > 90 && <div className="text-[10px] text-red-500 mt-0.5">⚠ 缓冲区接近满，send()可能阻塞</div>}
        </div>
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-emerald-600 dark:text-emerald-400">接收缓冲区 (Read Buffer)</span>
            <span className="text-xs font-mono text-text-secondary">{(recvUsed / 1024).toFixed(0)}/{(recvBufSize / 1024).toFixed(0)} KB</span>
          </div>
          <div className="relative h-6 bg-bg-tertiary rounded-full overflow-hidden border border-border-subtle">
            <div className={`h-full rounded-full transition-all ${recvPercent > 90 ? "bg-red-500" : recvPercent > 70 ? "bg-amber-500" : "bg-emerald-500"}`}
              style={{ width: `${recvPercent}%` }} />
          </div>
          <div className="flex gap-1 mt-1">
            <button onClick={fillRecv} className="flex-1 px-2 py-0.5 rounded bg-sky-500/15 text-sky-600 dark:text-sky-400 text-[10px]">接收 +8KB</button>
            <button onClick={drainRecv} className="flex-1 px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 text-[10px]">读取 -8KB</button>
          </div>
          {recvPercent > 90 && <div className="text-[10px] text-red-500 mt-0.5">⚠ 缓冲区接近满，窗口通告缩小</div>}
        </div>
      </div>
      <button onClick={() => setShowFlow(!showFlow)} className="w-full px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary hover:text-text-primary transition-colors mb-3">
        {showFlow ? "隐藏" : "显示"}数据流路径
      </button>
      {showFlow && (
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-4 text-xs space-y-2">
          <div className="flex items-center gap-2"><span className="text-text-primary font-medium">应用层</span><span className="text-text-tertiary">→</span><span className="text-sky-500">write() 写入发送缓冲区</span></div>
          <div className="flex items-center gap-2"><span className="text-sky-500">发送缓冲区</span><span className="text-text-tertiary">→</span><span className="text-text-secondary">内核TCP协议栈封装</span><span className="text-text-tertiary">→</span><span className="text-text-secondary">网卡发送</span></div>
          <div className="flex items-center gap-2"><span className="text-text-secondary">网卡接收</span><span className="text-text-tertiary">→</span><span className="text-text-secondary">内核TCP解封装</span><span className="text-text-tertiary">→</span><span className="text-emerald-500">接收缓冲区</span></div>
          <div className="flex items-center gap-2"><span className="text-emerald-500">接收缓冲区</span><span className="text-text-tertiary">→</span><span className="text-text-primary font-medium">read() 读取到应用</span></div>
        </div>
      )}
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1 mt-3">
        <div className="font-medium text-text-primary">缓冲区与流量控制</div>
        <div>• TCP滑动窗口大小受接收缓冲区剩余空间控制</div>
        <div>• 发送缓冲区满时，write()阻塞或返回EAGAIN</div>
        <div>• 接收缓冲区满时，TCP通告零窗口，发送方暂停</div>
        <div>• setsockopt(SO_SNDBUF/SO_RCVBUF)可调整缓冲区大小</div>
      </div>
    </div>
  );
}
export default SocketBufferVisualizer;
