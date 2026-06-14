'use client';
import { useState, useEffect, useRef } from 'react';

export function SearchConvergenceCurve() {
  const [data, setData] = useState<number[]>([45]);
  const [iteration, setIteration] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [searchMethod, setSearchMethod] = useState<'random' | 'bayesian' | 'evolutionary'>('bayesian');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maxIterations = 50;

  const methodColors = { random: '#f59e0b', bayesian: '#06b6d4', evolutionary: '#a855f7' };
  const methodLabels = { random: '随机搜索', bayesian: '贝叶斯优化', evolutionary: '进化算法' };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (plotH / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, h - padding.bottom);
    ctx.lineTo(w - padding.right, h - padding.bottom);
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('搜索迭代次数', w / 2, h - 8);
    for (let i = 0; i <= 5; i++) {
      ctx.fillText(`${i * 10}`, padding.left + (plotW / 5) * i, h - padding.bottom + 15);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      ctx.fillText(`${100 - i * 20}%`, padding.left - 8, padding.top + (plotH / 5) * i + 4);
    }

    if (data.length < 2) return;

    // Draw curve
    ctx.strokeStyle = methodColors[searchMethod];
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = padding.left + (i / maxIterations) * plotW;
      const y = padding.top + plotH - (v / 100) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw points
    data.forEach((v, i) => {
      const x = padding.left + (i / maxIterations) * plotW;
      const y = padding.top + plotH - (v / 100) * plotH;
      ctx.fillStyle = i === data.length - 1 ? '#ffffff' : methodColors[searchMethod];
      ctx.beginPath();
      ctx.arc(x, y, i === data.length - 1 ? 4 : 2, 0, Math.PI * 2);
      ctx.fill();
    });

    // Best value marker
    const bestVal = Math.max(...data);
    const bestIdx = data.indexOf(bestVal);
    const bx = padding.left + (bestIdx / maxIterations) * plotW;
    const by = padding.top + plotH - (bestVal / 100) * plotH;
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, by);
    ctx.lineTo(w - padding.right, by);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#22c55e';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`最优: ${bestVal.toFixed(1)}%`, w - padding.right - 60, by - 6);
  }, [data, searchMethod]);

  const runSearch = () => {
    if (isRunning) { setIsRunning(false); return; }
    setIsRunning(true);
    setData([45 + Math.random() * 10]);
    setIteration(0);

    const timer = setInterval(() => {
      setIteration(prev => {
        if (prev >= maxIterations - 1) { setIsRunning(false); clearInterval(timer); return prev; }
        setData(d => {
          const last = d[d.length - 1];
          let improvement: number;
          if (searchMethod === 'bayesian') {
            improvement = Math.max(0.5, (95 - last) * 0.15 + Math.random() * 2);
          } else if (searchMethod === 'evolutionary') {
            improvement = Math.max(0.3, (95 - last) * 0.12 + Math.random() * 3);
          } else {
            improvement = Math.max(0.1, (95 - last) * 0.08 + Math.random() * 2);
          }
          return [...d, Math.min(last + improvement, 97)];
        });
        return prev + 1;
      });
    }, 100);
  };

  const reset = () => {
    setIsRunning(false);
    setData([45 + Math.random() * 10]);
    setIteration(0);
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-cyan-400">搜索收敛曲线</h2>
        <div className="flex gap-2">
          <select value={searchMethod} onChange={e => setSearchMethod(e.target.value as typeof searchMethod)}
            className="bg-gray-700 text-sm rounded px-2 py-1">
            <option value="random">随机搜索</option>
            <option value="bayesian">贝叶斯优化</option>
            <option value="evolutionary">进化算法</option>
          </select>
          <button onClick={runSearch}
            className={`px-4 py-2 rounded-lg text-sm transition-all ${isRunning ? 'bg-red-600' : 'bg-cyan-600 hover:bg-cyan-500'}`}>
            {isRunning ? '⏹ 停止' : '▶ 运行'}
          </button>
          <button onClick={reset} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm">🔄</button>
        </div>
      </div>

      <canvas ref={canvasRef} width={600} height={300} className="w-full rounded-lg border border-gray-700" />

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mt-4">
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-cyan-400">{iteration}/{maxIterations}</div>
          <div className="text-xs text-gray-400">迭代次数</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-green-400">{data.length > 0 ? Math.max(...data).toFixed(1) : '0'}%</div>
          <div className="text-xs text-gray-400">最优性能</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-purple-400">{methodLabels[searchMethod]}</div>
          <div className="text-xs text-gray-400">搜索方法</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-yellow-400">
            {data.length > 1 ? `+${(data[data.length - 1] - data[0]).toFixed(1)}%` : '0%'}
          </div>
          <div className="text-xs text-gray-400">总提升</div>
        </div>
      </div>

      <div className="mt-4 text-xs text-gray-500">
        贝叶斯优化通过代理模型预测最有价值的搜索点，通常在较少迭代内收敛到较优解。
      </div>
    </div>
  );
}
