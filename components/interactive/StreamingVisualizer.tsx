'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, Zap } from 'lucide-react';

const sampleText = "Once upon a time, in a world powered by AI, there lived a curious robot named Claude. Claude loved to learn and explore new concepts every day.";

export default function StreamingVisualizer() {
  const [mode, setMode] = useState<'streaming' | 'non-streaming'>('streaming');
  const [isPlaying, setIsPlaying] = useState(false);
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [stats, setStats] = useState({
    totalTime: 0,
    firstTokenTime: 0,
    tokensPerSecond: 0
  });
  const [startTime, setStartTime] = useState<number | null>(null);

  useEffect(() => {
    if (!isPlaying) return;

    if (mode === 'streaming') {
      // 流式模式：逐字显示
      const interval = setInterval(() => {
        if (currentIndex < sampleText.length) {
          if (currentIndex === 0) {
            setStartTime(Date.now());
            setStats(prev => ({ ...prev, firstTokenTime: 50 }));
          }
          
          setDisplayedText(sampleText.slice(0, currentIndex + 1));
          setCurrentIndex(prev => prev + 1);
        } else {
          setIsPlaying(false);
          const totalTime = startTime ? (Date.now() - startTime) / 1000 : 0;
          setStats(prev => ({
            ...prev,
            totalTime,
            tokensPerSecond: totalTime > 0 ? sampleText.split(' ').length / totalTime : 0
          }));
        }
      }, 50);

      return () => clearInterval(interval);
    } else {
      // 非流式模式：模拟等待后一次性显示
      const timeout = setTimeout(() => {
        setDisplayedText(sampleText);
        setIsPlaying(false);
        setStats({
          totalTime: 3.0,
          firstTokenTime: 3000,
          tokensPerSecond: sampleText.split(' ').length / 3.0
        });
      }, 3000);

      return () => clearTimeout(timeout);
    }
  }, [isPlaying, currentIndex, mode, startTime]);

  const togglePlay = () => {
    if (!isPlaying) {
      reset();
      setIsPlaying(true);
    } else {
      setIsPlaying(false);
    }
  };

  const reset = () => {
    setDisplayedText('');
    setCurrentIndex(0);
    setIsPlaying(false);
    setStartTime(null);
    setStats({
      totalTime: 0,
      firstTokenTime: 0,
      tokensPerSecond: 0
    });
  };

  const switchMode = (newMode: 'streaming' | 'non-streaming') => {
    setMode(newMode);
    reset();
  };

  return (
    <div className="w-full bg-gradient-to-br from-slate-900 via-emerald-900/20 to-slate-900 rounded-2xl p-8 shadow-2xl">
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold text-white mb-2">
          流式 vs 非流式输出对比
        </h3>
        <p className="text-slate-400">
          体验流式输出带来的用户体验提升
        </p>
      </div>

      {/* 模式选择 */}
      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => switchMode('streaming')}
          className={`
            px-6 py-3 rounded-lg font-medium transition-all flex items-center gap-2
            ${mode === 'streaming'
              ? 'bg-gradient-to-r from-green-600 to-emerald-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }
          `}
        >
          <Zap className="w-5 h-5" />
          流式输出
        </button>
        <button
          onClick={() => switchMode('non-streaming')}
          className={`
            px-6 py-3 rounded-lg font-medium transition-all
            ${mode === 'non-streaming'
              ? 'bg-gradient-to-r from-orange-600 to-red-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }
          `}
        >
          非流式输出
        </button>
      </div>

      {/* 输出显示区 */}
      <div className="bg-slate-800 rounded-xl p-6 mb-6 min-h-[200px]">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-3">
            <div className={`
              w-3 h-3 rounded-full
              ${isPlaying
                ? 'bg-green-500 animate-pulse'
                : displayedText
                  ? 'bg-blue-500'
                  : 'bg-slate-600'
              }
            `} />
            <span className="text-slate-400 text-sm">
              {isPlaying ? '生成中...' : displayedText ? '完成' : '等待开始'}
            </span>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={togglePlay}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg 
                       font-medium transition-colors flex items-center gap-2"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4" />
                  暂停
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  {displayedText ? '重新' : ''}播放
                </>
              )}
            </button>
            <button
              onClick={reset}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg 
                       font-medium transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="bg-slate-900 rounded-lg p-4 min-h-[120px]">
          {mode === 'non-streaming' && isPlaying && !displayedText ? (
            <div className="flex items-center justify-center h-[100px]">
              <motion.div
                className="flex gap-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-3 h-3 bg-orange-500 rounded-full"
                    animate={{
                      scale: [1, 1.5, 1],
                      opacity: [0.3, 1, 0.3]
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 1,
                      delay: i * 0.2
                    }}
                  />
                ))}
              </motion.div>
              <span className="ml-4 text-slate-400">等待完整响应...</span>
            </div>
          ) : (
            <p className="text-slate-200 leading-relaxed">
              {displayedText}
              {isPlaying && mode === 'streaming' && (
                <motion.span
                  className="inline-block w-1 h-5 bg-green-500 ml-1"
                  animate={{ opacity: [1, 0] }}
                  transition={{ repeat: Infinity, duration: 0.8 }}
                />
              )}
            </p>
          )}
        </div>
      </div>

      {/* 统计信息 */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        <div className="bg-slate-800 rounded-xl p-4">
          <div className="text-slate-400 text-sm mb-1">首个 Token 时间</div>
          <div className="text-2xl font-bold text-white">
            {stats.firstTokenTime > 0 ? `${stats.firstTokenTime}ms` : '--'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {mode === 'streaming' ? '极快' : '需等待'}
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-4">
          <div className="text-slate-400 text-sm mb-1">总耗时</div>
          <div className="text-2xl font-bold text-white">
            {stats.totalTime > 0 ? `${stats.totalTime.toFixed(2)}s` : '--'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {mode === 'streaming' ? '逐步显示' : '一次性显示'}
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-4">
          <div className="text-slate-400 text-sm mb-1">Tokens/秒</div>
          <div className="text-2xl font-bold text-white">
            {stats.tokensPerSecond > 0 ? stats.tokensPerSecond.toFixed(1) : '--'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            处理速率
          </div>
        </div>
      </div>

      {/* 对比表格 */}
      <div className="bg-slate-800 rounded-xl p-6">
        <h4 className="text-lg font-semibold text-white mb-4">特性对比</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-4 text-slate-300">特性</th>
                <th className="text-left py-3 px-4 text-slate-300">流式输出</th>
                <th className="text-left py-3 px-4 text-slate-300">非流式输出</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-slate-400">首个响应时间</td>
                <td className="py-3 px-4 text-green-400">极快 (~50ms)</td>
                <td className="py-3 px-4 text-orange-400">慢 (~3s)</td>
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-slate-400">用户体验</td>
                <td className="py-3 px-4 text-green-400">优秀（实时反馈）</td>
                <td className="py-3 px-4 text-orange-400">一般（需等待）</td>
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-slate-400">实现复杂度</td>
                <td className="py-3 px-4 text-orange-400">中（需处理流）</td>
                <td className="py-3 px-4 text-green-400">低（简单调用）</td>
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-slate-400">内存占用</td>
                <td className="py-3 px-4 text-green-400">低（逐块处理）</td>
                <td className="py-3 px-4 text-orange-400">高（完整缓存）</td>
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-slate-400">适用场景</td>
                <td className="py-3 px-4 text-slate-300">聊天界面、实时生成</td>
                <td className="py-3 px-4 text-slate-300">批量处理、后台任务</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 代码示例 */}
      <div className="mt-6 bg-slate-900 rounded-xl p-6">
        <div className="text-sm font-semibold text-slate-300 mb-3">代码示例:</div>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-green-400 mb-2">✓ 流式输出</div>
            <pre className="text-green-400 font-mono text-xs overflow-x-auto">
{`# 异步流式
async for chunk in chain.astream(input):
    print(chunk, end="", flush=True)
    
# 用户立即看到输出`}
            </pre>
          </div>
          <div>
            <div className="text-xs text-orange-400 mb-2">非流式输出</div>
            <pre className="text-orange-400 font-mono text-xs overflow-x-auto">
{`# 阻塞调用
result = chain.invoke(input)
print(result)

# 用户需等待完整响应`}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
