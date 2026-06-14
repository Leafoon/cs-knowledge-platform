'use client';

import React, { useState, useEffect } from 'react';

export function BankConflictDemo() {
  const [showSwizzle, setShowSwizzle] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  const banks = 32;
  const threads = 32;

  useEffect(() => {
    if (isAnimating) {
      const timer = setInterval(() => {
        setAnimationStep((prev) => {
          if (prev >= 7) {
            setIsAnimating(false);
            return 0;
          }
          return prev + 1;
        });
      }, 500);
      return () => clearInterval(timer);
    }
  }, [isAnimating]);

  const getThreadAccess = (thread: number, useSwizzle: boolean) => {
    if (useSwizzle) {
      return (thread ^ (thread >> 2)) % banks;
    }
    return thread % banks;
  };

  const getConflictCount = (useSwizzle: boolean) => {
    const accesses = Array.from({ length: threads }, (_, i) => getThreadAccess(i, useSwizzle));
    const bankCounts = new Array(banks).fill(0);
    accesses.forEach((bank) => bankCounts[bank]++);
    return bankCounts.filter((count) => count > 1).length;
  };

  return (
    <div className="p-6 bg-gray-900 rounded-xl">
      <h2 className="text-2xl font-bold text-white mb-6">Bank Conflict 演示</h2>
      
      <div className="grid grid-cols-2 gap-6">
        {/* Without swizzle */}
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-red-400 font-bold">无 Swizzle</h3>
            <span className="text-gray-400 text-sm">
              冲突: {getConflictCount(false)} Banks
            </span>
          </div>
          
          <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${banks}, 1fr)` }}>
            {Array.from({ length: threads }).map((_, i) => {
              const bank = getThreadAccess(i, false);
              return (
                <div
                  key={i}
                  className="w-6 h-6 flex items-center justify-center text-[10px] font-mono rounded"
                  style={{
                    backgroundColor: getConflictCount(false) > 0 ? '#EF4444' : '#10B981',
                    opacity: animationStep >= i % 8 ? 1 : 0.3,
                  }}
                >
                  {bank}
                </div>
              );
            })}
          </div>
          
          <div className="mt-4 text-gray-400 text-sm">
            线程访问同一Bank，导致串行化
          </div>
        </div>
        
        {/* With swizzle */}
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-green-400 font-bold">使用 Swizzle</h3>
            <span className="text-gray-400 text-sm">
              冲突: {getConflictCount(true)} Banks
            </span>
          </div>
          
          <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${banks}, 1fr)` }}>
            {Array.from({ length: threads }).map((_, i) => {
              const bank = getThreadAccess(i, true);
              return (
                <div
                  key={i}
                  className="w-6 h-6 flex items-center justify-center text-[10px] font-mono rounded"
                  style={{
                    backgroundColor: '#10B981',
                    opacity: animationStep >= i % 8 ? 1 : 0.3,
                  }}
                >
                  {bank}
                </div>
              );
            })}
          </div>
          
          <div className="mt-4 text-gray-400 text-sm">
            XOR Swizzle消除冲突，所有Bank并行访问
          </div>
        </div>
      </div>
      
      {/* Animation control */}
      <div className="mt-6 flex justify-center gap-4">
        <button
          onClick={() => {
            setAnimationStep(0);
            setIsAnimating(true);
          }}
          disabled={isAnimating}
          className={`px-6 py-2 rounded-lg font-bold transition-all ${
            isAnimating
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
          }`}
        >
          {isAnimating ? '播放中...' : '播放动画'}
        </button>
        <button
          onClick={() => setShowSwizzle(!showSwizzle)}
          className="px-6 py-2 bg-gray-700 rounded-lg text-white hover:bg-gray-600"
        >
          {showSwizzle ? '隐藏' : '显示'} Swizzle 代码
        </button>
      </div>
      
      {/* Swizzle code */}
      {showSwizzle && (
        <div className="mt-6 bg-gray-800 rounded-lg p-4">
          <h3 className="text-white font-bold mb-2">Swizzle 实现</h3>
          <pre className="text-sm font-mono text-green-400">
{`# XOR Swizzle 实现
def swizzle_address(thread_id, bank_id):
    # 线程ID与BankID进行XOR
    return thread_id ^ (bank_id >> 2)

# 在 TileLang 中使用
A_shared = T.alloc_swizzle(
    (BM, BK),
    "float16",
    swizzle_banks=32
)`}
          </pre>
        </div>
      )}
    </div>
  );
}
