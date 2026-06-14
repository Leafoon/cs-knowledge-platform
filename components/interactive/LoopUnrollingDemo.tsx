'use client';

import { useState } from 'react';

const beforeCode = `for (int i = 0; i < N; i++) {
  sum += a[i] * b[i];
}`;

const afterCode = `for (int i = 0; i < N; i += 4) {
  sum0 += a[i] * b[i];
  sum1 += a[i+1] * b[i+1];
  sum2 += a[i+2] * b[i+2];
  sum3 += a[i+3] * b[i+3];
}
sum = sum0 + sum1 + sum2 + sum3;`;

export function LoopUnrollingDemo() {
  const [showAfter, setShowAfter] = useState(false);

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <h2 className="text-xl font-bold mb-4 text-gray-800">循环展开优化演示</h2>
      
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setShowAfter(false)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            !showAfter ? 'bg-red-500 text-white' : 'bg-gray-100'
          }`}
        >
          展开前
        </button>
        <button
          onClick={() => setShowAfter(true)}
          className={`px-4 py-2 rounded-lg transition-colors ${
            showAfter ? 'bg-green-500 text-white' : 'bg-gray-100'
          }`}
        >
          展开后 (4x)
        </button>
      </div>

      <div className="bg-gray-900 rounded-xl p-4 overflow-x-auto">
        <pre className="text-sm text-gray-100 font-mono">
          {showAfter ? afterCode : beforeCode}
        </pre>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className={`p-3 rounded-lg ${!showAfter ? 'bg-red-50' : 'bg-gray-50'}`}>
          <div className="text-sm text-gray-500">循环迭代次数</div>
          <div className="text-lg font-bold text-gray-800">{showAfter ? 'N/4' : 'N'}</div>
        </div>
        <div className={`p-3 rounded-lg ${showAfter ? 'bg-green-50' : 'bg-gray-50'}`}>
          <div className="text-sm text-gray-500">指令级并行</div>
          <div className="text-lg font-bold text-gray-800">{showAfter ? '4路' : '1路'}</div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700">
        💡 循环展开减少分支开销，提高指令级并行度
      </div>
    </div>
  );
}