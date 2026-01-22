"use client";

import React, { useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';

const IntegerMemoryLayout = () => {
  const [value, setValue] = useState(12345678901234567890);
  
  // Simulation of Python's ob_digit array (base 2^30)
  // For visualization, we'll use a smaller base (e.g., 2^15 = 32768) to show more chunks
  const BASE = 10000; 
  
  const getDigits = (num: number) => {
    const digits = [];
    let temp = BigInt(num);
    const bigBase = BigInt(BASE);
    
    if (temp === 0n) return [0];
    
    while (temp > 0n) {
      digits.push(Number(temp % bigBase));
      temp = temp / bigBase;
    }
    return digits;
  };

  const digits = getDigits(value);
  const size = digits.length;
  // PyVarObject header size (approx) + digit array
  const totalSize = 24 + size * 4; 

  return (
    <Card className="p-6 my-8 bg-slate-50 border-slate-200">
      <div className="space-y-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-800">CPython Integer Layout Simulator</h3>
            <p className="text-sm text-slate-600">See how Python stores large integers in the <code>ob_digit</code> array</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold">Input Number:</span>
            <input 
              type="number" 
              value={value.toString()} 
              onChange={(e) => setValue(Number(e.target.value) || 0)}
              className="px-3 py-1 border rounded bg-white w-48 font-mono text-sm"
            />
          </div>
        </div>

        {/* Memory View */}
        <div className="border border-slate-300 rounded-lg overflow-hidden bg-white shadow-sm">
          {/* Header Info */}
          <div className="bg-slate-100 p-3 border-b border-slate-200 grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
            <div>
              <span className="text-slate-500 block">PyObject_HEAD</span>
              <span className="font-bold text-blue-600">refcnt: 1, type: int</span>
            </div>
            <div>
              <span className="text-slate-500 block">ob_size</span>
              <span className="font-bold text-purple-600">{size} (chunks)</span>
            </div>
            <div>
              <span className="text-slate-500 block">Approx Memory</span>
              <span className="font-bold text-slate-700">{totalSize} bytes</span>
            </div>
          </div>

          {/* Array Visualization */}
          <div className="p-6 overflow-x-auto">
            <div className="flex items-center space-x-1">
              {/* C Structure Start */}
              <div className="flex-shrink-0 w-24 h-20 border-2 border-dashed border-slate-300 rounded bg-slate-50 flex flex-col items-center justify-center text-xs text-slate-400 mr-4">
                <span>PyObject</span>
                <span>Header</span>
              </div>

              {/* Digits */}
              {digits.map((digit, index) => (
                <div key={index} className="flex-shrink-0 flex flex-col items-center group relative">
                  <div className="w-16 h-20 border-2 border-indigo-200 bg-indigo-50 rounded flex items-center justify-center font-mono font-bold text-indigo-700 shadow-sm group-hover:bg-indigo-100 transition-colors">
                    {digit}
                  </div>
                  <div className="mt-2 text-xs text-slate-400 font-mono">
                    ob_digit[{index}]
                  </div>
                  
                  {/* Tooltip for value calculation */}
                  <div className="absolute bottom-full mb-2 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-800 text-white text-xs p-2 rounded w-40 z-10 pointer-events-none">
                    Value contribution:
                    <br/>
                    {digit} × BASE^{index}
                  </div>
                </div>
              ))}
              
              <div className="w-8 border-b-2 border-dotted border-slate-300 mx-2"></div>
              
            </div>
            
            <div className="mt-6 text-xs text-slate-500 bg-yellow-50 p-2 rounded border border-yellow-100">
              <span className="font-bold">Note:</span> Actual CPython uses base $2^{30}$ (30-bit). Here we simulate with base 10000 for readability.
              As the number grows, the array automatically expands (Arbitrary Precision).
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default IntegerMemoryLayout;