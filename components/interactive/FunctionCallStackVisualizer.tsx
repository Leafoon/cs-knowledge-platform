"use client";

import React, { useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';

const FunctionCallStackVisualizer = () => {
  const [stack, setStack] = useState<{name: string, vars: string[]}[]>([]);

  const pushFrame = (name: string, vars: string[]) => {
    setStack(prev => [...prev, { name, vars }]);
  };

  const popFrame = () => {
    setStack(prev => prev.slice(0, -1));
  };

  const demoRecursion = async () => {
    setStack([]);
    const steps = [
        () => pushFrame('<module>', ['n = 3']),
        () => pushFrame('fact(3)', ['n = 3']),
        () => pushFrame('fact(2)', ['n = 2']),
        () => pushFrame('fact(1)', ['n = 1']),
        () => popFrame(),
        () => popFrame(),
        () => popFrame(),
    ];

    for (let i = 0; i < steps.length; i++) {
        steps[i]();
        await new Promise(r => setTimeout(r, 800));
    }
  };

  return (
    <Card className="p-6 my-8 bg-slate-50">
      <div className="flex justify-between mb-4">
        <h3 className="font-bold">Call Stack Visualization</h3>
        <div className="space-x-2">
            <Button onClick={() => pushFrame('func()', ['x=1'])} size="sm">Push Frame</Button>
            <Button onClick={popFrame} variant="secondary" size="sm">Pop Frame</Button>
            <Button onClick={demoRecursion} variant="secondary" size="sm">Play Recursion Demo</Button>
        </div>
      </div>

      <div className="flex flex-col-reverse items-center justify-end min-h-[300px] border-l-4 border-b-4 border-slate-300 p-4 bg-slate-100 rounded-lg w-full max-w-md mx-auto relative">
         {stack.length === 0 && <div className="text-slate-400 absolute top-1/2">Empty Stack</div>}
         {stack.map((frame, i) => (
             <div key={i} className="w-full bg-white border-2 border-slate-600 rounded mb-1 p-3 shadow-md animate-in slide-in-from-top-4 duration-300">
                 <div className="font-mono font-bold border-b pb-1 mb-1 text-purple-700">{frame.name}</div>
                 <div className="text-xs font-mono text-slate-600">
                     {frame.vars.map((v, v_i) => <div key={v_i}>{v}</div>)}
                 </div>
             </div>
         ))}
      </div>
      <div className="text-center mt-2 text-sm text-slate-500">Stack grows upwards in UI (or downwards in memory addresses)</div>
    </Card>
  );
};

export default FunctionCallStackVisualizer;