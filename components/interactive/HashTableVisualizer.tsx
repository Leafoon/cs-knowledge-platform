"use client";

import React, { useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';

const HashTableVisualizer = () => {
  const SIZE = 8;
  const [table, setTable] = useState(Array(SIZE).fill(null));
  const [keyInput, setKeyInput] = useState("");
  const [valueInput, setValueInput] = useState("");
  const [log, setLog] = useState<string[]>([]);

  const hashFunction = (key: string) => {
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
        hash = (hash << 5) - hash + key.charCodeAt(i);
        hash |= 0; 
    }
    return Math.abs(hash);
  };

  const handleInsert = () => {
    if (!keyInput) return;
    
    let newTable = [...table];
    const hash = hashFunction(keyInput);
    let index = hash % SIZE;
    let probes = 0;
    const newLog = [`Hash('${keyInput}') = ${hash}`, `Initial Index = ${hash} % ${SIZE} = ${index}`];

    // Open addressing probe
    while (newTable[index] && newTable[index].key !== keyInput && probes < SIZE) {
        newLog.push(`Collision at index ${index}! Probing next...`);
        index = (index + 1) % SIZE;
        probes++;
    }

    if (probes >= SIZE) {
        newLog.push("Table full!");
    } else {
        newTable[index] = { key: keyInput, value: valueInput || keyInput, hash: hash };
        newLog.push(`Inserted at index ${index}`);
    }

    setTable(newTable);
    setLog(newLog);
    setKeyInput("");
    setValueInput("");
  };

  return (
    <Card className="p-6 my-8 space-y-6 bg-slate-50">
      <div className="flex justify-between items-center">
        <h3 className="font-bold text-lg">Python Dict (Hash Table) Visualizer</h3>
        <Button variant="secondary" onClick={() => { setTable(Array(SIZE).fill(null)); setLog([]); }}>Reset</Button>
      </div>

      <div className="flex gap-2">
        <input 
            className="border p-2 rounded w-24" 
            placeholder="Key" 
            value={keyInput}
            onChange={e => setKeyInput(e.target.value)}
        />
        <input 
            className="border p-2 rounded w-24" 
            placeholder="Value" 
            value={valueInput}
            onChange={e => setValueInput(e.target.value)}
        />
        <Button onClick={handleInsert}>Set Item</Button>
      </div>

      <div className="grid grid-cols-8 gap-2">
        {table.map((item, i) => (
            <div key={i} className={`
                h-24 border-2 rounded flex flex-col items-center justify-center text-xs p-1 relative transition-all
                ${item ? 'bg-green-100 border-green-500' : 'bg-white border-slate-200 border-dashed'}
            `}>
                <span className="absolute top-1 left-1 text-slate-400 font-mono text-[10px]">{i}</span>
                {item ? (
                    <>
                        <span className="font-bold text-slate-800">{item.key}</span>
                        <span className="text-slate-500">→</span>
                        <span className="text-slate-600 truncate w-full text-center">{item.value}</span>
                        <span className="text-[9px] text-green-700 mt-1 opacity-50">{item.hash}</span>
                    </>
                ) : (
                    <span className="text-slate-300">Empty</span>
                )}
            </div>
        ))}
      </div>

      <div className="bg-slate-900 text-green-400 p-3 rounded font-mono text-sm h-32 overflow-y-auto">
        {log.length === 0 ? <span className="opacity-50">// Operations log...</span> : log.map((l, i) => <div key={i}>{'>'} {l}</div>)}
      </div>
    </Card>
  );
};

export default HashTableVisualizer;