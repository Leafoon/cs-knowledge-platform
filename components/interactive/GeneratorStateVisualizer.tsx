"use client";
import React, { useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';

const GeneratorStateVisualizer = () => {
    const [yieldedValues, setYieldedValues] = useState<number[]>([]);
    const [state, setState] = useState<'suspended' | 'running' | 'closed'>('suspended');
    
    // Simulating:
    // def gen():
    //   yield 10
    //   yield 20
    //   yield 30

    const next = () => {
        if (state === 'closed') return;
        
        setState('running');
        
        setTimeout(() => {
            const nextVal = (yieldedValues.length + 1) * 10;
            if (nextVal > 30) {
                setState('closed');
            } else {
                setYieldedValues(prev => [...prev, nextVal]);
                setState('suspended');
            }
        }, 600);
    };

    return (
        <Card className="p-6 my-8 bg-slate-50">
             <div className="flex justify-between items-center mb-6">
                <h3 className="font-bold">Generator State Machine</h3>
                <div className="flex items-center gap-2">
                    Status: 
                    <Badge variant={state === 'running' ? 'warning' : state === 'suspended' ? 'success' : 'default'}>
                        {state.toUpperCase()}
                    </Badge>
                </div>
             </div>

             <div className="flex gap-8 items-start">
                 <div className="bg-white p-4 rounded border font-mono text-sm">
                     <div className={yieldedValues.length === 0 && state === 'running' ? 'bg-yellow-100' : ''}>def gen():</div>
                     <div className={yieldedValues.length === 1 && state === 'suspended' ? 'bg-green-100' : ''}>  yield 10</div>
                     <div className={yieldedValues.length === 2 && state === 'suspended' ? 'bg-green-100' : ''}>  yield 20</div>
                     <div className={yieldedValues.length === 3 && state === 'suspended' ? 'bg-green-100' : ''}>  yield 30</div>
                 </div>

                 <div className="flex-1 space-y-4">
                     <div>
                        Yielded Outputs:
                        <div className="flex gap-2 mt-2">
                            {yieldedValues.map(v => (
                                <div key={v} className="w-10 h-10 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold shadow-lg animate-in fade-in zoom-in">{v}</div>
                            ))}
                            {state === 'closed' && <div className="text-red-500 font-bold self-center">StopIteration!</div>}
                        </div>
                     </div>
                     
                     <Button onClick={next} disabled={state === 'closed' || state === 'running'} className="w-full">
                        {state === 'closed' ? 'Exhausted' : 'next()'}
                     </Button>
                 </div>
             </div>
        </Card>
    );
};

export default GeneratorStateVisualizer;