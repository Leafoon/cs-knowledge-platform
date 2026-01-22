"use client";
import React, { useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';

const DecoratorExecutionFlow = () => {
    const [step, setStep] = useState(0);

    return (
        <Card className="p-6 my-8 bg-slate-50">
            <h3 className="font-bold text-lg mb-4">Decorator "Onion" Model</h3>
            <div className="flex flex-col items-center gap-4">
                <div className={`p-4 border-2 border-dashed border-slate-300 rounded transition-all ${step >= 0 ? 'bg-blue-100 opacity-100' : 'opacity-0'}`}>
                    <span className="font-mono text-sm text-blue-700">@timer (Enter)</span>
                    <div className={`mt-2 p-4 border-2 border-dashed border-slate-300 rounded transition-all ${step >= 1 ? 'bg-green-100 opacity-100' : 'opacity-0'}`}>
                         <span className="font-mono text-sm text-green-700">@auth (Enter)</span>
                         <div className={`mt-2 p-4 bg-white border-2 border-slate-800 rounded transition-all shadow-md ${step >= 2 ? 'opacity-100 transform scale-100' : 'opacity-0 transform scale-95'}`}>
                            <span className="font-bold">Original Function Logic</span>
                         </div>
                         <span className={`block mt-2 font-mono text-sm text-green-700 transition-all ${step >= 3 ? 'opacity-100' : 'opacity-0'}`}>@auth (Exit)</span>
                    </div>
                    <span className={`block mt-2 font-mono text-sm text-blue-700 transition-all ${step >= 4 ? 'opacity-100' : 'opacity-0'}`}>@timer (Exit)</span>
                </div>
                
                <div className="flex gap-2 mt-4">
                    <Button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0} variant='outline'>Pre</Button>
                    <Button onClick={() => setStep(Math.min(4, step + 1))} disabled={step === 4}>Next Step</Button>
                    <div className="text-sm self-center text-slate-500 ml-2">Step {step}/4</div>
                </div>
            </div>
        </Card>
    );
};

export default DecoratorExecutionFlow;