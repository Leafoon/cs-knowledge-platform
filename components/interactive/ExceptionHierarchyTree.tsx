"use client";
import React from 'react';
import { Card } from '@/components/ui/Card';

const ExceptionHierarchyTree = () => {
    return (
        <Card className="p-6 my-8 bg-slate-50">
             <h3 className="font-bold mb-4">Python Exception Hierarchy (Simplified)</h3>
             <ul className="list-none space-y-2 font-mono text-sm">
                 <li>
                     <span className="font-bold text-slate-700">BaseException</span>
                     <ul className="pl-6 border-l-2 border-slate-300 ml-2 mt-2 space-y-2">
                         <li>SystemExit</li>
                         <li>KeyboardInterrupt</li>
                         <li>
                             <span className="font-bold text-blue-700">Exception</span> <span className="text-xs text-slate-500">(User code mostly inherits this)</span>
                             <ul className="pl-6 border-l-2 border-blue-200 ml-2 mt-2 space-y-2">
                                 <li>ArithmeticError
                                     <ul className="pl-6 border-l-2 border-slate-200 ml-2 mt-1">
                                         <li>ZeroDivisionError</li>
                                         <li>OverflowError</li>
                                     </ul>
                                 </li>
                                 <li>TypeError</li>
                                 <li>ValueError</li>
                                 <li>LookupError
                                     <ul className="pl-6 border-l-2 border-slate-200 ml-2 mt-1">
                                         <li>IndexError</li>
                                         <li>KeyError</li>
                                     </ul>
                                 </li>
                             </ul>
                         </li>
                     </ul>
                 </li>
             </ul>
        </Card>
    );
};

export default ExceptionHierarchyTree;