"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Bug, Play, ToggleLeft, ToggleRight, Terminal, AlertTriangle } from "lucide-react";

interface Mistake {
  id: number;
  name: string;
  code: string;
  normalOutput: string;
  debugWarnings: string[];
}

const MISTAKES: Mistake[] = [
  { id: 1, name: "Unawaited coroutine", code: "async def main():\n    asyncio.sleep(1)  # forgot await", normalOutput: "<coroutine object sleep at 0x...>", debugWarnings: ["RuntimeWarning: coroutine 'sleep' was never awaited", "Source: main(): line 2"] },
  { id: 2, name: "Slow callback", code: "async def handler():\n    time.sleep(0.5)  # blocks loop", normalOutput: "(silently blocks for 0.5s)", debugWarnings: ["Executing <Task> took 0.500 seconds", "Source: handler(): line 2"] },
  { id: 3, name: "Destroyed pending task", code: "async def main():\n    asyncio.create_task(work())\n    # task not awaited, gets GC'd", normalOutput: "(task silently disappears)", debugWarnings: ["Task was destroyed but it is pending!", "task: <Task pending name='work'>", "Source: main(): line 2"] },
  { id: 4, name: "Event loop closed twice", code: "loop = asyncio.new_event_loop()\nloop.run_until_complete(main())\nloop.close()\nloop.close()  # double close", normalOutput: "(silently ignored)", debugWarnings: ["Event loop is closed", "Source: line 4"] },
];

export function DebugModeDemo() {
  const [debugOn, setDebugOn] = useState(false);
  const [selected, setSelected] = useState<number>(0);

  const mistake = MISTAKES[selected];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-orange-50 dark:from-gray-900 dark:to-gray-800 rounded-xl shadow-lg">
      <h3 className="text-xl font-bold text-slate-800 dark:text-gray-100 mb-4 text-center flex items-center justify-center gap-2">
        <Bug className="w-5 h-5 text-orange-500" /> Debug Mode Comparison
      </h3>

      {/* Toggle */}
      <div className="flex items-center justify-center gap-3 mb-4">
        <span className="text-sm text-slate-600 dark:text-gray-300">asyncio debug mode:</span>
        <button onClick={() => setDebugOn(!debugOn)} className="flex items-center gap-2">
          {debugOn ? <ToggleRight className="w-8 h-8 text-emerald-500" /> : <ToggleLeft className="w-8 h-8 text-slate-400" />}
          <span className={`text-sm font-bold ${debugOn ? "text-emerald-600 dark:text-emerald-400" : "text-slate-400"}`}>{debugOn ? "ON" : "OFF"}</span>
        </button>
        <span className="text-xs text-slate-400 ml-2">loop.set_debug(True)</span>
      </div>

      {/* Mistake selector */}
      <div className="flex gap-2 mb-4 flex-wrap justify-center">
        {MISTAKES.map((m, i) => (
          <button key={m.id} onClick={() => setSelected(i)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium border ${selected === i ? "bg-orange-500 text-white border-orange-500" : "bg-white dark:bg-gray-800 text-slate-600 dark:text-gray-300 border-slate-200 dark:border-gray-700 hover:border-orange-300"}`}>
            {m.name}
          </button>
        ))}
      </div>

      {/* Code */}
      <div className="bg-slate-900 rounded-lg p-4 mb-4">
        <div className="flex items-center gap-2 mb-2"><Terminal className="w-3 h-3 text-emerald-400" /><span className="text-xs text-emerald-400 font-mono">code</span></div>
        <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap">{mistake.code}</pre>
      </div>

      {/* Side-by-side output */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Normal mode */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700">
          <div className="text-xs font-bold text-slate-500 dark:text-gray-400 mb-2">Normal Mode Output</div>
          <div className="bg-slate-50 dark:bg-gray-900 rounded p-2 min-h-[4rem]">
            <pre className="text-xs font-mono text-slate-600 dark:text-gray-300 whitespace-pre-wrap">{mistake.normalOutput}</pre>
          </div>
          <p className="text-xs text-slate-400 dark:text-gray-500 mt-2">Problem silently ignored</p>
        </div>

        {/* Debug mode */}
        <div className={`rounded-lg p-4 border-2 ${debugOn ? "border-orange-400 bg-orange-50 dark:bg-orange-900/20" : "border-slate-200 dark:border-gray-700 bg-slate-100 dark:bg-gray-800"}`}>
          <div className="text-xs font-bold text-slate-500 dark:text-gray-400 mb-2">Debug Mode Output</div>
          <div className={`rounded p-2 min-h-[4rem] ${debugOn ? "bg-white dark:bg-gray-800" : "bg-slate-50 dark:bg-gray-900"}`}>
            {debugOn ? (
              <div className="space-y-1">
                {mistake.debugWarnings.map((w, i) => (
                  <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.2 }}
                    className="flex items-start gap-1">
                    <AlertTriangle className="w-3 h-3 text-orange-500 flex-shrink-0 mt-0.5" />
                    <pre className="text-xs font-mono text-orange-700 dark:text-orange-300 whitespace-pre-wrap">{w}</pre>
                  </motion.div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-400 text-center py-4">Turn on debug mode to see warnings</p>
            )}
          </div>
        </div>
      </div>

      <div className="text-center text-xs text-slate-500 dark:text-gray-400">
        Enable with: <code className="bg-slate-100 dark:bg-gray-700 px-1 rounded">asyncio.run(main(), debug=True)</code> or <code className="bg-slate-100 dark:bg-gray-700 px-1 rounded">loop.set_debug(True)</code>
      </div>
    </div>
  );
}
