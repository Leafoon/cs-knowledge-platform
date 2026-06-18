"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Users, ArrowRight } from "lucide-react";

const MODES = [
  { id: "hierarchical", name: "主从模式", agents: ["主管", "员工1", "员工2"] },
  { id: "peer", name: "对等模式", agents: ["Agent A", "Agent B", "Agent C"] },
];

export function MultiAgentCollaborationModes() {
  const [selected, setSelected] = useState(0);
  const mode = MODES[selected];

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 dark:from-slate-900 dark:to-slate-800 rounded-2xl shadow-xl">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4">多Agent协作模式</h3>
      <div className="flex gap-3 mb-6">
        {MODES.map((m, i) => (
          <button key={m.id} onClick={() => setSelected(i)}
            className={`px-4 py-2 rounded-lg ${selected === i ? "bg-violet-600 text-white" : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300"}`}>
            {m.name}
          </button>
        ))}
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-center gap-4">
          {mode.agents.map((a, i) => (
            <React.Fragment key={i}>
              <div className="w-24 h-24 rounded-full bg-violet-100 dark:bg-violet-900/30 flex items-center justify-center">
                <Users className="w-8 h-8 text-violet-500" />
              </div>
              {i < mode.agents.length - 1 && <ArrowRight className="w-6 h-6 text-violet-300" />}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}
