"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/Tabs";
import { InlineMath } from "@/components/ui/Math";

export function OnPolicyVsOffPolicy() {
    return (
        <Card className="p-6 w-full max-w-4xl mx-auto bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            <h3 className="text-xl font-bold mb-6 text-center">On-policy vs Off-policy 学习机制对比</h3>

            <Tabs defaultValue="on-policy" className="w-full">
                <TabsList className="grid w-full grid-cols-2 mb-8">
                    <TabsTrigger value="on-policy">On-Policy (同策略)</TabsTrigger>
                    <TabsTrigger value="off-policy">Off-Policy (异策略)</TabsTrigger>
                </TabsList>

                <div className="h-64 relative bg-slate-50 dark:bg-slate-800 rounded-xl border border-dashed border-slate-300 dark:border-slate-700 overflow-hidden flex items-center justify-center p-8">

                    <TabsContent value="on-policy" className="w-full h-full flex items-center justify-center relative mt-0">
                        {/* On-Policy Diagram */}
                        <motion.div
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}
                            className="flex flex-col items-center gap-6"
                        >
                            <div className="flex items-center gap-8">
                                <div className="flex flex-col items-center">
                                    <div className="w-24 h-24 rounded-full bg-blue-100 dark:bg-blue-900/30 border-4 border-blue-500 flex flex-col items-center justify-center relative shadow-lg">
                                        <span className="text-3xl">🤖</span>
                                        <span className="font-bold text-blue-700 dark:text-blue-300">Agent</span>
                                        <div className="absolute -bottom-8 font-mono text-sm bg-blue-500 text-white px-2 py-0.5 rounded">Strategy <InlineMath>{"\\pi"}</InlineMath></div>
                                    </div>
                                    <div className="mt-10 text-xs text-slate-500 text-center max-w-[120px]">
                                        既负责产生行为<br />又负责被优化
                                    </div>
                                </div>

                                <motion.div
                                    className="h-1 w-32 bg-slate-300 relative"
                                    initial={{ width: 0 }} animate={{ width: 128 }}
                                >
                                    <motion.div
                                        className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-orange-500 rounded-full"
                                        animate={{
                                            left: ["0%", "100%", "0%"],
                                            backgroundColor: ["#f97316", "#22c55e", "#f97316"]
                                        }}
                                        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                                    />
                                    <div className="absolute -top-6 w-full text-center text-xs font-mono text-slate-500">
                                        Data (Experience)
                                    </div>
                                </motion.div>

                                <div className="flex flex-col items-center">
                                    <div className="w-24 h-24 rounded-full bg-green-100 dark:bg-green-900/30 border-4 border-dashed border-green-500 flex flex-col items-center justify-center relative opacity-50 grayscale">
                                        <span className="text-3xl">🎯</span>
                                        <span className="font-bold text-green-700 dark:text-green-300">Target</span>
                                        <div className="absolute -bottom-8 font-mono text-sm bg-slate-500 text-white px-2 py-0.5 rounded">Strategy $\pi$</div>
                                    </div>
                                    <div className="mt-10 text-xs text-slate-500 text-center max-w-[120px]">
                                        Target = Behavior<br />(Self-Improvement)
                                    </div>
                                </div>
                            </div>

                            <div className="bg-blue-50 p-2 rounded text-xs text-blue-600 border border-blue-200 mt-4">
                                "我只能从我当前的行为中学习，一旦更新策略，旧经验就作废了。"
                            </div>
                        </motion.div>
                    </TabsContent>

                    <TabsContent value="off-policy" className="w-full h-full flex items-center justify-center relative mt-0">
                        {/* Off-Policy Diagram */}
                        <motion.div
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}
                            className="flex flex-col items-center gap-6"
                        >
                            <div className="flex items-center gap-12">
                                <div className="flex flex-col items-center">
                                    <div className="w-20 h-20 rounded-full bg-orange-100 dark:bg-orange-900/30 border-4 border-orange-500 flex flex-col items-center justify-center relative shadow-lg">
                                        <span className="text-2xl">🤸</span>
                                        <span className="font-bold text-orange-700 dark:text-orange-300 text-sm">Behavior</span>
                                        <div className="absolute -bottom-6 font-mono text-xs bg-orange-500 text-white px-2 py-0.5 rounded">Strategy $\mu$</div>
                                    </div>
                                    <div className="mt-8 text-xs text-slate-500 text-center max-w-[100px]">
                                        负责探索 & 采样<br />(Daredevil)
                                    </div>
                                </div>

                                <motion.div
                                    className="h-1 w-24 bg-slate-300 relative"
                                    initial={{ width: 0 }} animate={{ width: 96 }}
                                >
                                    <motion.div
                                        className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-purple-500 rounded-full"
                                        animate={{ left: ["0%", "100%"] }}
                                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                                    />
                                    <div className="absolute -top-5 w-full text-center text-xs font-mono text-slate-500">
                                        Exp Replay
                                    </div>
                                </motion.div>

                                <div className="flex flex-col items-center">
                                    <div className="w-24 h-24 rounded-full bg-purple-100 dark:bg-purple-900/30 border-4 border-purple-500 flex flex-col items-center justify-center relative shadow-lg">
                                        <span className="text-3xl">🧠</span>
                                        <span className="font-bold text-purple-700 dark:text-purple-300">Target</span>
                                        <div className="absolute -bottom-8 font-mono text-sm bg-purple-500 text-white px-2 py-0.5 rounded">Strategy $\pi$</div>
                                    </div>
                                    <div className="mt-10 text-xs text-slate-500 text-center max-w-[120px]">
                                        旁观并学习<br />(Policy Iteration)
                                    </div>
                                </div>
                            </div>

                            <div className="bg-purple-50 p-2 rounded text-xs text-purple-600 border border-purple-200 mt-4">
                                "我可以从任何人（过去的我、随机策略、人类）的经验中学习。"
                            </div>
                        </motion.div>
                    </TabsContent>
                </div>
            </Tabs>

            <div className="grid grid-cols-2 gap-4 mt-6 text-sm">
                <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <h4 className="font-bold mb-2 text-slate-700 dark:text-slate-300">On-Policy 特点</h4>
                    <ul className="space-y-1 text-slate-600 dark:text-slate-400 list-disc ml-4">
                        <li>样本效率低 (Sample Inefficient)</li>
                        <li>训练更稳定 (Stable)</li>
                        <li>代表算法: SARSA, PPO, TRPO</li>
                    </ul>
                </div>
                <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <h4 className="font-bold mb-2 text-slate-700 dark:text-slate-300">Off-Policy 特点</h4>
                    <ul className="space-y-1 text-slate-600 dark:text-slate-400 list-disc ml-4">
                        <li>样本效率高 (借助 Replay Buffer)</li>
                        <li>可能不稳定 (Divergence Risk)</li>
                        <li>代表算法: Q-Learning, DQN, SAC</li>
                    </ul>
                </div>
            </div>
        </Card>
    );
}
