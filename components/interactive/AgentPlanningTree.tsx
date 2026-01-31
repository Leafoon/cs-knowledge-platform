"use client";

import { useState } from "react";
import { motion } from "framer-motion";

interface TreeNode {
    id: string;
    label: string;
    children: string[];
    status: "pending" | "in_progress" | "completed" | "failed";
    depth: number;
}

export function AgentPlanningTree() {
    const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(["root"]));

    const goal = "组织一次团队建设活动";

    const tree: Record<string, TreeNode> = {
        "root": {
            id: "root",
            label: goal,
            children: ["task1", "task2", "task3"],
            status: "in_progress",
            depth: 0
        },
        "task1": {
            id: "task1",
            label: "确定活动时间和地点",
            children: ["task1_1", "task1_2"],
            status: "completed",
            depth: 1
        },
        "task1_1": {
            id: "task1_1",
            label: "调查team成员可用时间",
            children: [],
            status: "completed",
            depth: 2
        },
        "task1_2": {
            id: "task1_2",
            label: "预订活动场地",
            children: [],
            status: "completed",
            depth: 2
        },
        "task2": {
            id: "task2",
            label: "策划活动内容",
            children: ["task2_1", "task2_2", "task2_3"],
            status: "in_progress",
            depth: 1
        },
        "task2_1": {
            id: "task2_1",
            label: "团队游戏设计",
            children: [],
            status: "completed",
            depth: 2
        },
        "task2_2": {
            id: "task2_2",
            label: "准备奖品和物资",
            children: [],
            status: "in_progress",
            depth: 2
        },
        "task2_3": {
            id: "task2_3",
            label: "安排餐饮",
            children: [],
            status: "pending",
            depth: 2
        },
        "task3": {
            id: "task3",
            label: "通知和确认参与",
            children: ["task3_1"],
            status: "pending",
            depth: 1
        },
        "task3_1": {
            id: "task3_1",
            label: "发送邀请邮件",
            children: [],
            status: "pending",
            depth: 2
        }
    };

    const toggleNode = (nodeId: string) => {
        const newExpanded = new Set(expandedNodes);
        if (newExpanded.has(nodeId)) {
            newExpanded.delete(nodeId);
        } else {
            newExpanded.add(nodeId);
        }
        setExpandedNodes(newExpanded);
    };

    const getStatusColor = (status: TreeNode["status"]) => {
        switch (status) {
            case "completed": return "green";
            case "in_progress": return "blue";
            case "pending": return "gray";
            case "failed": return "red";
        }
    };

    const getStatusIcon = (status: TreeNode["status"]) => {
        switch (status) {
            case "completed": return "✓";
            case "in_progress": return "⟳";
            case "pending": return "○";
            case "failed": return "✗";
        }
    };

    const getStatusLabel = (status: TreeNode["status"]) => {
        switch (status) {
            case "completed": return "已完成";
            case "in_progress": return "进行中";
            case "pending": return "待处理";
            case "failed": return "失败";
        }
    };

    const renderNode = (nodeId: string) => {
        const node = tree[nodeId];
        const isExpanded = expandedNodes.has(nodeId);
        const hasChildren = node.children.length > 0;

        return (
            <div key={nodeId} className="mb-2">
                <div
                    onClick={() => hasChildren && toggleNode(nodeId)}
                    className={`p-3 rounded-lg border-2 cursor-pointer transition ${isExpanded
                            ? `border-${getStatusColor(node.status)}-500 bg-${getStatusColor(node.status)}-50 dark:bg-${getStatusColor(node.status)}-900/20`
                            : `border-${getStatusColor(node.status)}-300 dark:border-${getStatusColor(node.status)}-700 bg-white dark:bg-slate-800`
                        }`}
                    style={{ marginLeft: `${node.depth * 24}px` }}
                >
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            {hasChildren && (
                                <span className="text-sm">
                                    {isExpanded ? "▼" : "▶"}
                                </span>
                            )}
                            <div className={`w-8 h-8 rounded-full bg-${getStatusColor(node.status)}-600 text-white flex items-center justify-center font-bold`}>
                                {getStatusIcon(node.status)}
                            </div>
                            <div>
                                <div className="font-semibold text-slate-800 dark:text-slate-100">
                                    {node.label}
                                </div>
                                {node.children.length > 0 && (
                                    <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                                        {node.children.length}个子任务
                                    </div>
                                )}
                            </div>
                        </div>
                        <span className={`text-xs px-2 py-1 rounded-full bg-${getStatusColor(node.status)}-100 dark:bg-${getStatusColor(node.status)}-900/30 text-${getStatusColor(node.status)}-700 dark:text-${getStatusColor(node.status)}-400`}>
                            {getStatusLabel(node.status)}
                        </span>
                    </div>
                </div>

                {isExpanded && hasChildren && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-2"
                    >
                        {node.children.map(childId => renderNode(childId))}
                    </motion.div>
                )}
            </div>
        );
    };

    // 计算统计
    const allNodes = Object.values(tree);
    const completedCount = allNodes.filter(n => n.status === "completed").length;
    const inProgressCount = allNodes.filter(n => n.status === "in_progress").length;
    const pendingCount = allNodes.filter(n => n.status === "pending").length;
    const progress = (completedCount / allNodes.length) * 100;

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Agent 规划树
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    层次化任务分解与执行跟踪
                </p>
            </div>

            {/* 目标 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-3 text-slate-800 dark:text-slate-100">最终目标</h4>
                <div className="bg-violet-50 dark:bg-violet-900/20 p-4 rounded-lg border-2 border-violet-300 dark:border-violet-700">
                    <div className="text-xl font-semibold text-slate-800 dark:text-slate-100">{goal}</div>
                </div>
            </div>

            {/* 进度统计 */}
            <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-white dark:bg-slate-800 p-4 rounded-xl shadow-lg">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">总体进度</div>
                    <div className="text-3xl font-bold text-violet-600 dark:text-violet-400 mb-2">
                        {progress.toFixed(0)}%
                    </div>
                    <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                        <div
                            className="h-full bg-violet-600 rounded-full transition-all"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-xl border border-green-300 dark:border-green-700">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">已完成</div>
                    <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                        {completedCount}
                    </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-xl border border-blue-300 dark:border-blue-700">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">进行中</div>
                    <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                        {inProgressCount}
                    </div>
                </div>

                <div className="bg-gray-50 dark:bg-gray-900/20 p-4 rounded-xl border border-gray-300 dark:border-gray-700">
                    <div className="text-sm text-slate-600 dark:text-slate-400 mb-1">待处理</div>
                    <div className="text-3xl font-bold text-gray-600 dark:text-gray-400">
                        {pendingCount}
                    </div>
                </div>
            </div>

            {/* 任务树 */}
            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4 text-slate-800 dark:text-slate-100">任务分解树</h4>

                <div>
                    {renderNode("root")}
                </div>
            </div>

            {/* 规划策略 */}
            <div className="grid grid-cols-3 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-300 dark:border-blue-700">
                    <h5 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">深度优先</h5>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                        先完成一个子任务的所有后续任务，再处理其他子任务
                    </p>
                </div>

                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-300 dark:border-purple-700">
                    <h5 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">广度优先</h5>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                        先完成所有同级子任务，再深入下一层
                    </p>
                </div>

                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-300 dark:border-green-700">
                    <h5 className="font-semibold text-green-700 dark:text-green-400 mb-2">优先级排序</h5>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                        基于重要性和依赖关系动态调整执行顺序
                    </p>
                </div>
            </div>

            <div className="mt-6 bg-violet-100 dark:bg-violet-900/30 p-4 rounded-lg text-center text-sm text-slate-700 dark:text-slate-300">
                💡 <strong>层次化规划</strong>: 将复杂目标分解为可管理的子任务，提升执行效率和成功率
            </div>
        </div>
    );
}
