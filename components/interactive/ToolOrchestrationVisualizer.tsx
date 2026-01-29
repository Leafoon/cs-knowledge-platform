"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, CheckCircle, XCircle, Clock, Zap, GitBranch } from 'lucide-react';

type ToolStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

type Tool = {
  id: string;
  name: string;
  description: string;
  inputs: string[];
  output: string;
  duration: number;
  status: ToolStatus;
  condition?: string;
  parallel?: boolean;
};

type ExecutionMode = 'sequential' | 'conditional' | 'parallel';

const toolDefinitions: Record<ExecutionMode, Tool[]> = {
  sequential: [
    { id: 't1', name: 'fetch_user_info', description: '获取用户信息', inputs: ['user_id'], output: 'user_info', duration: 800, status: 'pending' },
    { id: 't2', name: 'get_user_orders', description: '查询用户订单', inputs: ['user_id'], output: 'orders', duration: 1200, status: 'pending' },
    { id: 't3', name: 'calculate_total', description: '计算总消费', inputs: ['orders'], output: 'total_spent', duration: 500, status: 'pending' },
    { id: 't4', name: 'send_email', description: '发送个性化邮件', inputs: ['user_info', 'total_spent'], output: 'email_result', duration: 1500, status: 'pending' }
  ],
  conditional: [
    { id: 'c1', name: 'fetch_user_info', description: '获取用户信息', inputs: ['user_id'], output: 'user_info', duration: 800, status: 'pending' },
    { id: 'c2', name: 'get_user_orders', description: '查询用户订单', inputs: ['user_id'], output: 'orders', duration: 1200, status: 'pending' },
    { id: 'c3', name: 'send_email', description: '发送邮件', inputs: ['user_info', 'total_spent'], output: 'email_result', duration: 1500, status: 'pending', condition: 'len(orders) > 0' },
    { id: 'c4', name: 'log_no_orders', description: '记录无订单日志', inputs: ['user_id'], output: 'log_result', duration: 300, status: 'pending', condition: 'len(orders) == 0' }
  ],
  parallel: [
    { id: 'p1', name: 'fetch_user_info', description: '获取用户信息', inputs: ['user_id'], output: 'user_info', duration: 800, status: 'pending', parallel: true },
    { id: 'p2', name: 'get_user_orders', description: '查询用户订单', inputs: ['user_id'], output: 'orders', duration: 1200, status: 'pending', parallel: true },
    { id: 'p3', name: 'get_user_reviews', description: '获取用户评价', inputs: ['user_id'], output: 'reviews', duration: 1000, status: 'pending', parallel: true },
    { id: 'p4', name: 'generate_report', description: '生成报告', inputs: ['user_info', 'orders', 'reviews'], output: 'report', duration: 1500, status: 'pending' }
  ]
};

export default function ToolOrchestrationVisualizer() {
  const [mode, setMode] = useState<ExecutionMode>('sequential');
  const [tools, setTools] = useState<Tool[]>(toolDefinitions.sequential);
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [executionLog, setExecutionLog] = useState<string[]>([]);
  const [context, setContext] = useState<Record<string, any>>({ user_id: '12345' });

  const reset = () => {
    setTools(toolDefinitions[mode].map(t => ({ ...t, status: 'pending' })));
    setCurrentIndex(-1);
    setIsPlaying(false);
    setExecutionLog([]);
    setContext({ user_id: '12345' });
  };

  const handleModeChange = (newMode: ExecutionMode) => {
    setMode(newMode);
    setTools(toolDefinitions[newMode].map(t => ({ ...t, status: 'pending' })));
    setCurrentIndex(-1);
    setIsPlaying(false);
    setExecutionLog([]);
    setContext({ user_id: '12345' });
  };

  const log = (message: string) => {
    setExecutionLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
  };

  const checkCondition = (tool: Tool): boolean => {
    if (!tool.condition) return true;
    
    // 简化条件检查
    if (tool.condition.includes('len(orders)')) {
      const orders = context.orders || [];
      if (tool.condition.includes('> 0')) return orders.length > 0;
      if (tool.condition.includes('== 0')) return orders.length === 0;
    }
    
    return true;
  };

  const executeNextTool = () => {
    if (mode === 'parallel') {
      executeParallel();
    } else {
      executeSequential();
    }
  };

  const executeSequential = () => {
    const nextIndex = currentIndex + 1;
    if (nextIndex >= tools.length) {
      setIsPlaying(false);
      log('✅ 所有工具执行完成');
      return;
    }

    const tool = tools[nextIndex];
    
    // 检查条件
    if (tool.condition && !checkCondition(tool)) {
      log(`⏭️ 跳过 ${tool.name}（条件不满足：${tool.condition}）`);
      const updatedTools = [...tools];
      updatedTools[nextIndex].status = 'skipped';
      setTools(updatedTools);
      setCurrentIndex(nextIndex);
      
      setTimeout(executeNextTool, 300);
      return;
    }

    // 标记为运行中
    log(`▶️ 执行 ${tool.name}`);
    const updatedTools = [...tools];
    updatedTools[nextIndex].status = 'running';
    setTools(updatedTools);
    setCurrentIndex(nextIndex);

    // 模拟执行
    setTimeout(() => {
      // 完成
      updatedTools[nextIndex].status = 'completed';
      setTools([...updatedTools]);
      
      // 更新上下文
      const newContext = { ...context };
      if (tool.name === 'fetch_user_info') {
        newContext[tool.output] = { name: 'Alice', email: 'alice@example.com' };
      } else if (tool.name === 'get_user_orders') {
        newContext[tool.output] = [{ id: 'ORD001', amount: 299 }];
      } else if (tool.name === 'calculate_total') {
        newContext[tool.output] = 299;
      } else {
        newContext[tool.output] = `result_of_${tool.name}`;
      }
      setContext(newContext);
      
      log(`✅ ${tool.name} 完成 → ${tool.output}`);
      
      setTimeout(executeNextTool, 500);
    }, tool.duration);
  };

  const executeParallel = () => {
    // 找到所有待执行的并行工具
    const parallelTools = tools.filter((t, i) => 
      i > currentIndex && t.parallel && tools.slice(0, i).every(pt => !pt.parallel || pt.status === 'completed')
    );

    if (parallelTools.length > 0) {
      // 并行执行
      log(`🔀 并行执行 ${parallelTools.length} 个工具`);
      
      const parallelIndices = parallelTools.map(t => tools.findIndex(tool => tool.id === t.id));
      const updatedTools = [...tools];
      
      parallelIndices.forEach(idx => {
        updatedTools[idx].status = 'running';
      });
      setTools(updatedTools);
      setCurrentIndex(Math.max(...parallelIndices));

      // 等待所有并行工具完成
      const promises = parallelTools.map(tool => 
        new Promise(resolve => {
          setTimeout(() => {
            const idx = tools.findIndex(t => t.id === tool.id);
            updatedTools[idx].status = 'completed';
            
            const newContext = { ...context };
            if (tool.name.includes('fetch_user_info')) {
              newContext[tool.output] = { name: 'Alice' };
            } else if (tool.name.includes('orders')) {
              newContext[tool.output] = [{ amount: 299 }];
            } else if (tool.name.includes('reviews')) {
              newContext[tool.output] = [{ rating: 5 }];
            }
            setContext({ ...context, ...newContext });
            
            log(`✅ ${tool.name} 完成`);
            resolve(null);
          }, tool.duration);
        })
      );

      Promise.all(promises).then(() => {
        setTools([...updatedTools]);
        setTimeout(executeNextTool, 500);
      });
    } else {
      // 执行下一个非并行工具
      executeSequential();
    }
  };

  const togglePlay = () => {
    if (currentIndex === -1 || currentIndex >= tools.length - 1) {
      reset();
      setIsPlaying(true);
      setTimeout(executeNextTool, 100);
    } else {
      setIsPlaying(!isPlaying);
      if (!isPlaying) {
        executeNextTool();
      }
    }
  };

  const getStatusIcon = (status: ToolStatus) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'failed': return <XCircle className="w-5 h-5 text-red-600" />;
      case 'running': return <Clock className="w-5 h-5 text-blue-600 animate-spin" />;
      case 'skipped': return <GitBranch className="w-5 h-5 text-gray-400" />;
      default: return <Clock className="w-5 h-5 text-gray-300" />;
    }
  };

  const getStatusColor = (status: ToolStatus) => {
    switch (status) {
      case 'completed': return 'border-green-500 bg-green-50';
      case 'failed': return 'border-red-500 bg-red-50';
      case 'running': return 'border-blue-500 bg-blue-50';
      case 'skipped': return 'border-gray-400 bg-gray-50';
      default: return 'border-gray-300';
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">工具编排可视化</h3>
        <p className="text-gray-600">演示工具链、条件调用、并发执行</p>
      </div>

      {/* 模式选择 */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">执行模式</label>
        <div className="grid grid-cols-3 gap-3">
          {[
            { id: 'sequential' as ExecutionMode, name: '顺序执行', desc: '工具依次调用，后者依赖前者输出', icon: '→' },
            { id: 'conditional' as ExecutionMode, name: '条件调用', desc: '根据中间结果决定是否执行某工具', icon: '?' },
            { id: 'parallel' as ExecutionMode, name: '并发执行', desc: '独立工具同时执行，提高效率', icon: '⇉' }
          ].map(m => (
            <button
              key={m.id}
              onClick={() => handleModeChange(m.id)}
              className={`p-3 rounded border-2 text-left transition-all ${
                mode === m.id ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-300'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-2xl">{m.icon}</span>
                <span className="font-semibold">{m.name}</span>
              </div>
              <div className="text-xs text-gray-600">{m.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={togglePlay}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {currentIndex === -1 ? '开始执行' : isPlaying ? '暂停' : '继续'}
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
        >
          <RotateCcw className="w-4 h-4" />
          重置
        </button>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* 工具链可视化 */}
        <div>
          <h4 className="font-semibold mb-3">工具执行流程</h4>
          <div className="space-y-3">
            {tools.map((tool, index) => (
              <motion.div
                key={tool.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <div className={`p-3 rounded-lg border-2 ${getStatusColor(tool.status)}`}>
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 mt-0.5">
                      {getStatusIcon(tool.status)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-mono text-sm font-semibold">{tool.name}</span>
                        {tool.parallel && <div title="并行执行"><Zap className="w-4 h-4 text-yellow-600" /></div>}
                      </div>
                      <p className="text-xs text-gray-600 mb-2">{tool.description}</p>
                      
                      <div className="flex items-center gap-3 text-xs">
                        <div>
                          <span className="text-gray-500">输入:</span>
                          <span className="ml-1 font-mono text-blue-600">{tool.inputs.join(', ')}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">输出:</span>
                          <span className="ml-1 font-mono text-green-600">{tool.output}</span>
                        </div>
                      </div>

                      {tool.condition && (
                        <div className="mt-2 text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
                          条件: {tool.condition}
                        </div>
                      )}

                      {tool.status === 'running' && (
                        <div className="mt-2">
                          <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                            <motion.div
                              className="h-full bg-blue-600"
                              initial={{ width: '0%' }}
                              animate={{ width: '100%' }}
                              transition={{ duration: tool.duration / 1000 }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {index < tools.length - 1 && !tool.parallel && (
                  <div className="flex justify-center">
                    <div className="text-2xl text-gray-400">↓</div>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* 右侧：执行日志和上下文 */}
        <div className="space-y-6">
          {/* 执行上下文 */}
          <div>
            <h4 className="font-semibold mb-3">执行上下文</h4>
            <div className="bg-gray-50 p-3 rounded-lg border h-48 overflow-y-auto">
              <pre className="text-xs font-mono">
                {JSON.stringify(context, null, 2)}
              </pre>
            </div>
          </div>

          {/* 执行日志 */}
          <div>
            <h4 className="font-semibold mb-3">执行日志</h4>
            <div className="bg-gray-900 text-green-400 p-3 rounded-lg h-64 overflow-y-auto font-mono text-xs">
              {executionLog.length === 0 ? (
                <div className="text-gray-500">等待执行...</div>
              ) : (
                executionLog.map((log, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    {log}
                  </motion.div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 代码示例 */}
      <div className="mt-6 bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
        <div className="text-xs font-mono">
          <div className="text-green-400"># 定义工具链步骤</div>
          <div>steps = [</div>
          <div className="ml-4">{'{'}</div>
          <div className="ml-8"><span className="text-orange-400">"tool"</span>: <span className="text-yellow-400">"fetch_user_info"</span>,</div>
          <div className="ml-8"><span className="text-orange-400">"input"</span>: {'{'}user_id: <span className="text-yellow-400">"user_id"</span>{'}'},</div>
          <div className="ml-8"><span className="text-orange-400">"output"</span>: <span className="text-yellow-400">"user_info"</span></div>
          <div className="ml-4">{'}'},</div>
          <div className="ml-4 text-gray-500">{'// ...'}</div>
          <div>]</div>
          <div className="mt-2">results = chain.<span className="text-yellow-400">execute_chain</span>(steps, initial_input)</div>
        </div>
      </div>
    </div>
  );
}
