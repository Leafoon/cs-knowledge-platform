"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, MapPin, Briefcase, Heart, Plus, Trash2 } from 'lucide-react';

interface Entity {
  id: string;
  name: string;
  type: 'person' | 'place' | 'organization' | 'interest';
  attributes: Record<string, string>;
  connections: string[];
}

const SAMPLE_ENTITIES: Entity[] = [
  {
    id: 'alice',
    name: 'Alice',
    type: 'person',
    attributes: {
      job: 'Software Engineer',
      company: 'Google',
      location: 'New York',
      interest: 'Machine Learning'
    },
    connections: ['google', 'new_york', 'ml']
  },
  {
    id: 'google',
    name: 'Google',
    type: 'organization',
    attributes: {
      industry: 'Technology',
      headquarters: 'Mountain View'
    },
    connections: ['alice']
  },
  {
    id: 'new_york',
    name: 'New York',
    type: 'place',
    attributes: {
      country: 'USA',
      population: '8.3M'
    },
    connections: ['alice']
  },
  {
    id: 'ml',
    name: 'Machine Learning',
    type: 'interest',
    attributes: {
      category: 'Technology',
      related: 'AI, Data Science'
    },
    connections: ['alice']
  }
];

const CONVERSATIONS = [
  { turn: 1, text: "My friend Alice works at Google in New York", entities: ['alice', 'google', 'new_york'] },
  { turn: 2, text: "Alice is a software engineer", entities: ['alice'] },
  { turn: 3, text: "She loves machine learning", entities: ['alice', 'ml'] },
];

export default function EntityMemoryGraph() {
  const [entities, setEntities] = useState<Entity[]>(SAMPLE_ENTITIES);
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);
  const [currentTurn, setCurrentTurn] = useState(0);

  const getEntityIcon = (type: Entity['type']) => {
    switch (type) {
      case 'person':
        return <User className="w-5 h-5" />;
      case 'place':
        return <MapPin className="w-5 h-5" />;
      case 'organization':
        return <Briefcase className="w-5 h-5" />;
      case 'interest':
        return <Heart className="w-5 h-5" />;
    }
  };

  const getEntityColor = (type: Entity['type']) => {
    switch (type) {
      case 'person':
        return 'bg-blue-500 text-white';
      case 'place':
        return 'bg-green-500 text-white';
      case 'organization':
        return 'bg-purple-500 text-white';
      case 'interest':
        return 'bg-orange-500 text-white';
    }
  };

  const getEntityBorderColor = (type: Entity['type']) => {
    switch (type) {
      case 'person':
        return 'border-blue-300';
      case 'place':
        return 'border-green-300';
      case 'organization':
        return 'border-purple-300';
      case 'interest':
        return 'border-orange-300';
    }
  };

  const playConversation = () => {
    setCurrentTurn(0);
    setSelectedEntity(null);
    
    let turn = 0;
    const interval = setInterval(() => {
      if (turn >= CONVERSATIONS.length) {
        clearInterval(interval);
        return;
      }
      setCurrentTurn(turn + 1);
      turn++;
    }, 2000);
  };

  const getVisibleEntities = () => {
    if (currentTurn === 0) return [];
    
    const visibleEntityIds = new Set<string>();
    CONVERSATIONS.slice(0, currentTurn).forEach(conv => {
      conv.entities.forEach(id => visibleEntityIds.add(id));
    });
    
    return entities.filter(e => visibleEntityIds.has(e.id));
  };

  const visibleEntities = getVisibleEntities();

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Entity Memory Graph</h3>
        <p className="text-slate-600">可视化实体记忆的构建过程与关系图谱</p>
      </div>

      {/* Controls */}
      <div className="mb-6 flex gap-3">
        <button
          onClick={playConversation}
          className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
        >
          开始对话
        </button>
        <button
          onClick={() => { setCurrentTurn(0); setSelectedEntity(null); }}
          className="px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 transition-colors"
        >
          重置
        </button>
        <div className="ml-auto flex items-center gap-2 text-sm text-slate-600">
          <span>对话轮次: {currentTurn}/{CONVERSATIONS.length}</span>
        </div>
      </div>

      {/* Conversation Display */}
      {currentTurn > 0 && (
        <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
          <h4 className="font-semibold text-slate-800 mb-3">对话历史</h4>
          <div className="space-y-2">
            {CONVERSATIONS.slice(0, currentTurn).map((conv, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-start gap-3 p-3 bg-blue-50 rounded-lg border border-blue-200"
              >
                <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold flex-shrink-0">
                  {conv.turn}
                </div>
                <div className="flex-1">
                  <p className="text-sm text-slate-800">{conv.text}</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {conv.entities.map(entityId => {
                      const entity = entities.find(e => e.id === entityId);
                      return entity ? (
                        <span
                          key={entityId}
                          className={`px-2 py-0.5 rounded text-xs font-medium ${getEntityColor(entity.type)}`}
                        >
                          {entity.name}
                        </span>
                      ) : null;
                    })}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Entity Graph */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h4 className="font-semibold text-slate-800 mb-4">实体关系图</h4>
          
          {visibleEntities.length === 0 ? (
            <div className="text-center py-12 text-slate-400">
              点击"开始对话"查看实体图谱
            </div>
          ) : (
            <div className="relative min-h-[400px]">
              {/* Connections */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                {visibleEntities.map(entity => {
                  const fromPos = getEntityPosition(entity.id, visibleEntities.length);
                  return entity.connections
                    .filter(connId => visibleEntities.some(e => e.id === connId))
                    .map(connId => {
                      const toPos = getEntityPosition(connId, visibleEntities.length);
                      return (
                        <motion.line
                          key={`${entity.id}-${connId}`}
                          initial={{ pathLength: 0 }}
                          animate={{ pathLength: 1 }}
                          transition={{ duration: 0.5 }}
                          x1={fromPos.x}
                          y1={fromPos.y}
                          x2={toPos.x}
                          y2={toPos.y}
                          stroke="#cbd5e1"
                          strokeWidth="2"
                          strokeDasharray="4 4"
                        />
                      );
                    });
                })}
              </svg>

              {/* Entity Nodes */}
              <AnimatePresence>
                {visibleEntities.map((entity, idx) => {
                  const pos = getEntityPosition(entity.id, visibleEntities.length);
                  return (
                    <motion.button
                      key={entity.id}
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: idx * 0.2 }}
                      onClick={() => setSelectedEntity(entity)}
                      className={`absolute p-4 rounded-lg border-2 ${getEntityBorderColor(entity.type)} ${
                        selectedEntity?.id === entity.id ? 'ring-4 ring-blue-300' : ''
                      } bg-white shadow-lg hover:shadow-xl transition-all cursor-pointer`}
                      style={{
                        left: `${pos.x}px`,
                        top: `${pos.y}px`,
                        transform: 'translate(-50%, -50%)'
                      }}
                    >
                      <div className={`w-10 h-10 rounded-full ${getEntityColor(entity.type)} flex items-center justify-center mb-2`}>
                        {getEntityIcon(entity.type)}
                      </div>
                      <div className="text-sm font-semibold text-slate-800 text-center whitespace-nowrap">
                        {entity.name}
                      </div>
                    </motion.button>
                  );
                })}
              </AnimatePresence>
            </div>
          )}
        </div>

        {/* Entity Details */}
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h4 className="font-semibold text-slate-800 mb-4">实体详情</h4>
          
          {selectedEntity ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="flex items-center gap-3 mb-4 pb-4 border-b border-slate-200">
                <div className={`w-12 h-12 rounded-full ${getEntityColor(selectedEntity.type)} flex items-center justify-center`}>
                  {getEntityIcon(selectedEntity.type)}
                </div>
                <div>
                  <h5 className="font-bold text-lg text-slate-800">{selectedEntity.name}</h5>
                  <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${getEntityColor(selectedEntity.type)}`}>
                    {selectedEntity.type}
                  </span>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <h6 className="text-sm font-semibold text-slate-700 mb-2">属性：</h6>
                  <div className="space-y-2">
                    {Object.entries(selectedEntity.attributes).map(([key, value]) => (
                      <div key={key} className="flex justify-between text-sm">
                        <span className="text-slate-600">{key}:</span>
                        <span className="font-medium text-slate-800">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h6 className="text-sm font-semibold text-slate-700 mb-2">关联实体：</h6>
                  <div className="flex flex-wrap gap-2">
                    {selectedEntity.connections
                      .filter(connId => visibleEntities.some(e => e.id === connId))
                      .map(connId => {
                        const connEntity = entities.find(e => e.id === connId);
                        return connEntity ? (
                          <button
                            key={connId}
                            onClick={() => setSelectedEntity(connEntity)}
                            className={`px-3 py-1 rounded text-sm font-medium ${getEntityColor(connEntity.type)} hover:opacity-80 transition-opacity`}
                          >
                            {connEntity.name}
                          </button>
                        ) : null;
                      })}
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="text-center py-12 text-slate-400">
              点击实体节点查看详情
            </div>
          )}
        </div>
      </div>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">Entity Memory 代码示例</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`from langchain.memory import ConversationEntityMemory

# 创建实体记忆
memory = ConversationEntityMemory(llm=llm)

# 添加对话
memory.save_context(
    {"input": "My friend Alice works at Google in New York"},
    {"output": "That's interesting! Google's NYC office is impressive."}
)

# 查看实体
print(memory.entity_store)
# {
#   'Alice': 'Friend of the user, works at Google',
#   'Google': 'Company where Alice works, has office in New York',
#   'New York': 'Location of Google office'
# }

# 加载特定实体信息
context = memory.load_memory_variables({"input": "Tell me about Alice"})
# 包含 Alice 相关的所有信息`}
        </pre>
      </div>

      {/* Stats */}
      <div className="mt-6 grid grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-blue-600">
            {visibleEntities.filter(e => e.type === 'person').length}
          </div>
          <div className="text-sm text-slate-600 mt-1">人物</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-green-600">
            {visibleEntities.filter(e => e.type === 'place').length}
          </div>
          <div className="text-sm text-slate-600 mt-1">地点</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-purple-600">
            {visibleEntities.filter(e => e.type === 'organization').length}
          </div>
          <div className="text-sm text-slate-600 mt-1">组织</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-orange-600">
            {visibleEntities.filter(e => e.type === 'interest').length}
          </div>
          <div className="text-sm text-slate-600 mt-1">兴趣</div>
        </div>
      </div>
    </div>
  );
}

// Helper function to calculate entity positions
function getEntityPosition(entityId: string, totalEntities: number): { x: number; y: number } {
  const positions: Record<string, { x: number; y: number }> = {
    alice: { x: 200, y: 200 },
    google: { x: 400, y: 150 },
    new_york: { x: 400, y: 250 },
    ml: { x: 200, y: 350 }
  };
  return positions[entityId] || { x: 300, y: 200 };
}
