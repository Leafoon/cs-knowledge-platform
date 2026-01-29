"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bell, Plus, Trash2, Mail, MessageSquare, Webhook, Check } from 'lucide-react';

interface AlertRule {
  id: string;
  name: string;
  metric: string;
  condition: string;
  threshold: number;
  duration: number;
  notifications: string[];
  enabled: boolean;
}

const AlertRuleBuilder: React.FC = () => {
  const [rules, setRules] = useState<AlertRule[]>([
    {
      id: '1',
      name: '高错误率告警',
      metric: 'error_rate',
      condition: 'greater_than',
      threshold: 5,
      duration: 5,
      notifications: ['email', 'slack'],
      enabled: true,
    },
    {
      id: '2',
      name: 'P99 延迟告警',
      metric: 'p99_latency',
      condition: 'greater_than',
      threshold: 5000,
      duration: 3,
      notifications: ['slack'],
      enabled: true,
    },
  ]);

  const [showForm, setShowForm] = useState(false);
  const [editingRule, setEditingRule] = useState<AlertRule | null>(null);

  const metricOptions = [
    { value: 'error_rate', label: '错误率 (%)', unit: '%' },
    { value: 'success_rate', label: '成功率 (%)', unit: '%' },
    { value: 'avg_latency', label: '平均延迟', unit: 'ms' },
    { value: 'p95_latency', label: 'P95 延迟', unit: 'ms' },
    { value: 'p99_latency', label: 'P99 延迟', unit: 'ms' },
    { value: 'request_rate', label: '请求速率', unit: 'req/min' },
    { value: 'token_cost', label: 'Token 成本', unit: '$/hour' },
  ];

  const conditionOptions = [
    { value: 'greater_than', label: '大于 >' },
    { value: 'less_than', label: '小于 <' },
    { value: 'equals', label: '等于 =' },
  ];

  const notificationOptions = [
    { value: 'email', label: 'Email', icon: Mail },
    { value: 'slack', label: 'Slack', icon: MessageSquare },
    { value: 'webhook', label: 'Webhook', icon: Webhook },
  ];

  const [formData, setFormData] = useState<Partial<AlertRule>>({
    name: '',
    metric: 'error_rate',
    condition: 'greater_than',
    threshold: 5,
    duration: 5,
    notifications: [],
    enabled: true,
  });

  const handleCreateOrUpdate = () => {
    if (editingRule) {
      // 更新现有规则
      setRules(rules.map(r => r.id === editingRule.id ? { ...formData as AlertRule, id: editingRule.id } : r));
      setEditingRule(null);
    } else {
      // 创建新规则
      const newRule: AlertRule = {
        ...formData as AlertRule,
        id: Date.now().toString(),
      };
      setRules([...rules, newRule]);
    }
    setShowForm(false);
    setFormData({
      name: '',
      metric: 'error_rate',
      condition: 'greater_than',
      threshold: 5,
      duration: 5,
      notifications: [],
      enabled: true,
    });
  };

  const handleEdit = (rule: AlertRule) => {
    setEditingRule(rule);
    setFormData(rule);
    setShowForm(true);
  };

  const handleDelete = (id: string) => {
    setRules(rules.filter(r => r.id !== id));
  };

  const toggleEnabled = (id: string) => {
    setRules(rules.map(r => r.id === id ? { ...r, enabled: !r.enabled } : r));
  };

  const getMetricLabel = (value: string) => {
    return metricOptions.find(m => m.value === value)?.label || value;
  };

  const getConditionLabel = (value: string) => {
    return conditionOptions.find(c => c.value === value)?.label || value;
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-orange-50 to-red-50 rounded-xl shadow-lg">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <Bell className="w-6 h-6 text-orange-600" />
              告警规则配置器
            </h3>
            <p className="text-gray-600">可视化配置监控告警，及时发现生产问题</p>
          </div>
          <button
            onClick={() => {
              setShowForm(!showForm);
              setEditingRule(null);
              setFormData({
                name: '',
                metric: 'error_rate',
                condition: 'greater_than',
                threshold: 5,
                duration: 5,
                notifications: [],
                enabled: true,
              });
            }}
            className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors font-medium"
          >
            <Plus className="w-4 h-4" />
            新建规则
          </button>
        </div>
      </div>

      {/* 规则列表 */}
      <div className="space-y-3 mb-6">
        <AnimatePresence>
          {rules.map((rule) => (
            <motion.div
              key={rule.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className={`p-4 rounded-lg border-2 ${
                rule.enabled ? 'border-orange-200 bg-white' : 'border-gray-200 bg-gray-50'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-grow">
                  <div className="flex items-center gap-3 mb-2">
                    <h4 className="text-lg font-semibold text-gray-800">{rule.name}</h4>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      rule.enabled ? 'bg-green-100 text-green-700' : 'bg-gray-200 text-gray-600'
                    }`}>
                      {rule.enabled ? '已启用' : '已禁用'}
                    </span>
                  </div>
                  <div className="text-sm text-gray-700 mb-2">
                    当 <strong>{getMetricLabel(rule.metric)}</strong>{' '}
                    <strong>{getConditionLabel(rule.condition)}</strong>{' '}
                    <strong>{rule.threshold}{metricOptions.find(m => m.value === rule.metric)?.unit}</strong>{' '}
                    持续 <strong>{rule.duration} 分钟</strong> 时触发
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500">通知渠道:</span>
                    {rule.notifications.map((n) => {
                      const NotifIcon = notificationOptions.find(opt => opt.value === n)?.icon || Bell;
                      return (
                        <div key={n} className="flex items-center gap-1 px-2 py-1 bg-blue-50 rounded text-xs text-blue-700">
                          <NotifIcon className="w-3 h-3" />
                          {notificationOptions.find(opt => opt.value === n)?.label}
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={() => toggleEnabled(rule.id)}
                    className={`p-2 rounded transition-colors ${
                      rule.enabled ? 'bg-green-100 text-green-600 hover:bg-green-200' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                    title={rule.enabled ? '禁用' : '启用'}
                  >
                    <Check className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleEdit(rule)}
                    className="p-2 bg-blue-100 text-blue-600 rounded hover:bg-blue-200 transition-colors"
                  >
                    编辑
                  </button>
                  <button
                    onClick={() => handleDelete(rule.id)}
                    className="p-2 bg-red-100 text-red-600 rounded hover:bg-red-200 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* 创建/编辑表单 */}
      <AnimatePresence>
        {showForm && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="p-6 bg-white rounded-lg shadow border-2 border-orange-200 mb-6"
          >
            <h4 className="text-lg font-semibold text-gray-800 mb-4">
              {editingRule ? '编辑规则' : '新建告警规则'}
            </h4>
            
            <div className="space-y-4">
              {/* 规则名称 */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">规则名称</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="例如：高错误率告警"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                />
              </div>

              {/* 监控指标 */}
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">监控指标</label>
                  <select
                    value={formData.metric}
                    onChange={(e) => setFormData({ ...formData, metric: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    {metricOptions.map((opt) => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">条件</label>
                  <select
                    value={formData.condition}
                    onChange={(e) => setFormData({ ...formData, condition: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    {conditionOptions.map((opt) => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">阈值</label>
                  <input
                    type="number"
                    value={formData.threshold}
                    onChange={(e) => setFormData({ ...formData, threshold: parseFloat(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
              </div>

              {/* 持续时间 */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  持续时间（分钟）- 条件满足多久后触发
                </label>
                <input
                  type="number"
                  value={formData.duration}
                  onChange={(e) => setFormData({ ...formData, duration: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                />
              </div>

              {/* 通知渠道 */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">通知渠道</label>
                <div className="flex gap-3">
                  {notificationOptions.map((opt) => {
                    const Icon = opt.icon;
                    const isSelected = formData.notifications?.includes(opt.value);
                    return (
                      <button
                        key={opt.value}
                        onClick={() => {
                          const notifications = formData.notifications || [];
                          setFormData({
                            ...formData,
                            notifications: isSelected
                              ? notifications.filter(n => n !== opt.value)
                              : [...notifications, opt.value],
                          });
                        }}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg border-2 transition-colors ${
                          isSelected
                            ? 'border-orange-500 bg-orange-50 text-orange-700'
                            : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                        {opt.label}
                        {isSelected && <Check className="w-4 h-4" />}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* 按钮 */}
              <div className="flex gap-3 pt-2">
                <button
                  onClick={handleCreateOrUpdate}
                  disabled={!formData.name || !formData.notifications || formData.notifications.length === 0}
                  className="px-6 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                >
                  {editingRule ? '更新规则' : '创建规则'}
                </button>
                <button
                  onClick={() => {
                    setShowForm(false);
                    setEditingRule(null);
                  }}
                  className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
                >
                  取消
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 说明 */}
      <div className="p-4 bg-white rounded-lg border border-orange-200">
        <h4 className="font-semibold text-gray-800 mb-2">💡 告警配置最佳实践</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• <strong>避免告警疲劳</strong>：设置合理的阈值和持续时间，避免误报</li>
          <li>• <strong>分级告警</strong>：区分警告（Warning）和严重（Critical）两个级别</li>
          <li>• <strong>多渠道通知</strong>：关键告警使用多个渠道（Email + Slack）</li>
          <li>• <strong>定期审查</strong>：每月检查告警触发情况，调整规则</li>
          <li>• <strong>持续时间</strong>：建议设置 3-5 分钟，避免短暂波动触发告警</li>
        </ul>
      </div>
    </div>
  );
};

export default AlertRuleBuilder;
