"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Eye, EyeOff, Lock, Trash2, Download, CheckCircle } from 'lucide-react';

type WorkflowStep = 'pii-detection' | 'redaction' | 'encryption' | 'gdpr-deletion' | 'complete';

interface PIIMatch {
  type: string;
  value: string;
  redacted: string;
}

const sampleMessage = "你好，我叫张三，我的邮箱是 zhangsan@example.com，电话是 138-1234-5678，身份证号是 110101199001011234。";

const piiPatterns = {
  email: /[\w\.-]+@[\w\.-]+\.\w+/g,
  phone: /1[3-9]\d{1}-\d{4}-\d{4}/g,
  id_card: /\d{17}[\dXx]/g
};

export default function PrivacyComplianceFlow() {
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('pii-detection');
  const [detectedPII, setDetectedPII] = useState<PIIMatch[]>([]);
  const [redactedMessage, setRedactedMessage] = useState<string>(sampleMessage);
  const [encryptedMessage, setEncryptedMessage] = useState<string>('');
  const [redactionMode, setRedactionMode] = useState<'partial' | 'full'>('partial');
  const [isPlaying, setIsPlaying] = useState(false);

  const detectPII = () => {
    const matches: PIIMatch[] = [];
    
    // Email
    const emailMatches = sampleMessage.match(piiPatterns.email);
    if (emailMatches) {
      emailMatches.forEach(email => {
        const [local, domain] = email.split('@');
        matches.push({
          type: 'Email',
          value: email,
          redacted: redactionMode === 'partial' 
            ? `${local.slice(0, 2)}***@${domain}`
            : '***@***.***'
        });
      });
    }

    // Phone
    const phoneMatches = sampleMessage.match(piiPatterns.phone);
    if (phoneMatches) {
      phoneMatches.forEach(phone => {
        matches.push({
          type: 'Phone',
          value: phone,
          redacted: redactionMode === 'partial'
            ? phone.replace(/(\d{3})-(\d{4})-(\d{4})/, '$1-****-$3')
            : '***-****-****'
        });
      });
    }

    // ID Card
    const idMatches = sampleMessage.match(piiPatterns.id_card);
    if (idMatches) {
      idMatches.forEach(id => {
        matches.push({
          type: 'ID Card',
          value: id,
          redacted: redactionMode === 'partial'
            ? id.slice(0, 6) + '********' + id.slice(-4)
            : '******************'
        });
      });
    }

    setDetectedPII(matches);
    setCurrentStep('redaction');
  };

  const applyRedaction = () => {
    let redacted = sampleMessage;
    detectedPII.forEach(pii => {
      redacted = redacted.replace(pii.value, pii.redacted);
    });
    setRedactedMessage(redacted);
    setCurrentStep('encryption');
  };

  const applyEncryption = () => {
    // Simulate Fernet encryption
    const mockEncrypted = btoa(redactedMessage).replace(/=/g, '').slice(0, 64);
    setEncryptedMessage(`gAAAAAB${mockEncrypted}...`);
    setCurrentStep('gdpr-deletion');
  };

  const performGDPRDeletion = () => {
    setCurrentStep('complete');
  };

  const playWorkflow = async () => {
    setIsPlaying(true);
    setCurrentStep('pii-detection');
    
    await new Promise(resolve => setTimeout(resolve, 1500));
    detectPII();
    
    await new Promise(resolve => setTimeout(resolve, 1500));
    applyRedaction();
    
    await new Promise(resolve => setTimeout(resolve, 1500));
    applyEncryption();
    
    await new Promise(resolve => setTimeout(resolve, 1500));
    performGDPRDeletion();
    
    setIsPlaying(false);
  };

  const steps: { id: WorkflowStep; label: string; icon: React.ElementType }[] = [
    { id: 'pii-detection', label: 'PII 检测', icon: Eye },
    { id: 'redaction', label: '脱敏处理', icon: EyeOff },
    { id: 'encryption', label: '加密存储', icon: Lock },
    { id: 'gdpr-deletion', label: 'GDPR 删除', icon: Trash2 },
    { id: 'complete', label: '完成', icon: CheckCircle }
  ];

  const getStepIndex = (step: WorkflowStep) => steps.findIndex(s => s.id === step);

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-green-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">
          Privacy & Compliance Workflow
        </h3>
        <p className="text-slate-600">
          隐私保护与 GDPR 合规流程演示
        </p>
      </div>

      {/* Workflow Timeline */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 mb-6">
        <div className="flex justify-between items-center mb-8">
          {steps.map((step, idx) => (
            <div key={step.id} className="flex items-center">
              <div className="flex flex-col items-center">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
                  getStepIndex(currentStep) >= idx
                    ? 'bg-green-500 text-white'
                    : 'bg-slate-200 text-slate-400'
                }`}>
                  <step.icon className="w-6 h-6" />
                </div>
                <span className="text-xs text-slate-600 mt-2 text-center">{step.label}</span>
              </div>
              
              {idx < steps.length - 1 && (
                <div className={`w-16 h-1 mx-2 transition-colors ${
                  getStepIndex(currentStep) > idx ? 'bg-green-500' : 'bg-slate-200'
                }`} />
              )}
            </div>
          ))}
        </div>

        <div className="flex gap-3">
          <button
            onClick={playWorkflow}
            disabled={isPlaying}
            className="px-6 py-2 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isPlaying ? '执行中...' : '开始演示'}
          </button>

          <button
            onClick={() => setCurrentStep('pii-detection')}
            className="px-6 py-2 bg-slate-200 text-slate-700 rounded-lg font-medium hover:bg-slate-300"
          >
            重置
          </button>
        </div>
      </div>

      {/* Step Details */}
      <AnimatePresence mode="wait">
        {currentStep === 'pii-detection' && (
          <motion.div
            key="detection"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-white rounded-lg border border-slate-200 p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <Eye className="w-5 h-5 text-blue-500" />
              <h4 className="font-semibold text-slate-800">步骤 1: PII 检测</h4>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-slate-700 mb-2 block">原始消息</label>
                <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                  <p className="text-slate-800">{sampleMessage}</p>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-slate-700 mb-2 block">脱敏模式</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => setRedactionMode('partial')}
                    className={`px-4 py-2 rounded-lg font-medium ${
                      redactionMode === 'partial'
                        ? 'bg-blue-500 text-white'
                        : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    部分脱敏
                  </button>
                  <button
                    onClick={() => setRedactionMode('full')}
                    className={`px-4 py-2 rounded-lg font-medium ${
                      redactionMode === 'full'
                        ? 'bg-blue-500 text-white'
                        : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    完全脱敏
                  </button>
                </div>
              </div>

              <button
                onClick={detectPII}
                className="w-full px-6 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600"
              >
                开始检测 PII
              </button>
            </div>
          </motion.div>
        )}

        {currentStep === 'redaction' && (
          <motion.div
            key="redaction"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-white rounded-lg border border-slate-200 p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <EyeOff className="w-5 h-5 text-purple-500" />
              <h4 className="font-semibold text-slate-800">步骤 2: 脱敏处理</h4>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-slate-700 mb-2 block">检测到的 PII</label>
                <div className="space-y-2">
                  {detectedPII.map((pii, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-red-50 border border-red-200 rounded-lg">
                      <div>
                        <span className="text-sm font-medium text-red-700">{pii.type}</span>
                        <p className="text-sm text-slate-600 mt-1">
                          原始: <code className="text-red-600">{pii.value}</code>
                        </p>
                      </div>
                      <div>
                        <span className="text-xs text-slate-500">脱敏后</span>
                        <p className="text-sm text-green-600 font-mono">{pii.redacted}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <button
                onClick={applyRedaction}
                className="w-full px-6 py-3 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600"
              >
                应用脱敏
              </button>
            </div>
          </motion.div>
        )}

        {currentStep === 'encryption' && (
          <motion.div
            key="encryption"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-white rounded-lg border border-slate-200 p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <Lock className="w-5 h-5 text-orange-500" />
              <h4 className="font-semibold text-slate-800">步骤 3: 加密存储</h4>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-slate-700 mb-2 block">脱敏后的消息</label>
                <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                  <p className="text-slate-800">{redactedMessage}</p>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-slate-700 mb-2 block">加密算法</label>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <code className="text-sm text-slate-700">Fernet (AES-128-CBC + HMAC-SHA256)</code>
                </div>
              </div>

              <button
                onClick={applyEncryption}
                className="w-full px-6 py-3 bg-orange-500 text-white rounded-lg font-medium hover:bg-orange-600"
              >
                加密并存储
              </button>
            </div>
          </motion.div>
        )}

        {currentStep === 'gdpr-deletion' && (
          <motion.div
            key="gdpr"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-white rounded-lg border border-slate-200 p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <Trash2 className="w-5 h-5 text-red-500" />
              <h4 className="font-semibold text-slate-800">步骤 4: GDPR 删除请求</h4>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-slate-700 mb-2 block">加密数据</label>
                <div className="p-4 bg-slate-900 text-green-400 rounded-lg font-mono text-xs break-all">
                  {encryptedMessage || '等待加密...'}
                </div>
              </div>

              <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-sm text-yellow-800">
                  <strong>GDPR 合规操作：</strong>根据 GDPR 第17条（被遗忘权），用户有权要求删除其个人数据。
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={performGDPRDeletion}
                  className="px-6 py-3 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 flex items-center justify-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  删除用户数据
                </button>
                <button className="px-6 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 flex items-center justify-center gap-2">
                  <Download className="w-4 h-4" />
                  导出用户数据
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {currentStep === 'complete' && (
          <motion.div
            key="complete"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-lg border border-slate-200 p-6"
          >
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="w-8 h-8 text-green-500" />
              </div>
              <h4 className="text-xl font-bold text-slate-800 mb-2">合规流程完成！</h4>
              <p className="text-slate-600 mb-4">
                已成功完成 PII 检测、脱敏、加密和 GDPR 删除流程
              </p>
              <div className="grid grid-cols-3 gap-4 mt-6">
                <div className="p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{detectedPII.length}</div>
                  <div className="text-xs text-slate-600">PII 检测</div>
                </div>
                <div className="p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">100%</div>
                  <div className="text-xs text-slate-600">脱敏率</div>
                </div>
                <div className="p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">AES-128</div>
                  <div className="text-xs text-slate-600">加密强度</div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Code Example */}
      <div className="mt-6 p-4 bg-slate-900 text-slate-100 rounded-lg">
        <h4 className="font-semibold mb-3">隐私保护代码</h4>
        <pre className="text-xs font-mono overflow-x-auto">
{`from cryptography.fernet import Fernet
import re

class PrivacyProtectedMemory:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.pii_patterns = {
            'email': r'[\\w\\.-]+@[\\w\\.-]+\\.\\w+',
            'phone': r'1[3-9]\\d{1}-\\d{4}-\\d{4}',
            'id_card': r'\\d{17}[\\dXx]'
        }
    
    def detect_pii(self, text: str) -> dict:
        """检测 PII"""
        matches = {}
        for pii_type, pattern in self.pii_patterns.items():
            found = re.findall(pattern, text)
            if found:
                matches[pii_type] = found
        return matches
    
    def redact_pii(self, text: str, mode='partial') -> str:
        """脱敏 PII"""
        # Email: partial - z***@example.com
        text = re.sub(
            r'([\\w])([\\w\\.-]+)(@[\\w\\.-]+)',
            lambda m: f"{m.group(1)}***{m.group(3)}" if mode == 'partial' 
                     else '***@***.***',
            text
        )
        # Phone: partial - 138-****-5678
        text = re.sub(
            r'(\\d{3})-(\\d{4})-(\\d{4})',
            lambda m: f"{m.group(1)}-****-{m.group(3)}" if mode == 'partial'
                     else '***-****-****',
            text
        )
        return text
    
    def encrypt(self, text: str) -> str:
        """加密消息"""
        return self.cipher.encrypt(text.encode()).decode()
    
    def delete_user_data(self, user_id: str):
        """GDPR 删除"""
        # Redis
        self.redis.delete(f"chat_history:{user_id}")
        # PostgreSQL
        self.pg_cursor.execute(
            "DELETE FROM messages WHERE user_id = %s",
            (user_id,)
        )
        print(f"✓ 已删除用户 {user_id} 的所有数据")

# 使用
memory = PrivacyProtectedMemory()
pii = memory.detect_pii(message)
redacted = memory.redact_pii(message, mode='partial')
encrypted = memory.encrypt(redacted)`}
        </pre>
      </div>
    </div>
  );
}
