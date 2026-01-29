'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Eye, EyeOff, User, Mail, CreditCard, Phone, MapPin, Lock, FileText, Download, Trash2 } from 'lucide-react'

interface PIIEntity {
  type: string
  value: string
  start: number
  end: number
  icon: React.ReactNode
  color: string
}

const PII_TYPES = {
  PERSON: { icon: <User className="w-4 h-4" />, color: 'blue', label: '姓名' },
  EMAIL: { icon: <Mail className="w-4 h-4" />, color: 'green', label: '邮箱' },
  PHONE: { icon: <Phone className="w-4 h-4" />, color: 'purple', label: '手机号' },
  CREDIT_CARD: { icon: <CreditCard className="w-4 h-4" />, color: 'red', label: '信用卡' },
  SSN: { icon: <Lock className="w-4 h-4" />, color: 'orange', label: 'SSN' },
  ADDRESS: { icon: <MapPin className="w-4 h-4" />, color: 'pink', label: '地址' }
}

const SAMPLE_TEXTS = [
  {
    title: "客服对话",
    text: "您好，我是张三，邮箱是 zhangsan@example.com，手机号 13812345678。我需要更新账单地址为北京市朝阳区建国路1号。",
  },
  {
    title: "订单信息",
    text: "Order for John Doe (john.doe@company.com). Ship to 123 Main Street, New York, NY 10001. Contact: +1-555-0123. Payment: Visa 4532-1234-5678-9010",
  },
  {
    title: "医疗记录",
    text: "Patient: Emily Wilson, DOB: 03/15/1985, SSN: 123-45-6789. Contact: emily.w@gmail.com, (555) 987-6543. Diagnosis: Regular checkup.",
  }
]

export default function PIIDetectionFlow() {
  const [inputText, setInputText] = useState(SAMPLE_TEXTS[0].text)
  const [detectedPII, setDetectedPII] = useState<PIIEntity[]>([])
  const [anonymizationStrategy, setAnonymizationStrategy] = useState<'replace' | 'mask' | 'hash'>('replace')
  const [showOriginal, setShowOriginal] = useState(true)
  const [isProcessing, setIsProcessing] = useState(false)

  const detectPII = () => {
    setIsProcessing(true)
    setDetectedPII([])

    setTimeout(() => {
      const detected: PIIEntity[] = []

      // 检测邮箱
      const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g
      let match
      while ((match = emailRegex.exec(inputText)) !== null) {
        detected.push({
          type: 'EMAIL',
          value: match[0],
          start: match.index,
          end: match.index + match[0].length,
          icon: PII_TYPES.EMAIL.icon,
          color: PII_TYPES.EMAIL.color
        })
      }

      // 检测中国手机号
      const phoneRegex = /\b1[3-9]\d{9}\b/g
      while ((match = phoneRegex.exec(inputText)) !== null) {
        detected.push({
          type: 'PHONE',
          value: match[0],
          start: match.index,
          end: match.index + match[0].length,
          icon: PII_TYPES.PHONE.icon,
          color: PII_TYPES.PHONE.color
        })
      }

      // 检测国际手机号
      const intlPhoneRegex = /\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}/g
      while ((match = intlPhoneRegex.exec(inputText)) !== null) {
        // 避免重复检测
        if (!detected.some(d => d.start === match!.index)) {
          detected.push({
            type: 'PHONE',
            value: match[0],
            start: match.index,
            end: match.index + match[0].length,
            icon: PII_TYPES.PHONE.icon,
            color: PII_TYPES.PHONE.color
          })
        }
      }

      // 检测信用卡号
      const ccRegex = /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g
      while ((match = ccRegex.exec(inputText)) !== null) {
        detected.push({
          type: 'CREDIT_CARD',
          value: match[0],
          start: match.index,
          end: match.index + match[0].length,
          icon: PII_TYPES.CREDIT_CARD.icon,
          color: PII_TYPES.CREDIT_CARD.color
        })
      }

      // 检测 SSN
      const ssnRegex = /\b\d{3}-\d{2}-\d{4}\b/g
      while ((match = ssnRegex.exec(inputText)) !== null) {
        detected.push({
          type: 'SSN',
          value: match[0],
          start: match.index,
          end: match.index + match[0].length,
          icon: PII_TYPES.SSN.icon,
          color: PII_TYPES.SSN.color
        })
      }

      // 检测姓名（简单模式：大写开头的连续单词）
      const nameRegex = /\b[A-Z][a-z]+ [A-Z][a-z]+\b/g
      while ((match = nameRegex.exec(inputText)) !== null) {
        detected.push({
          type: 'PERSON',
          value: match[0],
          start: match.index,
          end: match.index + match[0].length,
          icon: PII_TYPES.PERSON.icon,
          color: PII_TYPES.PERSON.color
        })
      }

      // 中文姓名（简单模式）
      const chineseNameRegex = /[张王李赵刘陈杨黄周吴徐孙朱马胡郭林何高梁郑罗宋谢唐韩冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文][一-龥]{1,3}/g
      while ((match = chineseNameRegex.exec(inputText)) !== null) {
        detected.push({
          type: 'PERSON',
          value: match[0],
          start: match.index,
          end: match.index + match[0].length,
          icon: PII_TYPES.PERSON.icon,
          color: PII_TYPES.PERSON.color
        })
      }

      // 按位置排序（用于渲染高亮）
      detected.sort((a, b) => a.start - b.start)
      setDetectedPII(detected)
      setIsProcessing(false)
    }, 1000)
  }

  const anonymize = (text: string, entity: PIIEntity): string => {
    switch (anonymizationStrategy) {
      case 'replace':
        return `<${PII_TYPES[entity.type as keyof typeof PII_TYPES]?.label || entity.type}>`
      case 'mask':
        if (entity.value.length <= 4) return '****'
        return '*'.repeat(entity.value.length - 4) + entity.value.slice(-4)
      case 'hash':
        // 模拟哈希（实际应使用加密库）
        return `#${Math.random().toString(36).substring(2, 10)}`
      default:
        return entity.value
    }
  }

  const getAnonymizedText = (): string => {
    if (detectedPII.length === 0) return inputText

    let result = ''
    let lastEnd = 0

    detectedPII.forEach(entity => {
      result += inputText.substring(lastEnd, entity.start)
      result += anonymize(inputText, entity)
      lastEnd = entity.end
    })

    result += inputText.substring(lastEnd)
    return result
  }

  const renderHighlightedText = () => {
    if (detectedPII.length === 0) {
      return <span className="text-slate-700 dark:text-slate-300">{inputText}</span>
    }

    const parts = []
    let lastEnd = 0

    detectedPII.forEach((entity, idx) => {
      // 正常文本
      if (entity.start > lastEnd) {
        parts.push(
          <span key={`text-${idx}`} className="text-slate-700 dark:text-slate-300">
            {inputText.substring(lastEnd, entity.start)}
          </span>
        )
      }

      // PII 高亮
      const colorClass = `bg-${entity.color}-100 dark:bg-${entity.color}-900/30 text-${entity.color}-700 dark:text-${entity.color}-300 border-${entity.color}-300 dark:border-${entity.color}-600`
      
      parts.push(
        <span
          key={`pii-${idx}`}
          className={`px-1 py-0.5 rounded border font-mono text-sm ${colorClass}`}
          style={{
            backgroundColor: `var(--${entity.color}-100)`,
            borderColor: `var(--${entity.color}-300)`,
            color: `var(--${entity.color}-700)`
          }}
        >
          {showOriginal ? entity.value : anonymize(inputText, entity)}
        </span>
      )

      lastEnd = entity.end
    })

    // 剩余文本
    if (lastEnd < inputText.length) {
      parts.push(
        <span key="text-end" className="text-slate-700 dark:text-slate-300">
          {inputText.substring(lastEnd)}
        </span>
      )
    }

    return <>{parts}</>
  }

  const getPIIStats = () => {
    const stats = detectedPII.reduce((acc, entity) => {
      acc[entity.type] = (acc[entity.type] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    return Object.entries(stats).map(([type, count]) => ({
      type,
      count,
      ...PII_TYPES[type as keyof typeof PII_TYPES]
    }))
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* 标题 */}
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-purple-500 rounded-lg">
          <Eye className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            PII 检测与脱敏演示
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            实时检测并脱敏个人身份信息（PII），符合 GDPR/CCPA 隐私保护要求
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：输入区 */}
        <div className="lg:col-span-2 space-y-4">
          {/* 样本选择 */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              快速加载示例：
            </label>
            <div className="flex gap-2">
              {SAMPLE_TEXTS.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setInputText(sample.text)
                    setDetectedPII([])
                  }}
                  className="px-3 py-1.5 text-sm bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
                >
                  {sample.title}
                </button>
              ))}
            </div>
          </div>

          {/* 输入框 */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              输入文本（支持中英文）：
            </label>
            <textarea
              value={inputText}
              onChange={(e) => {
                setInputText(e.target.value)
                setDetectedPII([])
              }}
              rows={6}
              className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 resize-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              placeholder="输入包含 PII 的文本进行检测..."
            />
          </div>

          {/* 控制按钮 */}
          <div className="flex gap-3">
            <button
              onClick={detectPII}
              disabled={isProcessing || !inputText}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-purple-500 hover:bg-purple-600 disabled:bg-slate-400 text-white rounded-lg font-medium transition-colors"
            >
              <Eye className="w-4 h-4" />
              {isProcessing ? '检测中...' : '检测 PII'}
            </button>

            <select
              value={anonymizationStrategy}
              onChange={(e) => setAnonymizationStrategy(e.target.value as any)}
              className="px-4 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200"
            >
              <option value="replace">替换为占位符</option>
              <option value="mask">部分掩码</option>
              <option value="hash">哈希</option>
            </select>

            <button
              onClick={() => setShowOriginal(!showOriginal)}
              className="px-4 py-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 rounded-lg transition-colors"
              title={showOriginal ? '显示脱敏' : '显示原文'}
            >
              {showOriginal ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
          </div>

          {/* 结果显示 */}
          <AnimatePresence mode="wait">
            {detectedPII.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-3"
              >
                {/* 高亮显示 */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      检测结果（{showOriginal ? '原文' : '脱敏'}）：
                    </label>
                    <span className="text-xs text-purple-600 dark:text-purple-400 font-medium">
                      发现 {detectedPII.length} 个 PII 实体
                    </span>
                  </div>
                  <div className="p-4 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg leading-relaxed">
                    {renderHighlightedText()}
                  </div>
                </div>

                {/* 脱敏后文本 */}
                <div>
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-2">
                    <Lock className="w-4 h-4" />
                    完全脱敏文本（可安全存储/传输）：
                  </label>
                  <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-lg">
                    <code className="text-sm text-green-800 dark:text-green-200 break-all">
                      {getAnonymizedText()}
                    </code>
                  </div>
                </div>

                {/* 操作按钮 */}
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(getAnonymizedText())
                      alert('脱敏文本已复制到剪贴板')
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm font-medium"
                  >
                    <Download className="w-4 h-4" />
                    复制脱敏文本
                  </button>
                  
                  <button
                    onClick={() => {
                      setInputText('')
                      setDetectedPII([])
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg text-sm font-medium"
                  >
                    <Trash2 className="w-4 h-4" />
                    清除
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* 右侧：统计与图例 */}
        <div className="space-y-4">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-2">
            <FileText className="w-5 h-5" />
            检测统计
          </h4>

          {/* PII 类型统计 */}
          {detectedPII.length > 0 ? (
            <div className="space-y-2">
              {getPIIStats().map((stat, idx) => (
                <motion.div
                  key={stat.type}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="flex items-center justify-between p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
                >
                  <div className="flex items-center gap-2">
                    <div className={`p-1.5 bg-${stat.color}-100 dark:bg-${stat.color}-900/30 rounded`}>
                      {stat.icon}
                    </div>
                    <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      {stat.label}
                    </span>
                  </div>
                  <div className="px-2.5 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full text-xs font-bold">
                    {stat.count}
                  </div>
                </motion.div>
              ))}
            </div>
          ) : (
            <div className="p-6 bg-slate-100 dark:bg-slate-800 rounded-lg text-center text-slate-500 dark:text-slate-400">
              点击"检测 PII"开始分析
            </div>
          )}

          {/* 图例说明 */}
          <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <h5 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
              支持的 PII 类型：
            </h5>
            <div className="space-y-2 text-xs text-slate-600 dark:text-slate-400">
              {Object.entries(PII_TYPES).map(([type, config]) => (
                <div key={type} className="flex items-center gap-2">
                  <div className={`p-1 bg-${config.color}-100 dark:bg-${config.color}-900/30 rounded`}>
                    {config.icon}
                  </div>
                  <span>{config.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* 脱敏策略说明 */}
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-lg">
            <h5 className="text-sm font-semibold text-blue-800 dark:text-blue-200 mb-2">
              脱敏策略：
            </h5>
            <ul className="space-y-1 text-xs text-blue-700 dark:text-blue-300">
              <li><strong>替换：</strong>用类型占位符替换</li>
              <li><strong>掩码：</strong>保留后4位，其余用*</li>
              <li><strong>哈希：</strong>单向加密（不可逆）</li>
            </ul>
          </div>

          {/* 合规提示 */}
          <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-lg">
            <h5 className="text-sm font-semibold text-green-800 dark:text-green-200 mb-2">
              ✅ GDPR/CCPA 合规
            </h5>
            <ul className="space-y-1 text-xs text-green-700 dark:text-green-300">
              <li>✓ 自动检测敏感数据</li>
              <li>✓ 脱敏后可安全存储</li>
              <li>✓ 支持数据最小化</li>
              <li>✓ 审计追踪就绪</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
