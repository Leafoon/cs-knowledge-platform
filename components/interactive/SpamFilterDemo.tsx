"use client";
import { useState } from "react";

const spamWords: Record<string, number> = { 免费: 0.89, 中奖: 0.92, 优惠: 0.65, 贷款: 0.85, 发票: 0.78, 赢取: 0.88, 点击: 0.72, 立即: 0.68, 限时: 0.8, 红包: 0.75 };
const hamWords: Record<string, number> = { 会议: 0.85, 项目: 0.82, 报告: 0.78, 请查收: 0.88, 附件: 0.7, 讨论: 0.75, 安排: 0.72, 进度: 0.68 };

function tokenize(text: string): string[] {
  const all = { ...spamWords, ...hamWords };
  return Object.keys(all).filter((w) => text.includes(w));
}

function classify(text: string) {
  const tokens = tokenize(text);
  const pSpam = 0.4;
  const pHam = 0.6;
  let logSpam = Math.log(pSpam);
  let logHam = Math.log(pHam);
  const details: { word: string; pSw: number; pHw: number }[] = [];
  tokens.forEach((w) => {
    const pSw = spamWords[w] ?? 0.1;
    const pHw = hamWords[w] ?? 0.1;
    logSpam += Math.log(pSw);
    logHam += Math.log(pHw);
    details.push({ word: w, pSw, pHw });
  });
  const total = Math.exp(logSpam) + Math.exp(logHam);
  const prob = Math.exp(logSpam) / total;
  return { tokens, details, prob, isSpam: prob > 0.5 };
}

export function SpamFilterDemo() {
  const [text, setText] = useState("恭喜您中奖！点击立即领取免费红包，限时优惠！");
  const result = classify(text);
  const [custom, setCustom] = useState(false);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">垃圾邮件过滤器 (Bayesian Classifier)</h3>
      <div className="mb-4">
        <label className="text-text-secondary text-sm mb-1 block">输入邮件内容：</label>
        <textarea
          value={text}
          onChange={(e) => { setText(e.target.value); setCustom(true); }}
          className="w-full p-3 rounded border border-border-subtle bg-bg-primary text-text-primary text-sm resize-none h-20"
          placeholder="输入中文邮件内容..."
        />
      </div>
      <div className="flex gap-2 mb-4 flex-wrap">
        {["恭喜您中奖！点击立即领取免费红包，限时优惠！", "请查收项目报告附件，安排会议讨论进度", "免费贷款发票，立即赢取优惠"].map((s) => (
          <button key={s} onClick={() => { setText(s); setCustom(false); }}
            className={`text-xs px-3 py-1.5 rounded border transition-colors ${!custom && text === s ? "border-blue-400 bg-blue-500/10 text-blue-400" : "border-border-subtle text-text-muted hover:text-text-primary"}`}>
            {s.slice(0, 15)}...
          </button>
        ))}
      </div>
      <div className={`p-4 rounded-lg border-2 mb-4 ${result.isSpam ? "border-red-400 bg-red-500/10" : "border-green-400 bg-green-500/10"}`}>
        <div className="flex items-center justify-between mb-2">
          <span className={`font-semibold text-lg ${result.isSpam ? "text-red-400" : "text-green-400"}`}>
            {result.isSpam ? "🚫 垃圾邮件" : "✅ 正常邮件"}
          </span>
          <span className="text-text-secondary text-sm">P(spam) = {(result.prob * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2">
          <div className={`h-3 rounded-full transition-all ${result.isSpam ? "bg-red-500" : "bg-green-500"}`} style={{ width: `${result.prob * 100}%` }} />
        </div>
      </div>
      {result.details.length > 0 && (
        <div>
          <h4 className="text-text-primary text-sm font-medium mb-2">词概率分析 (P(spam|word))：</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {result.details.map((d) => (
              <div key={d.word} className="flex items-center justify-between p-2 rounded bg-bg-primary border border-border-subtle">
                <span className="text-text-primary text-sm font-mono">&quot;{d.word}&quot;</span>
                <div className="flex gap-3 text-xs">
                  <span className="text-red-400">spam: {(d.pSw * 100).toFixed(0)}%</span>
                  <span className="text-green-400">ham: {(d.pHw * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {result.details.length === 0 && <p className="text-text-muted text-sm">未识别到已知关键词，请尝试包含更多特征词。</p>}
    </div>
  );
}
export default SpamFilterDemo;
