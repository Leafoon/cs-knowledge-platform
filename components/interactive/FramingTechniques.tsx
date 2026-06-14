"use client";
import { useState } from "react";

export function FramingTechniques() {
  const [method, setMethod] = useState<"byte" | "char" | "bit">("byte");
  const [input, setInput] = useState("1011001111101111110");

  const methods = {
    byte: {
      name: "字节计数法",
      desc: "帧首字节指示帧长度",
      frame: (data: string) => {
        const frames: string[] = [];
        let i = 0;
        while (i < data.length) {
          const len = Math.min(8, data.length - i);
          frames.push(`[${len}]${data.slice(i, i + len)}`);
          i += len;
        }
        return frames;
      },
    },
    char: {
      name: "字符填充法",
      desc: "用FLAG字节定界,ESC转义",
      frame: (data: string) => {
        return [`FLAG|${data.replace(/11111/g, "11111<ESC>")}|FLAG`];
      },
    },
    bit: {
      name: "比特填充法",
      desc: "用01111110定界,每5个1插入0",
      frame: (data: string) => {
        const stuffed = data.replace(/11111/g, "111110");
        return [`01111110|${stuffed}|01111110`];
      },
    },
  };

  const m = methods[method];

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">帧定界技术对比</h3>
      <div className="flex gap-2 mb-4">
        {(["byte", "char", "bit"] as const).map((m) => (
          <button key={m} onClick={() => setMethod(m)}
            className={`px-3 py-1.5 rounded text-sm ${method === m ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>
            {m === "byte" ? "字节计数" : m === "char" ? "字符填充" : "比特填充"}
          </button>
        ))}
      </div>
      <div className="mb-4">
        <label className="text-sm text-text-secondary mb-1 block">输入比特序列:</label>
        <input type="text" value={input} onChange={(e) => setInput(e.target.value.replace(/[^01]/g, "").slice(0, 40))}
          className="w-full px-3 py-2 rounded border border-border-subtle bg-bg-subtle text-text-primary font-mono text-sm" />
      </div>
      <div className="bg-bg-muted rounded-lg p-4 mb-4">
        <div className="font-semibold text-text-primary mb-2">{m.name}</div>
        <p className="text-xs text-text-secondary mb-3">{m.desc}</p>
        <div className="font-mono text-sm text-text-primary space-y-1">
          {m.frame(input).map((f, i) => (
            <div key={i} className="p-2 bg-bg-subtle rounded break-all">{f}</div>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs text-text-secondary">
        <div className="p-2 bg-bg-muted rounded"><strong>字节计数:</strong> 简单但计数字段出错会导致级联错误</div>
        <div className="p-2 bg-bg-muted rounded"><strong>字符填充:</strong> 面向字符,依赖特定字节值</div>
        <div className="p-2 bg-bg-muted rounded"><strong>比特填充:</strong> 面向比特,HDLC使用,更通用</div>
      </div>
      <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded p-3 text-xs text-text-secondary">
        <div className="font-medium text-text-primary mb-1">成帧技术要点</div>
        <div>• 以太网使用长度/类型字段区分帧边界</div>
        <div>• PPP协议使用字节填充(Flag=0x7E, Escape=0x7D)</div>
        <div>• HDLC使用比特填充(01111110作为标志)</div>
        <div>• 现代高速网络倾向于使用固定长度帧</div>
      </div>
    </div>
  );
}

export default FramingTechniques;
