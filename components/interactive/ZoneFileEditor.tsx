"use client";
import { useState } from "react";

interface Record { type: string; name: string; value: string; ttl: number }

const defaultRecords: Record[] = [
  { type: "SOA", name: "@", value: "ns1.example.com. admin.example.com. 2024010101 3600 900 604800 86400", ttl: 86400 },
  { type: "NS", name: "@", value: "ns1.example.com.", ttl: 86400 },
  { type: "A", name: "www", value: "192.168.1.100", ttl: 3600 },
  { type: "A", name: "mail", value: "192.168.1.200", ttl: 3600 },
  { type: "AAAA", name: "www", value: "2001:db8::1", ttl: 3600 },
  { type: "MX", name: "@", value: "10 mail.example.com.", ttl: 3600 },
  { type: "CNAME", name: "blog", value: "www.example.com.", ttl: 3600 },
  { type: "TXT", name: "@", value: "v=spf1 include:_spf.example.com ~all", ttl: 3600 },
];

export function ZoneFileEditor() {
  const [records, setRecords] = useState<Record[]>(defaultRecords);
  const [errors, setErrors] = useState<string[]>([]);
  const [editIdx, setEditIdx] = useState<number | null>(null);
  const [newRec, setNewRec] = useState<Record>({ type: "A", name: "", value: "", ttl: 3600 });

  const validate = () => {
    const errs: string[] = [];
    records.forEach((r, i) => {
      if (!r.name) errs.push(`记录 ${i + 1}: 名称不能为空`);
      if (!r.value) errs.push(`记录 ${i + 1}: 值不能为空`);
      if (r.type === "A" && !/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(r.value)) errs.push(`记录 ${i + 1}: A 记录值必须是 IPv4 地址`);
      if (r.type === "MX" && !/^\d+\s+\S+/.test(r.value)) errs.push(`记录 ${i + 1}: MX 记录格式: 优先级 域名`);
      if (r.type === "SOA" && !/^\S+\s+\S+\s+\d+/.test(r.value)) errs.push(`记录 ${i + 1}: SOA 记录格式不正确`);
    });
    setErrors(errs);
    return errs.length === 0;
  };

  const addRecord = () => {
    if (!newRec.name || !newRec.value) return;
    setRecords([...records, { ...newRec }]);
    setNewRec({ type: "A", name: "", value: "", ttl: 3600 });
  };

  const removeRecord = (idx: number) => {
    setRecords(records.filter((_, i) => i !== idx));
  };

  const toZoneText = () =>
    `$ORIGIN example.com.\n$TTL 86400\n\n` +
    records.map((r) => `${r.name}\t${r.ttl}\tIN\t${r.type}\t${r.value}`).join("\n");

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">DNS 区域文件编辑器</h3>
      <div className="overflow-x-auto mb-4">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left py-2 text-text-muted">类型</th>
              <th className="text-left py-2 text-text-muted">名称</th>
              <th className="text-left py-2 text-text-muted">TTL</th>
              <th className="text-left py-2 text-text-muted">值</th>
              <th className="text-right py-2 text-text-muted">操作</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r, i) => (
              <tr key={i} className={`border-b border-border-subtle ${editIdx === i ? "bg-blue-500/5" : ""}`}>
                <td className="py-2">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                    r.type === "SOA" ? "bg-red-500/10 text-red-400" : r.type === "A" ? "bg-blue-500/10 text-blue-400" :
                    r.type === "AAAA" ? "bg-purple-500/10 text-purple-400" : r.type === "MX" ? "bg-yellow-500/10 text-yellow-400" :
                    "bg-green-500/10 text-green-400"
                  }`}>{r.type}</span>
                </td>
                <td className="py-2 font-mono text-text-primary">{r.name}</td>
                <td className="py-2 text-text-secondary">{r.ttl}</td>
                <td className="py-2 font-mono text-text-primary max-w-xs truncate">{r.value}</td>
                <td className="py-2 text-right">
                  <button onClick={() => removeRecord(i)} className="text-red-400 hover:text-red-300 text-xs">删除</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="p-3 rounded-lg bg-bg-primary border border-border-subtle mb-4">
        <h4 className="text-text-secondary text-xs font-medium mb-2">添加记录</h4>
        <div className="flex gap-2 flex-wrap">
          <select value={newRec.type} onChange={(e) => setNewRec({ ...newRec, type: e.target.value })}
            className="px-2 py-1.5 rounded border border-border-subtle bg-bg-elevated text-text-primary text-xs">
            <option>A</option><option>AAAA</option><option>CNAME</option><option>MX</option><option>NS</option><option>TXT</option><option>SOA</option>
          </select>
          <input value={newRec.name} onChange={(e) => setNewRec({ ...newRec, name: e.target.value })} placeholder="名称"
            className="px-2 py-1.5 rounded border border-border-subtle bg-bg-elevated text-text-primary text-xs w-24" />
          <input type="number" value={newRec.ttl} onChange={(e) => setNewRec({ ...newRec, ttl: Number(e.target.value) })} placeholder="TTL"
            className="px-2 py-1.5 rounded border border-border-subtle bg-bg-elevated text-text-primary text-xs w-20" />
          <input value={newRec.value} onChange={(e) => setNewRec({ ...newRec, value: e.target.value })} placeholder="值"
            className="px-2 py-1.5 rounded border border-border-subtle bg-bg-elevated text-text-primary text-xs flex-1 min-w-[120px]" />
          <button onClick={addRecord} className="px-3 py-1.5 rounded bg-green-500 text-white text-xs hover:bg-green-600">添加</button>
        </div>
      </div>
      <div className="flex gap-2 mb-3">
        <button onClick={validate} className="px-4 py-2 rounded bg-blue-500 text-white text-sm hover:bg-blue-600">验证语法</button>
      </div>
      {errors.length > 0 && (
        <div className="p-3 rounded bg-red-500/10 border border-red-400/30 mb-3">
          {errors.map((e, i) => <p key={i} className="text-red-400 text-xs">❌ {e}</p>)}
        </div>
      )}
      {errors.length === 0 && records.length > 0 && (
        <div className="p-3 rounded bg-green-500/10 border border-green-400/30 mb-3">
          <span className="text-green-400 text-xs">✅ 区域文件语法正确</span>
        </div>
      )}
      <div className="p-3 rounded bg-bg-primary border border-border-subtle">
        <h4 className="text-text-secondary text-xs font-medium mb-1">区域文件预览</h4>
        <pre className="text-text-primary text-xs font-mono whitespace-pre-wrap">{toZoneText()}</pre>
      </div>
    </div>
  );
}
export default ZoneFileEditor;
