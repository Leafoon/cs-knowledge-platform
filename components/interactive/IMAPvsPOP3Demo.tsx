"use client";
import { useState } from "react";

export function IMAPvsPOP3Demo() {
  const [protocol, setProtocol] = useState<"imap" | "pop3">("imap");
  const [emails] = useState([
    { id: 1, from: "alice@example.com", subject: "会议通知", size: "15KB" },
    { id: 2, from: "bob@example.com", subject: "项目报告", size: "250KB" },
    { id: 3, from: "carol@example.com", subject: "周末计划", size: "8KB" },
  ]);
  const [localEmails, setLocalEmails] = useState<typeof emails>([]);
  const [serverEmails, setServerEmails] = useState(emails);
  const [log, setLog] = useState<string[]>([]);

  const download = (id: number) => {
    const email = serverEmails.find((e) => e.id === id);
    if (!email) return;
    if (protocol === "pop3") {
      setServerEmails(serverEmails.filter((e) => e.id !== id));
      setLocalEmails([...localEmails, email]);
      setLog([...log, `[POP3] RETR ${id} → 从服务器获取并删除`]);
    } else {
      if (!localEmails.find((e) => e.id === id)) {
        setLocalEmails([...localEmails, email]);
      }
      setLog([...log, `[IMAP] FETCH ${id} → 从服务器获取副本,服务器保留原邮件`]);
    }
  };

  const deleteEmail = (id: number) => {
    if (protocol === "pop3") {
      setLog([...log, `[POP3] 邮件 #${id} 已在下载时从服务器删除`]);
    } else {
      setServerEmails(serverEmails.filter((e) => e.id !== id));
      setLog([...log, `[IMAP] STORE ${id} +DELETED → EXPUNGE → 服务器删除`]);
    }
  };

  const reset = () => {
    setServerEmails(emails);
    setLocalEmails([]);
    setLog([]);
  };

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">IMAP vs POP3 邮件协议对比</h3>
      <div className="flex gap-3 mb-4">
        <button onClick={() => { setProtocol("imap"); reset(); }}
          className={`px-4 py-2 rounded text-sm ${protocol === "imap" ? "bg-blue-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>IMAP</button>
        <button onClick={() => { setProtocol("pop3"); reset(); }}
          className={`px-4 py-2 rounded text-sm ${protocol === "pop3" ? "bg-green-500 text-white" : "bg-bg-subtle text-text-secondary"}`}>POP3</button>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-bg-muted rounded-lg p-3">
          <div className="text-sm font-semibold text-text-primary mb-2">服务器邮箱</div>
          {serverEmails.map((e) => (
            <div key={e.id} className="flex items-center justify-between p-2 bg-bg-subtle rounded mb-1 text-xs">
              <div>
                <span className="font-mono text-text-primary">{e.subject}</span>
                <span className="text-text-secondary ml-2">({e.size})</span>
              </div>
              <div className="flex gap-1">
                <button onClick={() => download(e.id)} className="px-2 py-0.5 bg-blue-500 text-white rounded">下载</button>
                <button onClick={() => deleteEmail(e.id)} className="px-2 py-0.5 bg-red-500 text-white rounded">删除</button>
              </div>
            </div>
          ))}
          {serverEmails.length === 0 && <div className="text-xs text-text-secondary">空</div>}
        </div>
        <div className="bg-bg-muted rounded-lg p-3">
          <div className="text-sm font-semibold text-text-primary mb-2">本地邮箱</div>
          {localEmails.map((e) => (
            <div key={e.id} className="p-2 bg-bg-subtle rounded mb-1 text-xs">
              <span className="font-mono text-text-primary">{e.subject}</span>
              <span className="text-text-secondary ml-2">({e.size})</span>
            </div>
          ))}
          {localEmails.length === 0 && <div className="text-xs text-text-secondary">空</div>}
        </div>
      </div>
      {log.length > 0 && (
        <div className="bg-bg-muted rounded-lg p-3 mb-4 max-h-24 overflow-y-auto text-xs font-mono text-text-secondary">
          {log.map((l, i) => <div key={i}>{l}</div>)}
        </div>
      )}
      <button onClick={reset} className="px-4 py-2 bg-bg-subtle text-text-secondary rounded hover:bg-bg-muted text-sm mb-4">重置</button>
      <div className="grid grid-cols-2 gap-2 text-xs text-text-secondary">
        <div className="p-2 bg-bg-muted rounded"><strong>IMAP:</strong> 邮件保留在服务器,支持多设备同步、文件夹管理</div>
        <div className="p-2 bg-bg-muted rounded"><strong>POP3:</strong> 下载后删除,仅单设备访问,简单但功能有限</div>
      </div>
    </div>
  );
}

export default IMAPvsPOP3Demo;
