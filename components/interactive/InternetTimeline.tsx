"use client";
import { useState } from "react";

const milestones = [
  { year: 1969, event: "ARPANET诞生", detail: "UCLA与SRI之间首次分组交换通信，4个节点" },
  { year: 1971, event: "电子邮件发明", detail: "Ray Tomlinson发送第一封网络邮件，使用@符号" },
  { year: 1973, event: "TCP/IP雏形", detail: "Vint Cerf和Bob Kahn设计TCP协议" },
  { year: 1983, event: "TCP/IP正式采用", detail: "ARPANET切换到TCP/IP，现代互联网的生日" },
  { year: 1984, event: "DNS诞生", detail: "域名系统取代hosts文件，层次化命名" },
  { year: 1989, event: "万维网提案", detail: "Tim Berners-Lee在CERN提出WWW构想" },
  { year: 1991, event: "WWW公开", detail: "第一个网页上线，HTTP/HTML标准形成" },
  { year: 1993, event: "Mosaic浏览器", detail: "图形化浏览器推动互联网普及" },
  { year: 1995, event: "商业化浪潮", detail: "Netscape IPO，Amazon/eBay成立" },
  { year: 1998, event: "Google成立", detail: "搜索引擎革命，互联网信息检索质变" },
  { year: 2004, event: "Web 2.0时代", detail: "Facebook上线，用户生成内容(UGC)兴起" },
  { year: 2007, event: "iPhone发布", detail: "移动互联网时代开启" },
  { year: 2010, event: "云计算普及", detail: "AWS引领IaaS，网络架构云化" },
  { year: 2015, event: "SDN/NFV商用", detail: "软件定义网络和网络功能虚拟化进入生产环境" },
  { year: 2020, event: "5G+IoT", detail: "第五代移动通信，万物互联" },
];

const eras = [
  { name: "ARPANET时代", range: "1969-1982", color: "bg-yellow-500" },
  { name: "互联网标准化", range: "1983-1994", color: "bg-blue-500" },
  { name: "商业化浪潮", range: "1995-2005", color: "bg-green-500" },
  { name: "移动与云时代", range: "2006-2019", color: "bg-purple-500" },
  { name: "智能互联", range: "2020+", color: "bg-red-500" },
];

export function InternetTimeline() {
  const [selected, setSelected] = useState(7);
  const [searchYear, setSearchYear] = useState("");
  const [eraFilter, setEraFilter] = useState<number | null>(null);

  const filtered = milestones.filter((m) => {
    const yearMatch = searchYear ? m.year.toString().includes(searchYear) : true;
    let eraMatch = true;
    if (eraFilter !== null) {
      const ranges: [number, number][] = [[1969, 1982], [1983, 1994], [1995, 2005], [2006, 2019], [2020, 2030]];
      const [lo, hi] = ranges[eraFilter];
      eraMatch = m.year >= lo && m.year <= hi;
    }
    return yearMatch && eraMatch;
  });

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        Internet Timeline <span className="text-text-secondary text-sm">— 互联网发展里程碑</span>
      </h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {eras.map((e, i) => (
          <button
            key={i}
            onClick={() => setEraFilter(eraFilter === i ? null : i)}
            className={`px-2 py-1 rounded text-xs text-white ${e.color} ${eraFilter === i ? "ring-2 ring-offset-1 ring-blue-400" : "opacity-60"}`}
          >
            {e.name}
          </button>
        ))}
      </div>
      <input
        type="text"
        value={searchYear}
        onChange={(e) => setSearchYear(e.target.value)}
        placeholder="搜索年份..."
        className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-text-primary text-sm mb-4"
      />
      <div className="relative">
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-300 dark:bg-gray-600" />
        <div className="space-y-3">
          {filtered.map((m, i) => {
            const realIdx = milestones.indexOf(m);
            return (
              <button
                key={realIdx}
                onClick={() => setSelected(realIdx)}
                className={`relative ml-8 p-3 rounded-lg w-[calc(100%-2rem)] text-left transition-all ${selected === realIdx ? "bg-blue-100 dark:bg-blue-900/40 ring-1 ring-blue-400" : "bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700"}`}
              >
                <div className="absolute -left-6 top-4 w-3 h-3 rounded-full bg-blue-600" />
                <div className="flex items-baseline gap-2">
                  <span className="font-mono font-bold text-blue-600 dark:text-blue-400">{m.year}</span>
                  <span className="font-semibold text-text-primary">{m.event}</span>
                </div>
                {selected === realIdx && (
                  <p className="text-sm text-text-secondary mt-1">{m.detail}</p>
                )}
              </button>
            );
          })}
        </div>
      </div>
      <div className="mt-4 text-xs text-text-secondary">
        显示 {filtered.length}/{milestones.length} 个里程碑
      </div>
    </div>
  );
}

export default InternetTimeline;
