"use client";

import { useState } from "react";

const steps = [
  {
    label: "RegisterDeviceAPI",
    color: "from-violet-500 to-purple-600",
    code: `TVM_REGISTER_GLOBAL("device_api.cuda")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CUDADeviceAPI::Global();
});`,
    desc: "将 DeviceAPI 实例注册到全局注册表",
  },
  {
    label: "实现接口",
    color: "from-indigo-500 to-blue-600",
    code: `class CUDADeviceAPI : public DeviceAPI {
  void SetDevice(int dev_id) override;
  void* AllocMemory(size_t, size_t) override;
  void FreeMemory(void*) override;
  void CopyDataFromTo(const void*, void*,
                      size_t, cudaMemcpyKind) override;
  void StreamSync(int dev_id, TVMStreamHandle) override;
};`,
    desc: "继承 DeviceAPI，实现内存管理和数据传输接口",
  },
  {
    label: "运行时获取",
    color: "from-blue-500 to-cyan-600",
    code: `auto* api = DeviceAPI::Get(Device{kDLCUDA, 0});
void* ptr = api->AllocMemory(1024);
api->CopyDataFromTo(src, dst, size, kDLCPUToCUDA);`,
    desc: "通过 DeviceAPI::Get 获取实例并调用",
  },
];

export function DeviceAPIRegistration() {
  const [active, setActive] = useState(0);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        DeviceAPI 注册机制
      </h3>

      <div className="flex items-center gap-3 mb-6">
        {steps.map((s, i) => (
          <div key={s.label} className="flex items-center">
            <button
              onClick={() => setActive(i)}
              className={`px-4 py-2 rounded-lg text-sm font-bold transition-all ${
                active === i
                  ? `bg-gradient-to-r ${s.color} text-white shadow-lg scale-105`
                  : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700 hover:border-indigo-300"
              }`}
            >
              {s.label}
            </button>
            {i < steps.length - 1 && (
              <svg
                className="w-6 h-6 text-slate-400 mx-1"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            )}
          </div>
        ))}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-5 border border-slate-200 dark:border-slate-700">
        <h4 className="font-bold text-slate-800 dark:text-slate-100 mb-2">
          {steps[active].label}
        </h4>
        <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
          {steps[active].desc}
        </p>
        <pre className="bg-slate-900 text-green-400 p-4 rounded-lg text-sm font-mono overflow-x-auto">
          {steps[active].code}
        </pre>
      </div>

      <div className="mt-4 flex gap-4">
        <button
          onClick={() => setActive(Math.max(0, active - 1))}
          disabled={active === 0}
          className="px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg text-sm font-bold disabled:opacity-30 text-slate-700 dark:text-slate-300"
        >
          ← 上一步
        </button>
        <button
          onClick={() => setActive(Math.min(steps.length - 1, active + 1))}
          disabled={active === steps.length - 1}
          className="px-4 py-2 bg-indigo-500 text-white rounded-lg text-sm font-bold disabled:opacity-30"
        >
          下一步 →
        </button>
      </div>
    </div>
  );
}
