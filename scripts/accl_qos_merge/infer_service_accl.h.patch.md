# InferServiceAccl 接入说明（对照用，不是 git patch）

## 1. `infer_service_accl.h` 增加

```cpp
#include "npu_qos_scheduler_accl.h"

class InferServiceAccl : public SubThread {
public:
    // ...
    // 仅 Core0 多模型实例打开 QoS；Core1 可关掉走旧 FIFO
    void enable_qos(bool on);
    bool qos_enabled() const { return m_qos_enabled; }

private:
    bool m_qos_enabled = false;
    NpuQosSchedulerAccl m_qos;
};
```

## 2. 构造 / start 时（Core0）

```cpp
// main 里创建 infer_core0 后：
infer_core0.enable_qos(true);  // 内部 init_core0_defaults()
```

```cpp
void InferServiceAccl::enable_qos(bool on)
{
    m_qos_enabled = on;
    if (on) {
        m_qos.init_core0_defaults();
        // 若实测耗时不同，在此按 run_us 重设 cost：
        // 可扩展 m_qos 提供 set_cost(type, ms)
    }
}
```

## 3. `loop_process` 核心替换

旧逻辑：`m_infer_queue.front()` FIFO。  
新逻辑：队列只表示“有过就绪通知”，真正选模用 `m_qos.pick()`。

见同目录 `infer_service_loop_process_qos.cpp.fragment`。

## 4. 不改动的部分

- `PreprocessServiceAccl` / RateGate：先保留
- `YOLO26Accl`：不动
- `update_task` latest-only / slot 归还：不动
- Core1 INNER：`enable_qos(false)` 继续 FIFO
