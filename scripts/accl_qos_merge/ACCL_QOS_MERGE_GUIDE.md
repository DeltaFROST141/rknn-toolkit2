# ACCL Core0 多模型 QoS 调度接入指南（CBS + SP + soft guard）

本文档说明如何把 `NpuQosSchedulerAccl` 接到现有 ACCL 推理链路。  
按步骤改即可，不必整文件替换。

---

## 0. 改之前先对齐认知

### 现有链路（保持不动的部分）

```
main.submit
  → PreprocessServiceAccl   （RateGate + latest-only + 写 NPU input ring）
  → InferServiceAccl        （当前 FIFO 串行 rknn_run）← 主要改这里
  → YOLO26Accl::run_after_preprocess
```

### 本次目标

| 项 | 做法 |
|----|------|
| Core0（OUTER / TOF / ROPE） | FIFO → CBS + SP + soft guard |
| Core1（INNER） | **不改**，继续 FIFO |
| Preprocess / RateGate / YOLO26Accl | **先不改** |

### 机制对应

| 机制 | 作用 |
|------|------|
| CBS | 每模型按 CIR=`cost_ms × target_fps` 攒/扣 credit，超速暂时不跑 |
| SP | 有额度时优先级：OUTER > TOF > ROPE |
| soft guard | ROPE（大包）仅在 OUTER 平均帧率未明显落后时才允许启动 |

### 默认参数（务必按板上 `run_us` 再改）

| 任务 | priority | target_fps | cost_ms 初值 | jumbo |
|------|----------|------------|--------------|-------|
| OUTER | 0 | 10（interval 100ms） | 7 | 否 |
| TOF | 1 | ≈14.93（interval 67ms） | 15 | 否 |
| ROPE | 2 | 10（interval 100ms） | 70 | 是 |

> 若 RateGate 满额且 cost 接近上表，Core0 占用约 **994 ms/s**，几乎满载。  
> 上线前请用 `perf_log_infer` 的 `run_us` 更新 `cost_ms`。

---

## 1. 新增文件（1 个）

把仓库中的：

```text
scripts/accl_qos_merge/npu_qos_scheduler_accl.h
```

复制到你工程里与 `infer_service_accl.h` 同级（或你习惯的 include 目录），例如：

```text
your_project/.../npu_qos_scheduler_accl.h
```

该头文件依赖已有：

```cpp
#include "infer_task_accl.h"
```

确保 include 路径能找到它。  
纯头文件实现，一般**不用单独加 .cpp 到工程**（只要有源文件 include 它即可）。

---

## 2. 修改 `infer_service_accl.h`

### 2.1 增加 include

在其它 include 附近增加：

```cpp
#include "npu_qos_scheduler_accl.h"
```

### 2.2 在 `public:` 增加接口

```cpp
// 仅 Core0 多模型打开；Core1 保持 false
void enable_qos(bool on);
bool qos_enabled() const { return m_qos_enabled; }
```

### 2.3 在 `private:` 增加成员

```cpp
bool m_qos_enabled = false;
NpuQosSchedulerAccl m_qos;
```

---

## 3. 修改 `infer_service_accl.cpp`

### 3.1 实现 `enable_qos`

放在 `bind_model` 附近即可：

```cpp
void InferServiceAccl::enable_qos(bool on)
{
    m_qos_enabled = on;
    if (on) {
        m_qos.init_core0_defaults();
        // 可选：按实测重设（若你给 NpuQosSchedulerAccl 加了 set_cost 接口）
        // 当前头文件可用 add_flow 前先 reset，或改 init_core0_defaults 里的常数
        CAM_LOGI(CAM_LOG_MOD_YOLO, "[%s] QoS enabled (CBS+SP+soft)", m_core_tag);
    } else {
        CAM_LOGI(CAM_LOG_MOD_YOLO, "[%s] QoS disabled (FIFO)", m_core_tag);
    }
}
```

### 3.2 替换 `loop_process`

用同目录文件：

```text
scripts/accl_qos_merge/infer_service_loop_process_qos.cpp.fragment
```

作为对照，把 `InferServiceAccl::loop_process` **整函数替换**为 fragment 中的实现。

注意把 fragment 末尾注释掉的「原有日志 / FPS 统计」按你现有代码补全：

- `actual_interval_us` / `submit_to_start_us` / `dispatch_gap_us`
- `perf_log_schedule(...)`
- `m_count_infer` + `print_fps("Infer", ...)`

关键点（fragment 已处理）：

1. `m_qos_enabled == false` → 走旧 FIFO（给 Core1 用）  
2. `m_qos_enabled == true` → 用 `m_latest_has[]` 做就绪集，调用 `m_qos.pick()`  
3. pick 失败（credit/guard）→ `wait_for(suggest_wait_ms)` 再试  
4. 未选中但仍有 `latest_has` 的 type → **重新入队**，防止饿死  
5. `run_after_preprocess` 后用整段耗时调用 `m_qos.on_complete(type, actual_cost_ms)`

### 3.3 `update_task` / `try_get`

**先不要改。**  
继续 latest-only + 同 type 去重即可。

---

## 4. 修改 main（打开开关）

找到创建两个 Infer 实例的地方，类似：

```cpp
InferServiceAccl infer_core0("core0");
InferServiceAccl infer_core1("core1");
```

在 `bind_model` / `start` 之前或之后加：

```cpp
infer_core0.enable_qos(true);   // Core0：OUTER/TOF/ROPE
infer_core1.enable_qos(false);  // Core1：INNER，保持 FIFO（默认 false 也可不写）
```

然后照常：

```cpp
infer_core0.bind_model(...);
infer_core1.bind_model(...);
infer_core0.start();
infer_core1.start();
```

---

## 5. 如何改默认 cost / 优先级（可选）

打开 `npu_qos_scheduler_accl.h` 中：

```cpp
void init_core0_defaults()
{
    reset();
    add_flow({TaskTypeAccl::PEOPLE_OUTER, 0, 10.f, 7.f, false, true});
    add_flow({TaskTypeAccl::PEOPLE_TOF, 1, 1000.f / 67.f, 15.f, false, true});
    add_flow({TaskTypeAccl::ROPE_MID, 2, 10.f, 70.f, true, true});
    m_high_prio = TaskTypeAccl::PEOPLE_OUTER;
    m_jumbo = TaskTypeAccl::ROPE_MID;
}
```

参数顺序：`{type, priority, target_fps, cost_ms, is_jumbo, enabled}`

建议流程：

1. 先跑起来看 `perf_log_infer` 的 `run_us`  
2. 把 `cost_ms` 改成 `run_us/1000` 的稳态值  
3. 若要保 TOF 而不是 OUTER，把 `m_high_prio` 改成 `PEOPLE_TOF`，并调整 `priority`

soft 灵敏度：

```cpp
m_qos.set_guard(NpuGuardMode::Soft, 1.0f);  // lag_tol=1 帧
```

---

## 6. 编译检查清单

- [ ] 新头文件在 include 路径内  
- [ ] `infer_service_accl.h` 已声明 `enable_qos` / `m_qos`  
- [ ] `infer_service_accl.cpp` 已实现 `enable_qos` 且 `loop_process` 已替换  
- [ ] main 仅对 **core0** `enable_qos(true)`  
- [ ] 链接无缺符号（纯头文件一般不会）  
- [ ] 预处理 / YOLO 文件未误改  

---

## 7. 上板验证建议

1. **对比开关**  
   - `enable_qos(false)`：旧 FIFO  
   - `enable_qos(true)`：新调度  

2. **看日志**  
   - Infer FPS：OUTER / TOF / ROPE 是否更接近 RateGate 目标  
   - `perf_log_schedule`：`submit_to_start` 是否对 OUTER 更友好  
   - ROPE 变慢或被压是预期（保 OUTER）  

3. **过载试一下**  
   - 临时把 ROPE 的 RateGate 间隔改小，观察 OUTER 是否仍优先  

4. **若 OUTER 仍掉帧**  
   - 降低 ROPE `target_fps` / 增大 ROPE RateGate interval  
   - 或把 ROPE `cost_ms` 按实测调大（CIR 更大 → 更易被整形）  
   - 检查 DDR/ISP 共场（调度解决不了带宽打满）  

---

## 8. 常见坑

| 现象 | 原因 | 处理 |
|------|------|------|
| 编译找不到 `NpuQosSchedulerAccl` | 没拷头文件 / include 路径不对 | 检查路径 |
| Core1 行为变了 | 误对 core1 `enable_qos(true)` | 只开 core0 |
| 有 latest 但再不推理 | pick 失败后未重新入队 | 确认用了最新 fragment（含重新入队） |
| ROPE 几乎不跑 | soft 保 OUTER + Core0 算力已满 | 降 ROPE 目标或接受被压制 |
| 和仿真差很多 | cost/到达模型与仿真不一致 | 用实测 `run_us` 重填 cost |
| 双重限速过狠 | RateGate + CBS 同时限 | 先保留两者；过稳后再放宽 RateGate |

---

## 9. 文件清单（仓库内）

```text
scripts/accl_qos_merge/
├── ACCL_QOS_MERGE_GUIDE.md                      ← 本文档
├── npu_qos_scheduler_accl.h                     ← 拷贝到工程（新文件）
├── infer_service_accl.h.patch.md                ← 头文件改动摘要
└── infer_service_loop_process_qos.cpp.fragment ← loop_process 对照实现
```

---

## 10. 建议修改顺序（最短路径）

1. 拷贝 `npu_qos_scheduler_accl.h`  
2. 改 `infer_service_accl.h`（include + 成员 + 声明）  
3. 改 `infer_service_accl.cpp`（`enable_qos` + `loop_process`）  
4. main 里 `infer_core0.enable_qos(true)`  
5. 编译 → 上板 → 对比 FIFO/QoS 日志  
6. 用实测 `run_us` 回填 `init_core0_defaults()` 的 `cost_ms`
