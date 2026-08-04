#ifndef NPU_QOS_SCHEDULER_ACCL_H
#define NPU_QOS_SCHEDULER_ACCL_H

// Core0 多模型：CBS(credit) + SP + soft/hard guard
// 用法：InferServiceAccl 在出队时用 pick()，跑完用 on_complete()

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#include "infer_task_accl.h"

struct NpuQosFlowCfg {
    TaskTypeAccl type = TaskTypeAccl::PEOPLE_OUTER;
    int priority = 0;          // 越小越高
    float target_fps = 10.f;
    float cost_ms = 10.f;      // 初值；建议用 run_us EMA 更新
    bool is_jumbo = false;     // 大包（如 ROPE）走 guard
    bool enabled = true;
};

struct NpuQosFlowState {
    NpuQosFlowCfg cfg;
    float credit = 0.f;
    float cir_ms_per_s = 0.f;
    float max_credit_ms = 0.f;
    float min_credit_ms = 0.f;
    uint64_t finished = 0;
    std::chrono::steady_clock::time_point last_credit_ts {};
    bool inited = false;
};

enum class NpuGuardMode { Soft, Hard };

class NpuQosSchedulerAccl {
public:
    static constexpr size_t kMaxFlows = 4;
    static constexpr float kLinkMsPerS = 1000.f;

    explicit NpuQosSchedulerAccl(NpuGuardMode guard_mode = NpuGuardMode::Soft,
                                 float lag_tol_frames = 1.f)
        : m_guard_mode(guard_mode), m_lag_tol_frames(lag_tol_frames)
    {
    }

    // Core0 默认：OUTER > TOF > ROPE；耗时请按板上 run_us 改
    void init_core0_defaults()
    {
        reset();
        add_flow({TaskTypeAccl::PEOPLE_OUTER, 0, 10.f, 7.f, false, true});
        add_flow({TaskTypeAccl::PEOPLE_TOF, 1, 1000.f / 67.f, 15.f, false, true});
        add_flow({TaskTypeAccl::ROPE_MID, 2, 10.f, 70.f, true, true});
        m_high_prio = TaskTypeAccl::PEOPLE_OUTER;
        m_jumbo = TaskTypeAccl::ROPE_MID;
    }

    void reset()
    {
        m_flow_count = 0;
        for (auto& f : m_flows) {
            f = NpuQosFlowState {};
        }
    }

    bool add_flow(const NpuQosFlowCfg& cfg)
    {
        if (m_flow_count >= kMaxFlows || !cfg.enabled) {
            return false;
        }
        auto& f = m_flows[m_flow_count++];
        f.cfg = cfg;
        f.cir_ms_per_s = cfg.cost_ms * cfg.target_fps;
        f.max_credit_ms = cfg.cost_ms * 2.f;
        f.min_credit_ms = -cfg.cost_ms;
        f.credit = 0.f;
        f.finished = 0;
        f.last_credit_ts = std::chrono::steady_clock::now();
        f.inited = true;
        return true;
    }

    void set_guard(NpuGuardMode mode, float lag_tol_frames)
    {
        m_guard_mode = mode;
        m_lag_tol_frames = lag_tol_frames;
    }

    void set_roles(TaskTypeAccl high_prio, TaskTypeAccl jumbo)
    {
        m_high_prio = high_prio;
        m_jumbo = jumbo;
    }

    // ready[i]=true 表示该 type 有 latest 可跑
    // 返回是否选中；选中 type 写入 out_type
    bool pick(const bool ready[kMaxFlows],
              TaskTypeAccl type_by_index[kMaxFlows],
              size_t n_types,
              TaskTypeAccl& out_type)
    {
        const auto now = std::chrono::steady_clock::now();
        update_credits(now);

        int best_prio = std::numeric_limits<int>::max();
        int best_flow = -1;

        for (size_t i = 0; i < n_types; ++i) {
            if (!ready[i]) {
                continue;
            }
            const int fi = find_flow(type_by_index[i]);
            if (fi < 0) {
                continue;
            }
            auto& f = m_flows[static_cast<size_t>(fi)];
            if (!f.cfg.enabled) {
                continue;
            }
            if (f.credit < 0.f) {
                continue;
            }
            if (f.cfg.is_jumbo && guard_blocks(now, f)) {
                continue;
            }
            if (f.cfg.priority < best_prio) {
                best_prio = f.cfg.priority;
                best_flow = fi;
            }
        }

        if (best_flow < 0) {
            return false;
        }
        out_type = m_flows[static_cast<size_t>(best_flow)].cfg.type;
        m_busy_flow = best_flow;
        m_busy_start = now;
        return true;
    }

    // 推理结束后调用；actual_cost_ms 用本次 run 耗时（建议）
    void on_complete(TaskTypeAccl type, float actual_cost_ms)
    {
        const auto now = std::chrono::steady_clock::now();
        update_credits(now);

        const int fi = find_flow(type);
        if (fi < 0) {
            m_busy_flow = -1;
            return;
        }
        auto& f = m_flows[static_cast<size_t>(fi)];
        if (actual_cost_ms > 0.f) {
            // EMA 更新成本与 CIR
            constexpr float a = 0.2f;
            f.cfg.cost_ms = (1.f - a) * f.cfg.cost_ms + a * actual_cost_ms;
            f.cir_ms_per_s = f.cfg.cost_ms * f.cfg.target_fps;
            f.max_credit_ms = f.cfg.cost_ms * 2.f;
            f.min_credit_ms = -f.cfg.cost_ms;
        }
        f.finished++;
        m_busy_flow = -1;
    }

    // 若因 credit/guard 暂时不能跑，建议 timed_wait 的毫秒数
    int suggest_wait_ms() const
    {
        float min_ms = 5.f;
        for (size_t i = 0; i < m_flow_count; ++i) {
            const auto& f = m_flows[i];
            if (f.credit >= 0.f) {
                continue;
            }
            if (f.cir_ms_per_s <= 1e-3f) {
                continue;
            }
            const float need_s = (-f.credit) / f.cir_ms_per_s;
            min_ms = std::min(min_ms, need_s * 1000.f);
        }
        return std::max(1, static_cast<int>(std::ceil(min_ms)));
    }

    const NpuQosFlowState* flow(TaskTypeAccl type) const
    {
        const int fi = find_flow(type);
        if (fi < 0) {
            return nullptr;
        }
        return &m_flows[static_cast<size_t>(fi)];
    }

private:
    int find_flow(TaskTypeAccl type) const
    {
        for (size_t i = 0; i < m_flow_count; ++i) {
            if (m_flows[i].cfg.type == type) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void update_credits(std::chrono::steady_clock::time_point now)
    {
        for (size_t i = 0; i < m_flow_count; ++i) {
            auto& f = m_flows[i];
            if (!f.inited) {
                continue;
            }
            if (f.last_credit_ts.time_since_epoch().count() == 0) {
                f.last_credit_ts = now;
                continue;
            }
            const float dt_s =
                std::chrono::duration<float>(now - f.last_credit_ts).count();
            if (dt_s <= 0.f) {
                continue;
            }
            if (m_busy_flow == static_cast<int>(i)) {
                f.credit += (f.cir_ms_per_s - kLinkMsPerS) * dt_s;
            } else {
                f.credit += f.cir_ms_per_s * dt_s;
            }
            f.credit = std::min(f.max_credit_ms, std::max(f.min_credit_ms, f.credit));
            f.last_credit_ts = now;
        }
    }

    float high_prio_lag_frames(std::chrono::steady_clock::time_point now) const
    {
        const int fi = find_flow(m_high_prio);
        if (fi < 0) {
            return 0.f;
        }
        const auto& f = m_flows[static_cast<size_t>(fi)];
        if (!m_start_valid) {
            return 0.f;
        }
        const float elapsed_s =
            std::chrono::duration<float>(now - m_sched_start).count();
        const float expected = elapsed_s * f.cfg.target_fps;
        return expected - static_cast<float>(f.finished);
    }

    bool guard_blocks(std::chrono::steady_clock::time_point now, const NpuQosFlowState& jumbo) const
    {
        if (m_guard_mode == NpuGuardMode::Hard) {
            // 硬门控：高优先级周期空隙不够大包则禁止
            const int fi = find_flow(m_high_prio);
            if (fi < 0) {
                return false;
            }
            const float period_ms = 1000.f / std::max(1e-3f, m_flows[static_cast<size_t>(fi)].cfg.target_fps);
            return period_ms < (jumbo.cfg.cost_ms + 2.f);
        }
        // soft：高优先级平均帧率滞后超过阈值则禁止大包
        if (!m_start_valid) {
            const_cast<NpuQosSchedulerAccl*>(this)->m_sched_start = now;
            const_cast<NpuQosSchedulerAccl*>(this)->m_start_valid = true;
            return false;
        }
        return high_prio_lag_frames(now) > m_lag_tol_frames;
    }

    std::array<NpuQosFlowState, kMaxFlows> m_flows {};
    size_t m_flow_count = 0;
    int m_busy_flow = -1;
    std::chrono::steady_clock::time_point m_busy_start {};
    NpuGuardMode m_guard_mode = NpuGuardMode::Soft;
    float m_lag_tol_frames = 1.f;
    TaskTypeAccl m_high_prio = TaskTypeAccl::PEOPLE_OUTER;
    TaskTypeAccl m_jumbo = TaskTypeAccl::ROPE_MID;
    mutable bool m_start_valid = false;
    mutable std::chrono::steady_clock::time_point m_sched_start {};
};

#endif
