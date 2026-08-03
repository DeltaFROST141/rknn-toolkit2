#!/usr/bin/env python3
"""RKNN 单核多模型调度仿真：串行轮询 vs CBS/TokenBucket + SP + M3 guard band.

不依赖第三方库。离散事件模拟非抢占 NPU 链路。

网络类比
--------
- 链路带宽     -> 单核 1000 NPU-ms/s
- 流/包长      -> 模型 / 单次推理耗时
- CIR          -> cost_ms * target_fps
- CBS credit   -> 每流整形
- SP           -> M1 > M2 > M3
- guard band   -> M3 发前准入（硬/软两种）
- AQM          -> 每模型 latest-only（默认可改队列深度）

重要结论（会在输出中体现）
--------------------------
M1@30fps 周期≈33ms，M3 包长 70ms 且不可分段。若 guard 要保护
每一次 M1 硬截止期，则 M3 永远无法发送。latest-only 下 M3 每跑
一次，M1 会永久丢掉约 2 个相机周期，故平均帧率也会受损。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


SECOND = 1000.0


@dataclass
class ModelCfg:
    name: str
    cost_ms: float
    target_fps: float
    priority: int  # smaller = higher
    cir_ms_per_s: float = 0.0
    max_credit_ms: float = 0.0
    min_credit_ms: float = 0.0
    queue_depth: int = 1

    def __post_init__(self) -> None:
        if self.cir_ms_per_s <= 0:
            self.cir_ms_per_s = self.cost_ms * self.target_fps
        if self.max_credit_ms <= 0:
            self.max_credit_ms = self.cost_ms * 2
        if self.min_credit_ms >= 0:
            self.min_credit_ms = -self.cost_ms


@dataclass
class Job:
    model: str
    enqueue_ts: float
    frame_id: int


@dataclass
class ModelState:
    cfg: ModelCfg
    credit: float = 0.0
    queue: List[Job] = field(default_factory=list)
    next_release_ts: float = 0.0
    released: int = 0
    enqueued: int = 0
    dropped_on_enqueue: int = 0
    started: int = 0
    finished: int = 0
    blocked_no_credit: int = 0
    blocked_guard: int = 0
    total_latency_ms: float = 0.0
    finish_ts: List[float] = field(default_factory=list)

    @property
    def pending(self) -> bool:
        return bool(self.queue)


class BaseSimulator:
    def __init__(self, models: List[ModelCfg], duration_ms: float, link_ms_per_s: float = 1000.0):
        self.duration_ms = duration_ms
        self.link_ms_per_s = link_ms_per_s
        self.states: Dict[str, ModelState] = {
            m.name: ModelState(cfg=m, next_release_ts=0.0) for m in models
        }
        self.now = 0.0
        self.busy_until = 0.0
        self.busy_model: Optional[str] = None
        self.busy_job: Optional[Job] = None
        self.total_busy_ms = 0.0

    def period(self, st: ModelState) -> float:
        return SECOND / st.cfg.target_fps

    def enqueue(self, st: ModelState, job: Job) -> None:
        depth = st.cfg.queue_depth
        if len(st.queue) >= depth:
            # 丢最旧，保留最新（实时流 AQM）
            st.queue.pop(0)
            st.dropped_on_enqueue += 1
        else:
            st.enqueued += 1
        st.queue.append(job)

    def release_due_jobs(self) -> None:
        for st in self.states.values():
            while st.next_release_ts <= self.now + 1e-9 and st.next_release_ts <= self.duration_ms + 1e-9:
                st.released += 1
                job = Job(model=st.cfg.name, enqueue_ts=st.next_release_ts, frame_id=st.released)
                self.enqueue(st, job)
                st.next_release_ts += self.period(st)

    def finish_if_needed(self) -> None:
        if self.busy_model is None or self.now + 1e-9 < self.busy_until:
            return
        st = self.states[self.busy_model]
        assert self.busy_job is not None
        st.finished += 1
        st.total_latency_ms += self.busy_until - self.busy_job.enqueue_ts
        st.finish_ts.append(self.busy_until)
        self.total_busy_ms += st.cfg.cost_ms
        self.busy_model = None
        self.busy_job = None

    def start_job(self, name: str) -> None:
        st = self.states[name]
        assert st.queue
        job = st.queue.pop(0)
        st.started += 1
        self.busy_model = name
        self.busy_job = job
        self.busy_until = self.now + st.cfg.cost_ms

    def next_release_event(self) -> float:
        return min(st.next_release_ts for st in self.states.values())

    def metrics(self) -> Dict:
        out = {
            "sim_duration_s": self.duration_ms / SECOND,
            "link_util": self.total_busy_ms / self.duration_ms,
            "models": {},
        }
        dur_s = self.duration_ms / SECOND
        for name, st in self.states.items():
            achieved = st.finished / dur_s if dur_s > 0 else 0.0
            avg_lat = (st.total_latency_ms / st.finished) if st.finished else math.nan
            fps_est = achieved
            if len(st.finish_ts) >= 4:
                cut = max(1, int(len(st.finish_ts) * 0.1))
                ts = st.finish_ts[cut:]
                gaps = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
                if gaps:
                    fps_est = SECOND / (sum(gaps) / len(gaps))
            out["models"][name] = {
                "target_fps": st.cfg.target_fps,
                "cost_ms": st.cfg.cost_ms,
                "cir_ms_per_s": st.cfg.cir_ms_per_s,
                "queue_depth": st.cfg.queue_depth,
                "released": st.released,
                "finished": st.finished,
                "dropped_on_enqueue": st.dropped_on_enqueue,
                "blocked_no_credit": st.blocked_no_credit,
                "blocked_guard": st.blocked_guard,
                "achieved_fps": achieved,
                "steady_fps_est": fps_est,
                "fps_ratio": achieved / st.cfg.target_fps if st.cfg.target_fps else math.nan,
                "avg_latency_ms": avg_lat,
                "npu_share": (st.finished * st.cfg.cost_ms) / self.duration_ms,
            }
        return out


class SerialRoundRobin(BaseSimulator):
    """现状：固定顺序有 pending 就跑。"""

    def __init__(self, models: List[ModelCfg], duration_ms: float):
        super().__init__(models, duration_ms)
        self.order = [m.name for m in models]
        self.rr_idx = 0

    def pick(self) -> Optional[str]:
        n = len(self.order)
        for i in range(n):
            name = self.order[(self.rr_idx + i) % n]
            if self.states[name].pending:
                self.rr_idx = (self.rr_idx + i + 1) % n
                return name
        return None

    def run(self) -> Dict:
        while self.now < self.duration_ms - 1e-9:
            self.release_due_jobs()
            self.finish_if_needed()
            if self.busy_model is not None:
                self.now = min(self.busy_until, self.duration_ms)
                continue
            picked = self.pick()
            if picked is not None:
                self.start_job(picked)
                continue
            nxt = self.next_release_event()
            if nxt >= self.duration_ms:
                break
            self.now = nxt
        self.now = self.duration_ms
        self.finish_if_needed()
        return self.metrics()


class CbsSpGuardSimulator(BaseSimulator):
    """CBS 每流整形 + SP + M3 guard band.

    guard 模式
    ----------
    hard: 若 now→M1下次释放 的空隙 < M3.cost+margin，则不开闸。
          M1@30fps 时空隙≤33ms < 72ms，M3 会饿死（用于说明硬门控不可行）。
    soft: 按 M1 完成帧相对目标的滞后做准入（允许抖动，保平均帧率）。
          M1 已落后超过 lag_tol 帧时不开 M3 闸；SP+CBS 仍优先 M1。
    """

    def __init__(
        self,
        models: List[ModelCfg],
        duration_ms: float,
        guard_mode: str = "soft",
        guard_margin_ms: float = 2.0,
        high_prio_name: str = "M1",
        jumbo_name: str = "M3",
        m1_lag_tol_frames: float = 1.0,
    ):
        super().__init__(models, duration_ms)
        if guard_mode not in ("soft", "hard"):
            raise ValueError("guard_mode must be soft|hard")
        self.guard_mode = guard_mode
        self.guard_margin_ms = guard_margin_ms
        self.high_prio_name = high_prio_name
        self.jumbo_name = jumbo_name
        self.m1_lag_tol_frames = m1_lag_tol_frames
        self.last_credit_ts = 0.0

    def update_credits(self) -> None:
        dt = (self.now - self.last_credit_ts) / SECOND
        if dt <= 0:
            return
        for name, st in self.states.items():
            idle_slope = st.cfg.cir_ms_per_s
            if self.busy_model == name:
                st.credit += (idle_slope - self.link_ms_per_s) * dt
            else:
                st.credit += idle_slope * dt
            st.credit = min(max(st.credit, st.cfg.min_credit_ms), st.cfg.max_credit_ms)
        self.last_credit_ts = self.now

    def time_to_next_high_release(self) -> float:
        hp = self.states[self.high_prio_name]
        if hp.pending:
            return 0.0
        return max(0.0, hp.next_release_ts - self.now)

    def m1_frame_lag(self) -> float:
        """M1 相对目标帧率的完成滞后（帧）。负值表示超前。"""
        hp = self.states[self.high_prio_name]
        expected = (self.now / SECOND) * hp.cfg.target_fps
        return expected - hp.finished

    def guard_blocks_jumbo(self) -> bool:
        jumbo = self.states[self.jumbo_name]
        if self.guard_mode == "hard":
            # 保护下一次 M1 硬截止期：空隙不够放大包
            need = jumbo.cfg.cost_ms + self.guard_margin_ms
            return self.time_to_next_high_release() < need
        # soft: M1 平均帧率滞后超阈值则禁止巨型帧（类 EF 保护）
        return self.m1_frame_lag() > self.m1_lag_tol_frames

    def credit_ok(self, name: str) -> bool:
        return self.states[name].credit >= 0

    def pick(self) -> Optional[str]:
        ordered = sorted(self.states.keys(), key=lambda n: self.states[n].cfg.priority)
        # 先看高优先级是否可跑，避免低优先级误计数
        for name in ordered:
            st = self.states[name]
            if not st.pending:
                continue
            if not self.credit_ok(name):
                st.blocked_no_credit += 1
                continue
            if name == self.jumbo_name and self.guard_blocks_jumbo():
                st.blocked_guard += 1
                continue
            return name
        return None

    def next_credit_cross_zero(self) -> float:
        t = math.inf
        for name, st in self.states.items():
            if not st.pending or st.credit >= 0 or self.busy_model == name:
                continue
            idle_slope = st.cfg.cir_ms_per_s
            if idle_slope <= 0:
                continue
            t = min(t, self.now + (-st.credit / idle_slope) * SECOND)
        return t

    def run(self) -> Dict:
        while self.now < self.duration_ms - 1e-9:
            self.release_due_jobs()
            self.update_credits()
            self.finish_if_needed()
            self.update_credits()

            if self.busy_model is not None:
                self.now = min(self.busy_until, self.duration_ms)
                continue

            picked = self.pick()
            if picked is not None:
                self.start_job(picked)
                continue

            nxt = self.next_release_event()
            nxt_credit = self.next_credit_cross_zero()
            # hard 门控：等到 M1 释放节奏变化后再评估空隙
            nxt_guard = math.inf
            if self.guard_mode == "hard":
                jumbo = self.states[self.jumbo_name]
                hp = self.states[self.high_prio_name]
                if jumbo.pending and self.credit_ok(self.jumbo_name) and self.guard_blocks_jumbo():
                    nxt_guard = hp.next_release_ts
            cand = min(nxt, nxt_credit, nxt_guard, self.duration_ms)
            if cand <= self.now + 1e-9:
                cand = min(max(nxt, self.now + 0.001), self.duration_ms)
            self.now = cand

        self.now = self.duration_ms
        self.update_credits()
        self.finish_if_needed()
        return self.metrics()


def clone_models(models: List[ModelCfg]) -> List[ModelCfg]:
    return [
        ModelCfg(
            name=m.name,
            cost_ms=m.cost_ms,
            target_fps=m.target_fps,
            priority=m.priority,
            cir_ms_per_s=m.cir_ms_per_s,
            max_credit_ms=m.max_credit_ms,
            min_credit_ms=m.min_credit_ms,
            queue_depth=m.queue_depth,
        )
        for m in models
    ]


def fmt_metrics(title: str, metrics: Dict) -> str:
    lines = [
        f"=== {title} ===",
        f"仿真时长: {metrics['sim_duration_s']:.1f}s  链路利用率: {metrics['link_util']*100:.1f}%",
        f"{'模型':<4} {'目标':>6} {'实测fps':>8} {'达成率':>8} {'完成':>6} "
        f"{'覆盖丢':>6} {'CBS挡':>6} {'门控挡':>6} {'均延迟':>8} {'占用':>7}",
    ]
    for name, m in metrics["models"].items():
        lat = m["avg_latency_ms"]
        lat_s = f"{lat:8.2f}" if lat == lat else f"{'nan':>8}"
        lines.append(
            f"{name:<4} {m['target_fps']:6.1f} {m['achieved_fps']:8.2f} {m['fps_ratio']*100:7.1f}% "
            f"{m['finished']:6d} {m['dropped_on_enqueue']:6d} {m['blocked_no_credit']:6d} "
            f"{m['blocked_guard']:6d} {lat_s} {m['npu_share']*100:6.1f}%"
        )
    cir_sum = sum(x["cir_ms_per_s"] for x in metrics["models"].values())
    lines.append(f"CIR 合计: {cir_sum:.1f} ms/s / 链路 1000 ms/s")
    return "\n".join(lines)


def theoretical_note(m1_fps: float, m1_cost: float, m3_fps: float, m3_cost: float) -> str:
    m1_period = SECOND / m1_fps
    lost_per_m3 = m3_cost / m1_period  # 约等于大包占用期间错过的 M1 周期数
    m1_ceil = m1_fps - m3_fps * lost_per_m3
    return (
        "【可调度性提示】latest-only + 非抢占:\n"
        f"  M1 周期={m1_period:.2f}ms, M3 包长={m3_cost:.1f}ms, "
        f"硬门控要求空隙>={m3_cost:.1f}ms → "
        f"{'不可行' if m1_period < m3_cost else '可能'}（hard guard）\n"
        f"  粗估 M3@{m3_fps}fps 时 M1 平均上限 ≈ {m1_fps:.1f} - {m3_fps:.1f}*{lost_per_m3:.2f} "
        f"= {m1_ceil:.1f} fps（软门控、SP 下仍会被大包打洞）\n"
        f"  CIR 账: M1={m1_cost*m1_fps:.0f} + M3={m3_cost*m3_fps:.0f} + … "
        f"容量够 ≠ 截止期/最新帧帧率够"
    )


def run_case(
    title: str,
    models: List[ModelCfg],
    duration_s: float,
    guard_mode: str,
    guard_margin_ms: float,
) -> str:
    dur = duration_s * SECOND
    serial = SerialRoundRobin(clone_models(models), dur).run()
    qos = CbsSpGuardSimulator(
        clone_models(models), dur, guard_mode=guard_mode, guard_margin_ms=guard_margin_ms
    ).run()
    return (
        f"\n****** {title} | guard={guard_mode} ******\n"
        + fmt_metrics("串行轮询 RR", serial)
        + "\n\n"
        + fmt_metrics(f"CBS+SP+Guard({guard_mode})", qos)
    )


def main() -> None:
    p = argparse.ArgumentParser(description="NPU CBS/SP/Guard-band scheduler simulator")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--guard-mode", choices=["soft", "hard"], default="soft")
    p.add_argument("--guard-margin", type=float, default=2.0)
    p.add_argument("--m1-fps", type=float, default=30.0)
    p.add_argument("--m2-fps", type=float, default=15.0)
    p.add_argument("--m3-fps", type=float, default=5.0)
    p.add_argument("--m1-queue", type=int, default=1, help="M1 队列深度，>1 可追赶补帧")
    p.add_argument("--all-cases", action="store_true", help="跑多组对比场景")
    args = p.parse_args()

    base = [
        ModelCfg("M1", 7.0, args.m1_fps, 0, queue_depth=args.m1_queue),
        ModelCfg("M2", 15.0, args.m2_fps, 1),
        ModelCfg("M3", 70.0, args.m3_fps, 2),
    ]

    print("RKNN 单核调度仿真 (CBS/TokenBucket + SP + M3 guard band)")
    print(
        f"参数: M1=7ms@{args.m1_fps}fps(q={args.m1_queue}), "
        f"M2=15ms@{args.m2_fps}fps, M3=70ms@{args.m3_fps}fps, "
        f"guard={args.guard_mode}, margin={args.guard_margin}ms\n"
    )
    print(theoretical_note(args.m1_fps, 7.0, args.m3_fps, 70.0))
    print(
        run_case(
            "主场景",
            base,
            args.duration,
            args.guard_mode,
            args.guard_margin,
        )
    )

    if args.all_cases:
        # 硬门控：说明 M3 饿死
        print(
            run_case(
                "对照：硬门控（保护每次 M1 截止期）",
                [
                    ModelCfg("M1", 7.0, 30.0, 0),
                    ModelCfg("M2", 15.0, 15.0, 1),
                    ModelCfg("M3", 70.0, 5.0, 2),
                ],
                args.duration,
                "hard",
                args.guard_margin,
            )
        )
        # 可行平均帧率组合
        print(
            run_case(
                "对照：可达成目标 M1=20/M2=15/M3=5",
                [
                    ModelCfg("M1", 7.0, 20.0, 0),
                    ModelCfg("M2", 15.0, 15.0, 1),
                    ModelCfg("M3", 70.0, 5.0, 2),
                ],
                args.duration,
                "soft",
                args.guard_margin,
            )
        )
        # M1 允许深度排队追赶（像网络缓冲，不再是纯实时）
        print(
            run_case(
                "对照：M1 队列深度=4（允许追赶，延迟换帧率）",
                [
                    ModelCfg("M1", 7.0, 30.0, 0, queue_depth=4),
                    ModelCfg("M2", 15.0, 15.0, 1),
                    ModelCfg("M3", 70.0, 5.0, 2),
                ],
                args.duration,
                "soft",
                args.guard_margin,
            )
        )
        # 过载：M3 要 10fps
        print(
            run_case(
                "对照：过载 M3@10fps",
                [
                    ModelCfg("M1", 7.0, 30.0, 0),
                    ModelCfg("M2", 15.0, 15.0, 1),
                    ModelCfg("M3", 70.0, 10.0, 2),
                ],
                args.duration,
                "soft",
                args.guard_margin,
            )
        )


if __name__ == "__main__":
    main()
