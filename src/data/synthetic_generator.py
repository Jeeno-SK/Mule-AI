"""Configurable, reproducible SYNTHETIC transaction generator.

All identifiers are random "SYN-######" strings (no real people or bank details). Account roles are
NOT encoded in identifiers. Ground truth is returned in separate tables and is never used as a model feature.
Patterns are illustrative simulations, not evidence of real banking behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TYPES = np.array(["UPI", "IMPS", "NEFT", "CARD"])
TYPE_P = np.array([0.50, 0.25, 0.15, 0.10])


@dataclass
class SyntheticConfig:
    seed: int = 42
    n_accounts: int = 1200
    n_legit_tx: int = 8000
    days: int = 30
    start: str = "2025-01-01"
    n_collection_rings: int = 4
    n_passthrough_chains: int = 3
    n_fanout_networks: int = 2
    n_cycle_rings: int = 1
    n_merchants: int = 3
    n_payroll: int = 2


@dataclass
class SyntheticDataset:
    transactions: pd.DataFrame
    ground_truth_accounts: pd.DataFrame   # account, role, scenario_id, scenario_type, is_mule_by_design
    ground_truth_transactions: pd.DataFrame  # transaction_id, scenario_id, scenario_type
    config: SyntheticConfig = field(default_factory=SyntheticConfig)


class _Builder:
    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.start = pd.Timestamp(cfg.start)
        self.rows: list[dict] = []
        self.roles: dict[str, tuple[str, str, str, bool]] = {}
        n_ids = cfg.n_accounts + 4 * (cfg.n_collection_rings * 6 + cfg.n_passthrough_chains * 8
                                      + cfg.n_fanout_networks * 30 + cfg.n_cycle_rings * 4 + 20)
        ids = self.rng.permutation(900000)[:n_ids] + 100000
        self.pool = [f"SYN-{i}" for i in ids]
        self.cursor = 0
        self.legit = self.take(cfg.n_accounts)

    def take(self, n: int) -> list[str]:
        out = self.pool[self.cursor:self.cursor + n]
        self.cursor += n
        return out

    def pick_legit(self, n: int) -> list[str]:
        return list(self.rng.choice(self.legit, size=min(n, len(self.legit)), replace=False))

    def tag(self, accounts, role, sid, stype, mule):
        for a in accounts:
            self.roles[a] = (role, sid, stype, mule)

    def add(self, ts, src, dst, amount, ttype=None, fraud=0, sid="", stype="legit_background"):
        self.rows.append({"timestamp": ts, "source_account": src, "destination_account": dst,
                          "amount": round(float(amount), 2),
                          "transaction_type": ttype or str(self.rng.choice(TYPES, p=TYPE_P)),
                          "is_fraud": int(fraud), "scenario_id": sid, "scenario_type": stype})

    def rand_time(self, base_day_offset: float | None = None) -> pd.Timestamp:
        day = self.rng.uniform(0, self.cfg.days - 3) if base_day_offset is None else base_day_offset
        return self.start + pd.Timedelta(days=float(day)) + pd.Timedelta(hours=float(np.clip(self.rng.normal(14, 4), 6, 23)))

    # ---- legitimate behaviour ----
    def background(self):
        cfg, rng = self.cfg, self.rng
        n = len(self.legit)
        contacts = np.array([rng.choice(n, size=rng.integers(2, 7), replace=False) for _ in range(n)], dtype=object)
        activity = rng.pareto(2.5, n) + 0.3
        scale = rng.lognormal(0, 0.6, n)
        src_idx = rng.choice(n, size=cfg.n_legit_tx, p=activity / activity.sum())
        for s in src_idx:
            d = int(rng.choice(contacts[s])) if rng.random() < 0.85 else int(rng.integers(n))
            if d == s:
                d = (d + 1) % n
            amount = rng.lognormal(np.log(1500), 0.9) * scale[s]
            self.add(self.rand_time(), self.legit[s], self.legit[d], max(amount, 10.0), sid="", stype="legit_background")

    def hard_negatives(self):
        cfg, rng = self.cfg, self.rng
        for m in self.take(cfg.n_merchants):
            self.tag([m], "legit_merchant", "", "legit_hard_negative", False)
            for c in self.pick_legit(int(rng.integers(120, 220))):
                self.add(self.rand_time(), c, m, rng.lognormal(np.log(800), 0.6), ttype="CARD", stype="legit_hard_negative")
        for p in self.take(cfg.n_payroll):
            self.tag([p], "legit_payroll", "", "legit_hard_negative", False)
            employees = self.pick_legit(int(rng.integers(18, 30)))
            for month_day in (2.0, 27.0):
                if month_day >= cfg.days - 1:
                    continue
                t0 = self.start + pd.Timedelta(days=month_day, hours=10)
                for e in employees:
                    self.add(t0 + pd.Timedelta(minutes=float(rng.uniform(0, 40))), p, e, max(12000.0, rng.normal(30000, 6000)),
                             ttype="NEFT", stype="legit_hard_negative")
            self.add(self.start + pd.Timedelta(days=1.5), self.pick_legit(1)[0], p, 900000.0, ttype="NEFT", stype="legit_hard_negative")

    # ---- simulated suspicious scenarios ----
    def camouflage(self, mules, sid, stype):
        for m in mules:
            for _ in range(int(self.rng.integers(1, 4))):
                src = self.pick_legit(1)[0]
                self.add(self.rand_time(), src, m, self.rng.lognormal(np.log(600), 0.5), sid=sid, stype=stype)

    def collection_rings(self):
        rng = self.rng
        for k in range(self.cfg.n_collection_rings):
            sid, st = f"RING{k + 1}", "collection_fan_in"
            collectors, cash = self.take(int(rng.integers(2, 4))), self.take(1)
            self.tag(collectors, "collector_mule", sid, st, True)
            self.tag(cash, "cashout_mule", sid, st, True)
            t0 = self.start + pd.Timedelta(days=float(rng.uniform(1, self.cfg.days - 4)))
            victims = self.pick_legit(int(rng.integers(8, 20)))
            self.tag(victims, "victim_legit_account", sid, st, False)
            for v in victims:
                col = collectors[int(rng.integers(len(collectors)))]
                self.add(t0 + pd.Timedelta(hours=float(rng.uniform(0, 36))), v, col, rng.uniform(5000, 90000), fraud=1, sid=sid, stype=st)
            tx = pd.DataFrame(self.rows)
            for col in collectors:
                got = tx[(tx.destination_account == col) & (tx.scenario_id == sid)]
                if got.empty:
                    continue
                total = got.amount.sum()
                last = pd.Timestamp(got.timestamp.max())
                for part in (0.55, 0.40):
                    self.add(last + pd.Timedelta(hours=float(rng.uniform(0.5, 5))), col, cash[0], total * part * rng.uniform(0.97, 1.0),
                             ttype="IMPS", fraud=1, sid=sid, stype=st)
            self.camouflage(collectors + cash, sid, st)

    def passthrough_chains(self):
        rng = self.rng
        for k in range(self.cfg.n_passthrough_chains):
            sid, st = f"CHAIN{k + 1}", "multi_hop_chain"
            length = int(rng.integers(3, 6))
            mules = self.take(length + 1)
            self.tag(mules, "chain_mule", sid, st, True)
            victim = self.pick_legit(1)[0]
            self.tag([victim], "victim_legit_account", sid, st, False)
            ts = self.start + pd.Timedelta(days=float(rng.uniform(1, self.cfg.days - 3)))
            amount = rng.uniform(40000, 200000)
            self.add(ts, victim, mules[0], amount, fraud=1, sid=sid, stype=st)
            for i in range(length):
                ts += pd.Timedelta(minutes=float(rng.uniform(10, 150)))
                amount *= rng.uniform(0.93, 0.99)
                self.add(ts, mules[i], mules[i + 1], amount, ttype="IMPS", fraud=1, sid=sid, stype=st)
            self.camouflage(mules, sid, st)

    def fanout_networks(self):
        rng = self.rng
        for k in range(self.cfg.n_fanout_networks):
            sid, st = f"FANOUT{k + 1}", "fan_out_dispersal"
            hub, recips = self.take(1)[0], self.take(int(rng.integers(12, 25)))
            self.tag([hub], "disperser_mule", sid, st, True)
            self.tag(recips, "recipient_mule", sid, st, True)
            sources = self.pick_legit(3)
            self.tag(sources, "victim_legit_account", sid, st, False)
            ts = self.start + pd.Timedelta(days=float(rng.uniform(1, self.cfg.days - 3)))
            total = 0.0
            for s in sources:
                amt = rng.uniform(80000, 250000)
                total += amt
                self.add(ts, s, hub, amt, fraud=1, sid=sid, stype=st)
                ts += pd.Timedelta(minutes=float(rng.uniform(5, 40)))
            ts += pd.Timedelta(minutes=30)
            share = total / len(recips)
            for r in recips:
                ts += pd.Timedelta(minutes=float(rng.uniform(1, 12)))
                self.add(ts, hub, r, share * rng.uniform(0.85, 0.99), ttype="IMPS", fraud=1, sid=sid, stype=st)
            self.camouflage([hub], sid, st)

    def cycle_rings(self):
        rng = self.rng
        for k in range(self.cfg.n_cycle_rings):
            sid, st = f"CYCLE{k + 1}", "circular_flow"
            members = self.take(int(rng.integers(3, 5)))
            self.tag(members, "cycle_member", sid, st, True)
            ts = self.start + pd.Timedelta(days=float(rng.uniform(1, self.cfg.days - 3)))
            amount = rng.uniform(20000, 60000)
            for lap in range(2):
                for i, m in enumerate(members):
                    ts += pd.Timedelta(minutes=float(rng.uniform(20, 120)))
                    amount *= rng.uniform(0.96, 0.995)
                    self.add(ts, m, members[(i + 1) % len(members)], amount, ttype="IMPS", fraud=1, sid=sid, stype=st)
            self.camouflage(members, sid, st)


def generate_synthetic(cfg: SyntheticConfig | None = None) -> SyntheticDataset:
    """Generate a reproducible synthetic dataset (same config + seed => identical output)."""
    cfg = cfg or SyntheticConfig()
    b = _Builder(cfg)
    b.background()
    b.hard_negatives()
    b.collection_rings()
    b.passthrough_chains()
    b.fanout_networks()
    b.cycle_rings()
    df = pd.DataFrame(b.rows).sort_values(["timestamp", "source_account", "destination_account"], kind="stable").reset_index(drop=True)
    df["transaction_id"] = [f"SYN-T{i + 1:07d}" for i in range(len(df))]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    tx = df[["transaction_id", "timestamp", "source_account", "destination_account", "amount", "transaction_type", "is_fraud"]]
    gt_tx = df[["transaction_id", "scenario_id", "scenario_type"]]
    accounts = pd.unique(pd.concat([tx.source_account, tx.destination_account]))
    rows = []
    for a in accounts:
        role, sid, st, mule = b.roles.get(a, ("legit", "", "legit_background", False))
        rows.append({"account": a, "role": role, "scenario_id": sid, "scenario_type": st, "is_mule_by_design": bool(mule)})
    return SyntheticDataset(tx, pd.DataFrame(rows), gt_tx, cfg)
