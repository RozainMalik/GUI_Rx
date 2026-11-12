import pytest

from oai import batches as b
from tests.conftest import make_data


def test_corpus_order_is_a_seeded_round_robin_over_apps(tmp_path):
    root = make_data(tmp_path / "data", apps=("Alpha", "Beta", "Gamma"), n=6)
    order = b.corpus_order(root, 7)
    assert len(order) == 18 and len({(a, n) for a, n, _ in order}) == 18
    assert [f[0] for f in order] == ["Alpha", "Beta", "Gamma"] * 6
    assert order == b.corpus_order(root, 7) and order != b.corpus_order(root, 8)
    per_app = {app: [f[2] for f in order if f[0] == app] for app in ("Alpha", "Beta", "Gamma")}
    assert all(sorted(v) == [1, 2, 3, 4, 5, 6] for v in per_app.values())
    assert any(v != sorted(v) for v in per_app.values())              # a random draw, not the opening screens


def test_uneven_apps_round_robin_until_the_longest_runs_out(tmp_path):
    root = tmp_path / "data"
    make_data(root, apps=("Alpha",), n=1)
    make_data(root, apps=("Beta",), n=3)
    assert [f[0] for f in b.corpus_order(root, 1)] == ["Alpha", "Beta", "Beta", "Beta"]


def test_plan_batches_cover_each_frame_once_and_sizes_progress(tmp_path):
    root = make_data(tmp_path / "data", apps=("Alpha", "Beta"), n=5)
    order = b.corpus_order(root, 7)
    path = tmp_path / "batches.json"
    plan = b.load_plan(path, 7)
    assert plan == {"version": 1, "seed": 7, "batches": []}
    sizes = (2, 3, 4)
    made = [b.plan_batch(plan, order, b.next_size(plan, sizes)) for _ in range(4)]
    assert [x.size for x in made] == [2, 3, 4, 1] and [x.id for x in made] == [1, 2, 3, 4] and made[0].planned_at
    frames = [tuple(f[:2]) for x in made for f in x.frames]
    assert len(frames) == 10 == len(set(frames))
    with pytest.raises(b.PlanError, match="every frame"):
        b.plan_batch(plan, order, 5)
    made[0].coord_mode, made[0].efforts = "pixels", {"ground": "high"}
    b.put_batch(plan, made[0])
    b.save_plan(path, plan)
    assert b.load_plan(path, 7) == plan and b.batches(plan)[3].size == 1
    reloaded = b.get_batch(b.load_plan(path, 7), 1)
    assert reloaded.coord_mode == "pixels" and reloaded.efforts == {"ground": "high"}
    with pytest.raises(b.PlanError, match="seed"):
        b.load_plan(path, 8)
    assert b.next_size({"batches": [{}] * 7}, (25, 50)) == 50


def test_state_machine_and_refusals(tmp_path):
    root = make_data(tmp_path / "data", n=3)
    order = b.corpus_order(root, 1)
    plan = b.load_plan(tmp_path / "batches.json", 1)
    one, two = b.plan_batch(plan, order, 2), b.plan_batch(plan, order, 2)
    assert b.check_runnable(plan, 1).id == 1                              # batch 1 has no predecessor
    with pytest.raises(b.PlanError, match="not approved"):
        b.check_runnable(plan, 2)
    with pytest.raises(b.PlanError, match="grade it first"):
        b.check_approvable(one, 0.75)
    b.put_batch(plan, b.transition(one, "running"))
    b.put_batch(plan, b.transition(one, "done"))
    with pytest.raises(b.PlanError, match="cannot become approved"):
        b.transition(one, "approved")
    b.transition(one, "graded")
    with pytest.raises(b.PlanError, match="no graded rows"):
        b.check_approvable(one, 0.75)
    one.correct_rate, one.graded_rows = 0.6, 20
    with pytest.raises(b.PlanError, match="--force"):
        b.check_approvable(one, 0.75)
    b.check_approvable(one, 0.75, force=True)
    one.forced = True
    b.put_batch(plan, b.transition(one, "approved"))
    assert b.check_runnable(plan, 2).id == 2 and b.get_batch(plan, 1).state == "approved" and b.get_batch(plan, 1).forced
    with pytest.raises(b.PlanError, match="cannot become"):
        b.transition(one, "running")                                      # approved is terminal
    two.stop_reason = "budget"
    b.put_batch(plan, b.transition(two, "stopped"))
    b.plan_batch(plan, order, 2)
    with pytest.raises(b.PlanError, match="--after-stop"):
        b.check_runnable(plan, 3)
    assert b.check_runnable(plan, 3, after_stop=True).id == 3
    assert b.check_runnable(plan, 2).state == "stopped"                  # a stopped batch itself may be re-run
    with pytest.raises(b.PlanError, match="no batch 9"):
        b.get_batch(plan, 9)
    assert [x.id for x in b.approved(plan)] == [1] and b.last_measured(plan) is None
    one.usd_per_frame = 0.01
    b.put_batch(plan, one)
    assert b.last_measured(plan).id == 1


def test_ungraded_approval_needs_a_finished_batch(tmp_path):
    root = make_data(tmp_path / "data", n=3)
    plan = b.load_plan(tmp_path / "batches.json", 1)
    one = b.plan_batch(plan, b.corpus_order(root, 1), 2)
    with pytest.raises(b.PlanError, match="run it to completion first"):
        b.check_approvable(one, 0.75, ungraded=True)                     # planned: nothing to approve yet
    b.transition(one, "running")
    with pytest.raises(b.PlanError, match="run it to completion first"):
        b.check_approvable(one, 0.75, ungraded=True)
    b.transition(one, "done")
    b.check_approvable(one, 0.75, ungraded=True)                         # done with no grades: the caller takes the risk


def test_corpus_order_can_skip_apps_without_reordering_the_rest(tmp_path):
    root = make_data(tmp_path / "data", apps=("Alpha", "RadiAnt", "Gamma"), n=4)
    full = b.corpus_order(root, 7)
    kept = b.corpus_order(root, 7, skip=lambda app: app == "RadiAnt")
    assert {f[0] for f in kept} == {"Alpha", "Gamma"} and len(kept) == 8
    assert kept == [f for f in full if f[0] != "RadiAnt"]               # same seeded order, one app removed


def test_a_batch_records_its_transport_and_old_rows_default_to_sync():
    assert b.Batch(id=1, size=1, frames=[]).transport == "sync"
    row = {"id": 1, "size": 1, "frames": [], "state": "done"}                      # a row written before the field existed
    assert b.Batch(**row).transport == "sync"
    assert b.Batch(**{**row, "transport": "batch"}).transport == "batch"


def test_balanced_batches_take_an_equal_share_of_each_apps_remaining_frames(tmp_path):
    root = tmp_path / "data"
    make_data(root, apps=("Alpha",), n=3)
    make_data(root, apps=("Beta", "Gamma"), n=10)
    orders = b.app_orders(root, 7)
    assert list(orders) == ["Alpha", "Beta", "Gamma"] and [len(v) for v in orders.values()] == [3, 10, 10]
    assert b.corpus_order(root, 7) == b.interleave(orders)
    plan = b.load_plan(tmp_path / "batches.json", 7)
    first = b.plan_batch_balanced(plan, orders, 20)
    per_app = {app: [f for f in first.frames if f[0] == app] for app in orders}
    assert [len(v) for v in per_app.values()] == [3, 9, 8]                          # round-robin over what each app still has
    assert per_app["Beta"] == orders["Beta"][:9] and per_app["Gamma"] == orders["Gamma"][:8]   # seeded order preserved
    second = b.plan_batch_balanced(plan, orders, 20)
    assert second.size == 3 and {f[0] for f in second.frames} == {"Beta", "Gamma"}  # Alpha is exhausted
    assert len({tuple(f[:2]) for x in (first, second) for f in x.frames}) == 23
    with pytest.raises(b.PlanError, match="every frame"):
        b.plan_batch_balanced(plan, orders, 1)


def test_balanced_planning_after_unbalanced_batches_never_moves_a_planned_frame(tmp_path):
    root = make_data(tmp_path / "data", apps=("Alpha", "Beta"), n=4)
    orders = b.app_orders(root, 7)
    plan = b.load_plan(tmp_path / "batches.json", 7)
    old = b.plan_batch(plan, b.corpus_order(root, 7), 3)
    new = b.plan_batch_balanced(plan, orders, 3)
    assert not ({tuple(f[:2]) for f in old.frames} & {tuple(f[:2]) for f in new.frames})
    assert [f[0] for f in new.frames] == ["Alpha", "Beta", "Alpha"]
