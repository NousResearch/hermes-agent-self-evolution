from evolution.code.swebench.loader import SWEInstance
from evolution.code.swebench.validity import validate_instance, Organism, Drop

def _inst(gold_patch=None):
    gp = gold_patch or ("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n"
                        "-def f(x):\n-    return x\n+def f(x):\n+    return x + 1\n")
    return SWEInstance(instance_id="flask-1", repo="pallets/flask", base_commit="b", version="2.0",
        gold_patch=gp, test_patch="", gold_file="app.py", fail_to_pass=("t::bug",),
        pass_to_pass=("t::keep",), raw={})

_BUGGY = "def f(x):\n    return x\n"
_GOLD = "def f(x):\n    return x + 1\n"

def _rep(f2p_fail, p2p_fail, eval_ok=True):
    return {"FAIL_TO_PASS": {"success": [], "failure": list(f2p_fail)},
            "PASS_TO_PASS": {"success": [], "failure": list(p2p_fail)},
            "_eval_ok": eval_ok, "_timed_out": False}

class _FakeEnv:
    """State machine: 'buggy' until apply_patch -> 'gold'; reset_file -> 'buggy'."""
    def __init__(self, buggy_rep, gold_rep, gold_src=_GOLD):
        self._b, self._g, self._gsrc = buggy_rep, gold_rep, gold_src; self.state = "buggy"
    def base_source(self, p): return _BUGGY
    def graded_report(self): return self._b if self.state == "buggy" else self._g
    def apply_patch(self, diff): self.state = "gold"
    def read_tool(self, p): return self._gsrc if self.state == "gold" else _BUGGY
    def reset_file(self, p): self.state = "buggy"

def _ok_env(gold_src=_GOLD):
    return _FakeEnv(_rep({"t::bug"}, set()), _rep(set(), set()), gold_src)

def test_valid_becomes_organism():
    org = validate_instance(_inst(), _ok_env())
    assert isinstance(org, Organism) and org.bug_tests == ("t::bug",)
    assert org.gold_loc > 0 and org.gold_hunks == 1 and org.base_src == _BUGGY
def test_bug_not_reproduced_dropped():
    env = _FakeEnv(_rep(set(), set()), _rep(set(), set()))
    d = validate_instance(_inst(), env); assert isinstance(d, Drop) and d.reason == "bug_not_reproduced"
def test_gold_unresolved_dropped():
    env = _FakeEnv(_rep({"t::bug"}, set()), _rep({"t::bug"}, set()))
    d = validate_instance(_inst(), env); assert isinstance(d, Drop) and d.reason == "gold_unresolved"
def test_gold_violating_freeze_dropped():
    gp = ("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-def f(x):\n+def renamed(x):\n")
    env = _ok_env(gold_src="def renamed(x):\n    return x + 1\n")
    d = validate_instance(_inst(gp), env); assert isinstance(d, Drop) and d.reason == "gold_violates_freeze"
def test_eval_error_dropped():
    env = _FakeEnv(_rep(set(), set(), eval_ok=False), _rep(set(), set()))
    d = validate_instance(_inst(), env); assert isinstance(d, Drop) and d.reason == "eval_error"
def test_multifile_gold_dropped():
    gp = ("diff --git a/app.py b/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x\n+y\n"
          "diff --git a/other.py b/other.py\n+++ b/other.py\n@@ -1 +1 @@\n-x\n+y\n")
    d = validate_instance(_inst(gp), _ok_env()); assert isinstance(d, Drop) and d.reason == "gold_multifile"
