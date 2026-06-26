from evolution.code.swebench.loader import files_in_patch, is_single_file, to_instance, patch_loc, patch_hunks

_SINGLE = {
    "instance_id": "pallets__flask-1", "repo": "pallets/flask", "base_commit": "abc",
    "patch": "diff --git a/src/flask/app.py b/src/flask/app.py\n--- a/src/flask/app.py\n+++ b/src/flask/app.py\n@@ -1 +1 @@\n-x\n+y\n",
    "test_patch": "diff --git a/tests/test_app.py b/tests/test_app.py\n+++ b/tests/test_app.py\n@@ -1 +1 @@\n-a\n+b\n",
    "FAIL_TO_PASS": "[\"tests/test_app.py::test_x\"]", "PASS_TO_PASS": "[\"tests/test_app.py::test_y\"]",
    "version": "2.0", "environment_setup_commit": "abc", "problem_statement": "bug",
}
_MULTI = {**_SINGLE, "instance_id": "m",
    "patch": "diff --git a/src/flask/app.py b/src/flask/app.py\n+++ b/src/flask/app.py\n@@ -1 +1 @@\n-x\n+y\n"
             "diff --git a/src/flask/cli.py b/src/flask/cli.py\n+++ b/src/flask/cli.py\n@@ -1 +1 @@\n-x\n+y\n"}

def test_files_in_patch_excludes_devnull():
    assert files_in_patch(_SINGLE["patch"]) == ["src/flask/app.py"]
def test_single_file_true_false():
    assert is_single_file(_SINGLE) is True and is_single_file(_MULTI) is False
def test_to_instance_parses_lists_and_keeps_raw():
    inst = to_instance(_SINGLE)
    assert inst.fail_to_pass == ("tests/test_app.py::test_x",)
    assert inst.gold_file == "src/flask/app.py"
    assert inst.raw is _SINGLE  # raw row retained for make_test_spec
def test_loc_and_hunks():
    assert patch_loc(_SINGLE["patch"]) == 2  # one +, one -
    assert patch_hunks(_SINGLE["patch"]) == 1
