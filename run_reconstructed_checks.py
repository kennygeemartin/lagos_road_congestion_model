"""Run the NEW study's complete verification gate, preserving legacy audit failures.

Legacy tests/test_protocol.py documents why the original archive could not be
locked; it is not the reconstructed-v1 training gate. No failed check is changed
to expected-failure or silently reported as passing.
"""
import io
import json
import sys
import unittest
from pathlib import Path
import torch
from reconstructed_study import ROOT,STUDY
from validate_reconstructed_results import CompletedStudy

def main():
    torch.set_num_threads(4)
    suite=unittest.TestSuite()
    loader=unittest.TestLoader()
    for pattern in ['test_controls.py','test_evaluation.py','test_models.py',
                    'test_reconstructed_protocol.py','test_reconstructed_training.py']:
        suite.addTests(loader.discover(str(ROOT/'tests'),pattern=pattern))
    suite.addTests(loader.loadTestsFromTestCase(CompletedStudy))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    output=STUDY/'results';output.mkdir(parents=True,exist_ok=True)
    (output/'new_study_checks.log').write_text(stream.getvalue(),encoding='utf-8')
    status=dict(scope='new reconstructed-v1 scientific/implementation gate',total=result.testsRun,
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        legacy_audit='Original archive prerequisite failures remain separately recorded; not part of this new study gate')
    (output/'new_study_check_summary.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    print(stream.getvalue());print(json.dumps(status,indent=2),flush=True)
    if not result.wasSuccessful() or result.skipped:raise SystemExit(1)

if __name__=='__main__':main()
