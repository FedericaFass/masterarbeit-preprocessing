
from ppm_preprocessing.inference import load_bundle, predict_running_case, predict_batch

BUNDLE_PATH = "outputs/single_task/model_bundle.joblib"


def _patch_bundle(bundle):
    """Patch old bundles that are missing target_log1p (pre-fix bundles)."""
    if "target_log1p" not in bundle:
        print("  [patch] Old bundle detected — setting target_log1p=True")
        bundle["target_log1p"] = True
    return bundle


def test_single_prediction():
    """Test predicting remaining time for a single running case (TEST SET)."""
    print("=" * 60)
    print("Loading model bundle...")
    bundle = load_bundle(BUNDLE_PATH)
    bundle = _patch_bundle(bundle)

    print(f"  strategy    : {bundle['strategy']}")
    print(f"  models      : {list(bundle['models'].keys())}")
    print(f"  encoder     : {type(bundle['encoder']).__name__}")
    print(f"  bucketer    : {type(bundle['bucketer']).__name__}")
    print(f"  target_log1p: {bundle.get('target_log1p')}")
    print()

    # TEST SET case: declaration 100010 (NOT in training data)
    running_case = [
        {
            "activity": "Declaration SUBMITTED by EMPLOYEE",
            "timestamp": "2018-01-30 09:43:21",
            "case:BudgetNumber": "budget 86566",
            "case:DeclarationNumber": "declaration number 100011",
            "case:Amount": 26.19,
            "org:resource": "STAFF MEMBER",
            "org:role": "EMPLOYEE",
        },
        {
            "activity": "Declaration APPROVED by ADMINISTRATION",
            "timestamp": "2018-01-30 09:43:23",
            "org:resource": "STAFF MEMBER",
            "org:role": "ADMINISTRATION",
        },
        {
            "activity": "Declaration FINAL_APPROVED by SUPERVISOR",
            "timestamp": "2018-01-30 10:03:19",
            "org:resource": "STAFF MEMBER",
            "org:role": "SUPERVISOR",
        },
    ]

    print("Test 1: Predict test case 'declaration 100010' with 3 events")
    result = predict_running_case(bundle, running_case, case_id="declaration 100010")
    print(f"  case_id        : {result['case_id']}")
    print(f"  prefix_len     : {result['prefix_len']}")
    print(f"  bucket_id      : {result['bucket_id']}")
    print(f"  predicted (sec): {result['predicted_remaining_time_sec']:,.1f}")
    print(f"  predicted (days): {result['predicted_remaining_time_days']:.2f}")
    print()

    # Ground truth from CSV: at prefix_len=3, remaining = 199,678 sec (~2.31 days)
    actual_remaining = 199678.0
    error = abs(result["predicted_remaining_time_sec"] - actual_remaining)
    print(f"  actual remaining (sec) : {actual_remaining:,.1f}")
    print(f"  absolute error (sec)   : {error:,.1f}")
    print(f"  absolute error (days)  : {error / 86400:.2f}")
    print()


def test_different_prefix_lengths():
    """Test with growing prefix lengths on a TEST SET case."""
    print("=" * 60)
    print("Test 2: Predictions at different prefix lengths (test case)")
    print("=" * 60)
    bundle = load_bundle(BUNDLE_PATH)
    bundle = _patch_bundle(bundle)

    # TEST SET case: declaration 100037
    all_events = [
        {
            "activity": "Declaration SUBMITTED by EMPLOYEE",
            "timestamp": "2018-02-01 14:16:11",
            "case:BudgetNumber": "budget 86566",
            "case:DeclarationNumber": "declaration number 100038",
            "case:Amount": 47.90,
            "org:resource": "STAFF MEMBER",
            "org:role": "EMPLOYEE",
        },
        {
            "activity": "Declaration APPROVED by ADMINISTRATION",
            "timestamp": "2018-02-01 14:17:38",
            "org:resource": "STAFF MEMBER",
            "org:role": "ADMINISTRATION",
        },
        {
            "activity": "Declaration FINAL_APPROVED by SUPERVISOR",
            "timestamp": "2018-02-01 16:47:22",
            "org:resource": "STAFF MEMBER",
            "org:role": "SUPERVISOR",
        },
        {
            "activity": "Request Payment",
            "timestamp": "2018-02-06 21:17:25",
            "org:resource": "SYSTEM",
            "org:role": "UNDEFINED",
        },
    ]

    # Ground truth remaining times from the CSV (test case declaration 100037)
    actual_remaining = {
        1: 616499.0,   # ~7.13 days
        2: 616412.0,   # ~7.13 days
        3: 607428.0,   # ~7.03 days
        4: 159225.0,   # ~1.84 days
    }

    print(f"  Case: declaration 100037 (test set)")
    print(f"  {'PrefLen':>7} {'Bucket':>6} {'Pred(days)':>10} {'Actual(days)':>12} {'Error(days)':>11}")
    print(f"  {'-'*50}")

    for n in range(1, len(all_events) + 1):
        prefix = all_events[:n]
        result = predict_running_case(bundle, prefix, case_id="declaration 100037")
        pred_days = result["predicted_remaining_time_days"]
        act_days = actual_remaining[n] / 86400.0
        err_days = abs(pred_days - act_days)
        print(f"  {n:>7} {result['bucket_id']:>6} {pred_days:>10.2f} {act_days:>12.2f} {err_days:>11.2f}")

    print()


def test_batch_prediction():
    """Test batch prediction on multiple TEST SET cases."""
    print("=" * 60)
    print("Test 3: Batch prediction (test cases)")
    print("=" * 60)
    bundle = load_bundle(BUNDLE_PATH)
    bundle = _patch_bundle(bundle)

    # All cases from the test set with ground truth
    cases = {
        # declaration 100010: prefix_len=2, actual remaining = 200,874 sec
        "declaration 100010": [
            {"activity": "Declaration SUBMITTED by EMPLOYEE", "timestamp": "2018-01-30 09:43:21",
             "case:BudgetNumber": "budget 86566", "case:Amount": 26.19,
             "org:resource": "STAFF MEMBER", "org:role": "EMPLOYEE"},
            {"activity": "Declaration APPROVED by ADMINISTRATION", "timestamp": "2018-01-30 09:43:23",
             "org:resource": "STAFF MEMBER", "org:role": "ADMINISTRATION"},
        ],
        # declaration 100037: prefix_len=1, actual remaining = 616,499 sec
        "declaration 100037": [
            {"activity": "Declaration SUBMITTED by EMPLOYEE", "timestamp": "2018-02-01 14:16:11",
             "case:BudgetNumber": "budget 86566", "case:Amount": 47.90,
             "org:resource": "STAFF MEMBER", "org:role": "EMPLOYEE"},
        ],
        # declaration 100027: prefix_len=3, actual remaining = 638,375 sec
        "declaration 100027": [
            {"activity": "Declaration SUBMITTED by EMPLOYEE", "timestamp": "2018-02-01 09:20:56",
             "case:BudgetNumber": "budget 86566", "case:Amount": 68.08,
             "org:resource": "STAFF MEMBER", "org:role": "EMPLOYEE"},
            {"activity": "Declaration APPROVED by ADMINISTRATION", "timestamp": "2018-02-01 09:26:22",
             "org:resource": "STAFF MEMBER", "org:role": "ADMINISTRATION"},
            {"activity": "Declaration FINAL_APPROVED by SUPERVISOR", "timestamp": "2018-02-05 08:11:45",
             "org:resource": "STAFF MEMBER", "org:role": "SUPERVISOR"},
        ],
    }

    results = predict_batch(bundle, cases)
    for r in results:
        print(f"  {r['case_id']:<25} prefix_len={r['prefix_len']}  "
              f"bucket={r['bucket_id']}  "
              f"predicted={r['predicted_remaining_time_days']:.2f} days")
    print()


if __name__ == "__main__":
    test_single_prediction()
    test_different_prefix_lengths()
    test_batch_prediction()
    print("All tests passed!")
