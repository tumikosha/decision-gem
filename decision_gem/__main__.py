"""Answer one Jev request ({"state", "questions"}) or one prompt ({"prompt", "state" optional, "max_tokens"}) and
print the response.

    python -m decision_gem request.json
    cat request.json | python -m decision_gem - --calibration cal.json
"""
import argparse
import json
import sys

from .core import MODEL, ORDERS, DecisionGem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", help="the request, JSON file or - for stdin")
    ap.add_argument("--model", default=MODEL, help="HF repo id or local model folder")
    ap.add_argument("--calibration", help="JSON {question id: [a, b] (noul) or T (choice / score)}")
    ap.add_argument("--orders", choices=ORDERS, default="reverse",
                    help="option orders: in order and reversed (2 passes), every rotation (n passes) "
                         "or in order only (1 pass, fastest, no bias cancelling)")
    ap.add_argument("--online", action="store_true", help="allow downloading the model")
    ap.add_argument("--raw", action="store_true",
                    help="add the uncalibrated option log-probabilities to every answer (input of fit_calibration)")
    args = ap.parse_args()
    request = json.load(sys.stdin if args.request == "-" else open(args.request, encoding="utf-8"))
    calibration = json.load(open(args.calibration, encoding="utf-8")) if args.calibration else None
    gem = DecisionGem(args.model, calibration, offline=not args.online, orders=args.orders)
    if args.raw:
        request = {**request, "raw": True}
    print(json.dumps(gem(request), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
