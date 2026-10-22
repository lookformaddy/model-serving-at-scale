"""CLI: model-serving demo | predict | models | serve"""
from __future__ import annotations

import argparse
import json


def _demo(args) -> int:
    from .demo import run_demo
    run_demo(); return 0

def _predict(args) -> int:
    from .models import PredictRequest
    from .service import build_demo_service
    svc = build_demo_service()
    feats = tuple(float(x) for x in args.features.split(","))
    resp = svc.predict(PredictRequest(args.request_id, feats, model_name=args.model, user_id=args.user))
    print(json.dumps(resp.to_dict(), indent=2)); return 0

def _models(args) -> int:
    from .registry import build_demo_registry
    reg = build_demo_registry()
    for n in reg.list_models():
        print(f"{n}: versions={reg.list_versions(n)} active={reg.active_version(n)}")
    return 0

def _serve(args) -> int:
    import uvicorn
    from .service import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port); return 0

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="model-serving", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo"); d.set_defaults(func=_demo)
    pr = sub.add_parser("predict"); pr.add_argument("--features", default="0.9,0.2,0.7,0.1")
    pr.add_argument("--model", default="churn"); pr.add_argument("--user", default="user-1")
    pr.add_argument("--request-id", default="req-001"); pr.set_defaults(func=_predict)
    m = sub.add_parser("models"); m.set_defaults(func=_models)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000); s.set_defaults(func=_serve)
    args = p.parse_args(argv); return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
