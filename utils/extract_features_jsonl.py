#!/usr/bin/env python3
# utils/extract_features_jsonl.py
import argparse, json, pickle
from pathlib import Path
from core.guardian_utils import FeatureExtractor

def load_jsonl(p):
    items=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="03_new_features.pkl")
    args=parser.parse_args()

    items = load_jsonl(args.input)
    if not items:
        print("No items to extract")
        return

    extractor = FeatureExtractor()
    B=64
    processed=[]
    for i in range(0,len(items),B):
        batch = items[i:i+B]
        qs=[it.get("question","") for it in batch]
        as_=[it.get("answer","") for it in batch]
        feats = extractor.extract(qs, as_)
        for j,f in enumerate(feats):
            processed.append({"features": f, "meta": batch[j]})
    with open(args.out,"wb") as f:
        pickle.dump(processed,f)
    print(f"Wrote {len(processed)} feature items -> {args.out}")

if __name__=="__main__":
    main()
